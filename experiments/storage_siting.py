"""Stage 1: where to install batteries, given the learned commitment policies.

Reads results/orders_<tag>.npz and backtest_<tag>.npz from backtest_global.py.
For every commitment policy and grid area it computes the annual value of
battery power (4-hour, 85% round trip), then
  (a) the cost-optimal battery size of every area,
  (b) the allocation of a global storage budget across areas,
  (c) the storage-equivalent of better forecasting: the battery power the
      status-quo policy needs to match the cost of soft GDMA without storage,
  (d) the cost of siting batteries with the wrong policy's error profile.
Money: one cost unit is the cost of one MWh of committed-but-idle capacity,
valued at IDLE_USD dollars; batteries cost BATTERY_USD dollars per MW-year.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from gdma.storage import value_curves, allocate_budget, optimal_sizes  # noqa: E402

TAG = os.environ.get("TAG", "global_w28_taucalibrated")
IDLE_USD = 10.0                 # $ per MWh of idle committed capacity
BATTERY_USD = [100e3, 150e3, 200e3]   # annualised $ per MW of 4-h battery power
FRACS = np.array([0, .0025, .005, .01, .015, .02, .03, .04, .05, .06, .08, .10, .12, .15])
BUDGETS_GW = [1, 5, 10, 20]
POLICIES = ["Official", "EW", "Pooled", "PerUnit", "FTO-PerUnit", "GDMA", "GDMA-soft"]
LABEL = {"Official": "Status quo", "EW": "Equal weights", "Pooled": "Pooled DF", "PerUnit": "Per-unit DF",
         "FTO-PerUnit": "Forecast-then-commit", "GDMA": "GDMA", "GDMA-soft": "Soft GDMA",
         "Shrink": "Shrinkage"}
CONTS = ["North America", "Europe", "Oceania", "Asia", "Africa"]


def main():
    r = np.load(os.path.join(ROOT, "results", f"backtest_{TAG}.npz"), allow_pickle=True)
    od = np.load(os.path.join(ROOT, "results", f"orders_{TAG}.npz"), allow_pickle=True)
    pn = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    ev = r["eval_days"]
    Y = pn["Y"][:, ev, :].astype(float)
    orders = od["orders"].astype(float)
    omethods = list(od["methods"])
    u, o = r["u"], r["o"]
    cont = r["continent"]
    code = r["code"]
    N = len(u)
    valid = np.isfinite(Y) & np.isfinite(orders).all(axis=0)
    years = valid.reshape(N, -1).sum(axis=1) / 8760.0
    scale = r["scale"]
    act = years > 0.25                      # at least three months of evaluation data
    grid = np.outer(scale, FRACS)

    curves, base = {}, {}
    for m in POLICIES:
        q = orders[omethods.index(m)]
        yz = np.where(valid, Y, 0.0)
        qz = np.where(valid, q, 0.0)
        short = np.maximum(yz - qz, 0).reshape(N, -1)
        surp = np.maximum(qz - yz, 0).reshape(N, -1)
        yrs = np.maximum(years, 1e-9)
        V = value_curves(short, surp, u, grid, years=1.0) / yrs[:, None]     # cost units per year
        V[~act] = 0.0
        curves[m] = V
        base[m] = (u * short.sum(1) + o * surp.sum(1)) / yrs                  # annual cost, no storage
        base[m][~act] = 0.0
        print("value curves", m, flush=True)

    rows = []
    for m in POLICIES:
        for cost in BATTERY_USD:
            Pm = optimal_sizes(grid, curves[m] * IDLE_USD, cost)
            gain = np.array([np.interp(Pm[i], grid[i], curves[m][i]) for i in range(N)]) * IDLE_USD - cost * Pm
            for c in CONTS:
                k = (cont == c) & act
                rows.append(dict(policy=m, battery_usd=cost, continent=c, gw=Pm[k].sum() / 1e3,
                                 net_benefit_musd=gain[k].sum() / 1e6, areas=int((Pm[k] > 0).sum()),
                                 n_areas=int(k.sum())))
    opt = pd.DataFrame(rows)
    opt.to_csv(os.path.join(ROOT, "results", "storage_optimal.csv"), index=False)

    # (b) budget allocation and (d) misallocation
    rows = []
    for B in BUDGETS_GW:
        alloc = {m: allocate_budget(grid[act], curves[m][act], B * 1e3) for m in POLICIES}
        for m in POLICIES:
            Pm = alloc[m]
            V = curves[m][act]
            g = grid[act]
            val = np.array([np.interp(Pm[i], g[i], V[i]) for i in range(len(Pm))])
            # the same siting evaluated under soft GDMA operations
            Vg = curves["GDMA-soft"][act]
            val_g = np.array([np.interp(Pm[i], g[i], Vg[i]) for i in range(len(Pm))])
            row = dict(budget_gw=B, policy=m, value_musd=val.sum() * IDLE_USD / 1e6,
                       value_under_soft_musd=val_g.sum() * IDLE_USD / 1e6,
                       system_cost_musd=(base[m][act].sum() - val.sum()) * IDLE_USD / 1e6)
            for c in CONTS:
                row[f"gw_{c}"] = Pm[cont[act] == c].sum() / 1e3
            rows.append(row)
    bud = pd.DataFrame(rows)
    bud.to_csv(os.path.join(ROOT, "results", "storage_budget.csv"), index=False)

    # (c) storage-equivalent of better forecasting
    target = base["GDMA-soft"][act].sum()
    rows = []
    for m in POLICIES:
        Bs = np.concatenate([np.linspace(0, 5, 21), np.linspace(6, 60, 55)])
        costs = []
        for B in Bs:
            Pm = allocate_budget(grid[act], curves[m][act], B * 1e3)
            val = np.array([np.interp(Pm[i], grid[act][i], curves[m][act][i]) for i in range(len(Pm))])
            costs.append(base[m][act].sum() - val.sum())
        costs = np.array(costs)
        reach = np.flatnonzero(costs <= target)
        rows.append(dict(policy=m, cost_no_storage_musd=base[m][act].sum() * IDLE_USD / 1e6,
                         gw_to_match_soft=Bs[reach[0]] if len(reach) else np.inf,
                         min_cost_musd=costs.min() * IDLE_USD / 1e6))
        np.save(os.path.join(ROOT, "results", f"storage_frontier_{m}.npy"), np.vstack([Bs, costs * IDLE_USD / 1e6]))
    eq = pd.DataFrame(rows)
    eq.to_csv(os.path.join(ROOT, "results", "storage_equivalent.csv"), index=False)

    # per-area optimal sizes under soft GDMA vs status quo (for the map/table)
    per = pd.DataFrame(dict(code=code, continent=cont, mean_mw=scale, years=years, tau=r["tau"],
                            mw_status_quo=optimal_sizes(grid, curves["Official"] * IDLE_USD, 150e3),
                            mw_soft=optimal_sizes(grid, curves["GDMA-soft"] * IDLE_USD, 150e3)))
    # marginal value (USD per MW-year) of the first battery increment
    per["mv1_status_quo"] = curves["Official"][:, 1] / np.maximum(grid[:, 1], 1e-9) * IDLE_USD
    per["mv1_soft"] = curves["GDMA-soft"][:, 1] / np.maximum(grid[:, 1], 1e-9) * IDLE_USD
    per.to_csv(os.path.join(ROOT, "results", "storage_per_area.csv"), index=False)
    pd.set_option("display.width", 200)
    print(opt[opt.battery_usd == 150e3].pivot(index="policy", columns="continent", values="gw").round(2))
    print(bud.round(1).to_string())
    print(eq.round(2).to_string())


if __name__ == "__main__":
    main()
