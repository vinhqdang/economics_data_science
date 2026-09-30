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
from gdma.storage import value_curves, allocate_budget, optimal_sizes, allocate_with_floors, allocate_maximin  # noqa: E402

TAG = os.environ.get("TAG", "global_w28_taucalibrated")
SUFFIX = "" if TAG == "global_w28_taucalibrated" else "_" + TAG.replace("global_w28_taucalibrated_", "")
IDLE_USD = 10.0                 # $ per MWh of idle committed capacity
BATTERY_USD = [100e3, 150e3, 200e3]   # annualised $ per MW of 4-h battery power
FRACS = np.array([0, .0025, .005, .01, .015, .02, .03, .04, .05, .06, .08, .10, .12, .15])
BUDGETS_GW = [1, 5, 10, 20]
POLICIES = ["Official", "EW", "Pooled", "PerUnit", "FTO-PerUnit", "GDMA", "GDMA-soft", "GDMA-fair"]
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
    POL = [m for m in POLICIES if m in omethods]
    for m in POL:
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
    for m in POL:
        for cost in BATTERY_USD:
            Pm = optimal_sizes(grid, curves[m] * IDLE_USD, cost)
            gain = np.array([np.interp(Pm[i], grid[i], curves[m][i]) for i in range(N)]) * IDLE_USD - cost * Pm
            for c in CONTS:
                k = (cont == c) & act
                rows.append(dict(policy=m, battery_usd=cost, continent=c, gw=Pm[k].sum() / 1e3,
                                 net_benefit_musd=gain[k].sum() / 1e6, areas=int((Pm[k] > 0).sum()),
                                 n_areas=int(k.sum())))
    opt = pd.DataFrame(rows)
    opt.to_csv(os.path.join(ROOT, "results", f"storage_optimal{SUFFIX}.csv"), index=False)

    # (b) budget allocation and (d) misallocation
    rows = []
    for B in BUDGETS_GW:
        alloc = {m: allocate_budget(grid[act], curves[m][act], B * 1e3) for m in POL}
        for m in POL:
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
    bud.to_csv(os.path.join(ROOT, "results", f"storage_budget{SUFFIX}.csv"), index=False)

    # (c) storage-equivalent of better forecasting
    target = base["GDMA-soft"][act].sum()
    rows = []
    for m in POL:
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
        np.save(os.path.join(ROOT, "results", f"storage_frontier{SUFFIX}_{m}.npy"), np.vstack([Bs, costs * IDLE_USD / 1e6]))
    eq = pd.DataFrame(rows)
    eq.to_csv(os.path.join(ROOT, "results", f"storage_equivalent{SUFFIX}.csv"), index=False)

    # (e) fair siting: every continent receives at least lambda times its share of
    # demand in the budget; price of fairness = value lost relative to efficiency
    dem = np.nan_to_num(scale) * act
    share = {c: dem[cont == c].sum() / dem.sum() for c in CONTS}
    rows = []
    Va, ga, ca = curves["GDMA-soft"][act], grid[act], cont[act]
    for B in BUDGETS_GW:
        for lam in [0.0, 0.5, 1.0]:
            floors = {c: lam * share[c] * B * 1e3 for c in CONTS}
            Pm = allocate_with_floors(ga, Va, B * 1e3, ca, floors)
            val = np.array([np.interp(Pm[i], ga[i], Va[i]) for i in range(len(Pm))])
            row = dict(budget_gw=B, floor_lambda=lam, value_musd=val.sum() * IDLE_USD / 1e6)
            for c in CONTS:
                row[f"gw_{c}"] = Pm[ca == c].sum() / 1e3
                row[f"value_{c}"] = val[ca == c].sum() * IDLE_USD / 1e6
            rows.append(row)
        # Rawlsian: maximise the minimum relative benefit across continents
        bc = {c: base["GDMA-soft"][act][ca == c].sum() for c in CONTS if (ca == c).any()}
        Pm = allocate_maximin(ga, Va, B * 1e3, ca, bc)
        val = np.array([np.interp(Pm[i], ga[i], Va[i]) for i in range(len(Pm))])
        row = dict(budget_gw=B, floor_lambda=-1.0, value_musd=val.sum() * IDLE_USD / 1e6)
        for c in CONTS:
            row[f"gw_{c}"] = Pm[ca == c].sum() / 1e3
            row[f"value_{c}"] = val[ca == c].sum() * IDLE_USD / 1e6
        rows.append(row)
    fair = pd.DataFrame(rows)
    # relative benefit of each continent: value captured / its cost without storage
    for c in CONTS:
        bc_c = base["GDMA-soft"][act][ca == c].sum() * IDLE_USD / 1e6
        fair[f"relben_{c}"] = fair[f"value_{c}"] / max(bc_c, 1e-12)
    base_v = fair[fair.floor_lambda == 0].set_index("budget_gw").value_musd
    fair["price_of_fairness"] = 1 - fair.value_musd / fair.budget_gw.map(base_v)
    fair.to_csv(os.path.join(ROOT, "results", f"storage_fair{SUFFIX}.csv"), index=False)
    pd.Series(share).to_csv(os.path.join(ROOT, "results", f"storage_demand_share{SUFFIX}.csv"))
    print(fair.round(3).to_string())

    # (f) out-of-sample siting: value curves from the first half of the
    # evaluation period decide where the batteries go; their value is measured
    # on the second half and compared with siting on second-half curves (oracle)
    tdays = orders.shape[2]
    halves = [np.arange(0, tdays // 2), np.arange(tdays // 2, tdays)]

    def half_curves(m, hd):
        q = orders[omethods.index(m)][:, hd]
        v = valid[:, hd]
        yz, qz = np.where(v, Y[:, hd], 0.0), np.where(v, q, 0.0)
        yrs = np.maximum(v.reshape(N, -1).sum(axis=1) / 8760.0, 1e-9)
        Vh = value_curves(np.maximum(yz - qz, 0).reshape(N, -1), np.maximum(qz - yz, 0).reshape(N, -1),
                          u, grid, years=1.0) / yrs[:, None]
        ok = (v.reshape(N, -1).sum(axis=1) / 8760.0) > 0.25
        return np.where(ok[:, None], Vh, 0.0), ok

    V1 = {m: half_curves(m, halves[0]) for m in ["Official", "GDMA-soft"]}
    V2, ok2 = half_curves("GDMA-soft", halves[1])
    rows = []
    for B in BUDGETS_GW:
        oracle = allocate_budget(grid, V2, B * 1e3)
        v_or = sum(np.interp(oracle[i], grid[i], V2[i]) for i in range(N))
        row = dict(budget_gw=B, oracle_musd=v_or * IDLE_USD / 1e6)
        for m in ["GDMA-soft", "Official"]:
            Pm = allocate_budget(grid, V1[m][0], B * 1e3)
            v = sum(np.interp(Pm[i], grid[i], V2[i]) for i in range(N))
            row[f"sited_on_{m}_musd"] = v * IDLE_USD / 1e6
        rows.append(row)
    oos = pd.DataFrame(rows)
    oos.to_csv(os.path.join(ROOT, "results", f"storage_oos{SUFFIX}.csv"), index=False)
    print("out-of-sample siting (value in the second half under soft GDMA operations)")
    print(oos.round(1).to_string())

    # per-area optimal sizes under soft GDMA vs status quo (for the map/table)
    per = pd.DataFrame(dict(code=code, continent=cont, mean_mw=scale, years=years, tau=r["tau"],
                            mw_status_quo=optimal_sizes(grid, curves["Official"] * IDLE_USD, 150e3),
                            mw_soft=optimal_sizes(grid, curves["GDMA-soft"] * IDLE_USD, 150e3)))
    # marginal value (USD per MW-year) of the first battery increment
    per["mv1_status_quo"] = curves["Official"][:, 1] / np.maximum(grid[:, 1], 1e-9) * IDLE_USD
    per["mv1_soft"] = curves["GDMA-soft"][:, 1] / np.maximum(grid[:, 1], 1e-9) * IDLE_USD
    per.to_csv(os.path.join(ROOT, "results", f"storage_per_area{SUFFIX}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(opt[opt.battery_usd == 150e3].pivot(index="policy", columns="continent", values="gw").round(2))
    print(bud.round(1).to_string())
    print(eq.round(2).to_string())


if __name__ == "__main__":
    main()
