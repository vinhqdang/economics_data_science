"""Extreme events and extreme-temperature days.

1. Named events on the U.S. balancing authorities, relative to the operators'
   official day-ahead forecasts: Winter Storm Uri (10-20 February 2021) and
   Winter Storm Elliott (21-26 December 2022, the window of the FERC-NERC
   inquiry), for each candidate set, with and without the BA-days on which load
   was shed (served demand then understates true demand): ERCOT 15-18 February
   2021, SPP and MISO 15-16 February 2021, TVA and Duke (DUK, CPLE) 23-24
   December 2022.
2. Extreme-temperature days, defined by a rule fixed before looking at costs:
   the area-days whose daily mean temperature (NASA POWER) lies below the
   area's 2.5% or above its 97.5% quantile over the evaluation period.  Cost
   ratios relative to the reference forecaster, with moving-block bootstrap
   intervals over days.
3. The cost of each weather candidate's own commitment policy.
Writes paper/tables/global_elliott.tex and paper/tables/global_extremes.tex.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from mcs import ratio_ci  # noqa: E402

EVENTS = {"Uri": ("2021-02-10", "2021-02-20"), "Elliott": ("2022-12-21", "2022-12-26")}
SHED = [("US-ERCO", "2021-02-15", "2021-02-18"), ("US-SWPP", "2021-02-15", "2021-02-16"),
        ("US-MISO", "2021-02-15", "2021-02-16"), ("US-TVA", "2022-12-23", "2022-12-24"),
        ("US-DUK", "2022-12-23", "2022-12-24"), ("US-CPLE", "2022-12-23", "2022-12-24")]
SPECS = ["", "_weather-gfs", "_weather-plus", "_weather-exact"]
MAIN = "_weather-plus"
LABEL = {"GDMA-soft": r"\textbf{Soft GDMA}", "GDMA-fair": "Fair soft GDMA", "GDMA": "GDMA",
         "KMeans2S": "Two-step $k$-means", "Shrink": "Shrinkage to pooled", "PerUnit": "Per-area DF weights",
         "Pooled": "Pooled DF weights", "Select": "Best single (per area)", "QR": "Quantile-regression comb.",
         "FTO-Grouped": "Forecast-then-commit, grouped"}
ALONE = {"Weather": "Temperature model", "WeatherPlus": "Multi-variable model",
         "WeatherEns": "Multi-variable, ensemble margin"}


def load(suffix):
    p = os.path.join(ROOT, "results", f"backtest_global_w28_taucalibrated{suffix}.npz")
    if not os.path.exists(p):
        return None
    r = dict(np.load(p, allow_pickle=True))
    ng = os.path.join(ROOT, "results", f"neuralgate_global_w28_taucalibrated{suffix}.npz")
    if os.path.exists(ng):
        g = np.load(ng, allow_pickle=True)
        r["daily_cost"] = np.concatenate([r["daily_cost"], g["daily_cost"]])
        r["methods"] = np.concatenate([r["methods"], g["methods"]])
    return r


def calendar(r):
    cal = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)["days"]
    return pd.to_datetime(cal[r["eval_days"]])


def shed_mask(r, days):
    """(N, days) True where load was shed."""
    m = np.zeros((len(r["code"]), len(days)), bool)
    code = list(r["code"])
    for c, a, b in SHED:
        if c in code:
            m[code.index(c)] |= (days >= a) & (days <= b)
    return m


def event_ratios(r, window, exclude_shed=False):
    days = calendar(r)
    sel = (days >= window[0]) & (days <= window[1])
    us = r["continent"] == "North America"
    c = r["daily_cost"][:, us][:, :, sel]
    ok = np.isfinite(c).all(axis=0)
    if exclude_shed:
        ok &= ~shed_mask(r, days)[us][:, sel]
    tot = np.where(ok, c, 0.0).sum(axis=(1, 2))
    m = list(r["methods"])
    return pd.Series(tot / tot[m.index("Official")], index=m)


def standalone(suffix, window):
    from gdma.core import newsvendor_cost
    r = load(suffix)
    cand = [str(c) for c in r["candidates"]]
    pp = os.path.join(ROOT, "data", "processed", f"global_policies_{len(cand)}_calibrated{suffix}.npy")
    if not os.path.exists(pp):
        return None
    P = np.load(pp, mmap_mode="r")
    pn = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    ev = r["eval_days"]
    Y = pn["Y"][:, ev, :].astype(float)
    u, o = r["u"], r["o"]
    valid = np.isfinite(Y) & np.isfinite(np.asarray(P[:, :, ev, :])).all(axis=0)
    days = calendar(r)
    E = (days >= window[0]) & (days <= window[1])
    us = r["continent"] == "North America"
    tot, tE = {}, {}
    for j, c in enumerate(cand):
        cost = np.where(valid, newsvendor_cost(np.where(valid, Y, 0), np.nan_to_num(np.asarray(P[j][:, ev])),
                                               u[:, None, None], o[:, None, None]), 0)
        tot[c], tE[c] = cost.sum(), cost[us][:, E].sum()
    return pd.DataFrame({"whole period": {c: v / tot["Official"] for c, v in tot.items()},
                         "Elliott": {c: v / tE["Official"] for c, v in tE.items()}})


def extreme_days(r):
    """(N, days) masks of cold and hot extreme area-days."""
    W = pd.read_parquet(os.path.join(ROOT, "data", "interim", "weather_t2m.parquet"))
    days = calendar(r)
    daily = W.resample("D").mean()
    daily.index = daily.index.tz_localize(None) if daily.index.tz is not None else daily.index
    T = daily.reindex(days)[list(r["code"])].values.T                  # (N, days)
    lo = np.nanquantile(T, 0.025, axis=1, keepdims=True)
    hi = np.nanquantile(T, 0.975, axis=1, keepdims=True)
    return T < lo, T > hi


def two(x, y):
    return r"\begin{tabular}[b]{@{}c@{}}" + x + r"\\" + y + r"\end{tabular}"


def main():
    res = {s: load(s) for s in SPECS}
    res = {s: r for s, r in res.items() if r is not None}
    f = lambda x: "--" if x is None or not np.isfinite(x) else f"{x:.3f}"

    # ---- named events, all candidate sets ----
    out = {}
    for s, r in res.items():
        for ev, win in EVENTS.items():
            out[(ev, s or "none")] = event_ratios(r, win)
            out[(ev + " (no shed)", s or "none")] = event_ratios(r, win, exclude_shed=True)
    out = pd.DataFrame(out)
    print(out.round(3).to_string())
    out.to_csv(os.path.join(ROOT, "results", "events_global.csv"))

    alone = {}
    for s in ["_weather-gfs", "_weather-plus", "_weather-exact"]:
        if s in res:
            sa = standalone(s, EVENTS["Elliott"])
            if sa is not None:
                print("candidates on their own", s)
                print(sa.round(3).to_string())
                sa.to_csv(os.path.join(ROOT, "results", f"candidates_standalone{s}.csv"))
                alone[s] = sa["Elliott"]

    cols = [s for s in SPECS if s in res]
    head = {"": two("No", "weather"), "_weather-gfs": two("GFS", "temperature"),
            "_weather-plus": two("GFS/GEFS", "multi-variable"), "_weather-exact": two("Realised", "temperature")}
    lines = [r"\begin{tabular}{@{}l" + "c" * (len(cols) + 1) + "@{}}", r"\toprule",
             " & " + " & ".join(head[c] for c in cols) + " & " + two("Main set,", "no shed days") + r" \\",
             r"\midrule", r"\multicolumn{" + str(len(cols) + 2) + r"}{@{}l}{\emph{Combination rules}} \\"]
    for m, lab in LABEL.items():
        if m not in out.index:
            continue
        vals = [f(out.loc[m, ("Elliott", c or "none")]) for c in cols]
        ns = out.loc[m, ("Elliott (no shed)", MAIN)] if ("Elliott (no shed)", MAIN) in out.columns else np.nan
        lines.append(lab + " & " + " & ".join(vals) + " & " + f(ns) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(len(cols) + 2) + r"}{@{}l}{\emph{Weather candidates on their own}} \\"]
    for m, lab in ALONE.items():
        lines.append(lab + " & " + " & ".join(f(alone[c].get(m, np.nan)) if c in alone else "--" for c in cols)
                     + r" & -- \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(ROOT, "paper", "tables", "global_elliott.tex"), "w").write("\n".join(lines))

    # ---- extreme-temperature days, main candidate set ----
    r = res.get(MAIN)
    if r is None:
        return
    cold, hot = extreme_days(r)
    C = r["daily_cost"]                                  # (K, N, days)
    meth = list(r["methods"])
    ok = np.isfinite(C).all(axis=0)
    us = r["continent"] == "North America"
    groups = {"World, cold": cold, "World, hot": hot, "U.S., cold": cold & us[:, None], "U.S., hot": hot & us[:, None]}
    k0, ks, kq = meth.index("Official"), meth.index("GDMA-soft"), meth.index("Select")
    rows, cis = {}, {}
    for gname, mask in groups.items():
        mm = mask & ok
        daily = np.where(mm[None], np.nan_to_num(C), 0.0).sum(axis=1)       # (K, days)
        rows[gname] = daily.sum(axis=1) / daily[k0].sum()
        keep = daily[k0] > 0
        cis[gname] = (ratio_ci(daily[ks][keep], daily[k0][keep]), ratio_ci(daily[ks][keep], daily[kq][keep]),
                      int(mm.sum()))
    tab = pd.DataFrame(rows, index=meth)
    print(tab.round(3).to_string())
    print({g: c for g, c in cis.items()})
    tab.to_csv(os.path.join(ROOT, "results", "extremes_global.csv"))
    gn = list(groups)
    lines = [r"\begin{tabular}{@{}l" + "c" * len(gn) + "@{}}", r"\toprule",
             " & " + " & ".join(two(*g.split(", ")) for g in gn) + r" \\", r"\midrule"]
    lab = dict(LABEL, Official="Reference forecaster$^a$", NeuralGate="Neural gate (deep MoE)",
               EW="Equal weights", **{"FTO-PerUnit": "Forecast-then-commit, per area"})
    for m in ["Official", "EW", "Select", "Pooled", "PerUnit", "Shrink", "KMeans2S", "QR", "FTO-PerUnit",
              "FTO-Grouped", "NeuralGate", "GDMA", "GDMA-soft", "GDMA-fair"]:
        if m in tab.index:
            lines.append(lab[m] + " & " + " & ".join(f"{tab.loc[m, g]:.3f}" for g in gn) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Soft GDMA / reference, 90\% CI & " + " & ".join(
        f"[{cis[g][0][0]:.2f}, {cis[g][0][1]:.2f}]" for g in gn) + r" \\")
    lines.append(r"Soft GDMA / best single, 90\% CI & " + " & ".join(
        f"[{cis[g][1][0]:.2f}, {cis[g][1][1]:.2f}]" for g in gn) + r" \\")
    lines.append(r"Area-days & " + " & ".join(f"{cis[g][2]:,}" for g in gn) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(ROOT, "paper", "tables", "global_extremes.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
