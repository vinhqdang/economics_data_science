"""Cost of every combination rule during Winter Storm Elliott (22-27 December
2022) on the U.S. balancing authorities of the global panel, relative to the
official day-ahead forecasts, for each candidate set; and the cost of each
candidate's own commitment policy, over the whole evaluation period (relative
to the single best forecaster) and during Elliott (relative to the official
forecasts)."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVENT = ("2022-12-22", "2022-12-27")


def event_ratios(suffix):
    p = os.path.join(ROOT, "results", f"backtest_global_w28_taucalibrated{suffix}.npz")
    if not os.path.exists(p):
        return None
    r = np.load(p, allow_pickle=True)
    cal = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)["days"]
    ev = pd.to_datetime(cal[r["eval_days"]])
    days = (ev >= EVENT[0]) & (ev <= EVENT[1])
    us = r["continent"] == "North America"
    c = r["daily_cost"][:, us][:, :, days]
    ok = np.isfinite(c).all(axis=0)
    tot = np.where(ok, c, 0.0).sum(axis=(1, 2))
    m = list(r["methods"])
    return pd.Series(tot / tot[m.index("Official")], index=m)


def standalone(suffix, n_cand):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from gdma.core import newsvendor_cost
    r = np.load(os.path.join(ROOT, "results", f"backtest_global_w28_taucalibrated{suffix}.npz"), allow_pickle=True)
    pn = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    P = np.load(os.path.join(ROOT, "data", "processed", f"global_policies_{n_cand}_calibrated{suffix}.npy"), mmap_mode="r")
    ev = r["eval_days"]
    Y = pn["Y"][:, ev, :].astype(float)
    u, o = r["u"], r["o"]
    valid = np.isfinite(Y) & np.isfinite(np.asarray(P[:, :, ev, :])).all(axis=0)
    days = pd.to_datetime(pn["days"][ev])
    E = (days >= EVENT[0]) & (days <= EVENT[1])
    us = r["continent"] == "North America"
    tot, tE = {}, {}
    for j, c in enumerate(r["candidates"]):
        cost = np.where(valid, newsvendor_cost(np.where(valid, Y, 0), np.nan_to_num(np.asarray(P[j][:, ev])),
                                               u[:, None, None], o[:, None, None]), 0)
        tot[str(c)], tE[str(c)] = cost.sum(), cost[us][:, E].sum()
    return pd.DataFrame({"whole period": {c: v / tot["Official"] for c, v in tot.items()},
                         "Elliott": {c: v / tE["Official"] for c, v in tE.items()}})


def main():
    specs = sys.argv[1:] or ["", "_weather-gfs", "_weather-plus", "_weather-exact"]
    res = {s or "none": event_ratios(s) for s in specs}
    out = pd.DataFrame({k: v for k, v in res.items() if v is not None})
    print(out.round(3).to_string())
    out.to_csv(os.path.join(ROOT, "results", "elliott_global.csv"))
    alone = {}
    for sfx, n in [("_weather-gfs", 9), ("_weather-plus", 11), ("_weather-exact", 9)]:
        if os.path.exists(os.path.join(ROOT, "data", "processed", f"global_policies_{n}_calibrated{sfx}.npy")):
            sa = standalone(sfx, n)
            print("candidates on their own", sfx)
            print(sa.round(3).to_string())
            sa.to_csv(os.path.join(ROOT, "results", f"candidates_standalone{sfx}.csv"))
            alone[sfx] = sa["Elliott"]
    latex_table(out, alone)


LABEL = {"GDMA-soft": r"\textbf{Soft GDMA}", "GDMA-fair": "Fair soft GDMA", "GDMA": "GDMA",
         "KMeans2S": "Two-step $k$-means", "Shrink": "Shrinkage to pooled", "PerUnit": "Per-area DF weights",
         "Pooled": "Pooled DF weights", "FTO-Grouped": "Forecast-then-commit, grouped"}
ALONE = {"Weather": "Temperature model", "WeatherPlus": "Multi-variable model",
         "WeatherEns": "Multi-variable, ensemble margin"}


def latex_table(out, alone):
    cols = [c for c in ["none", "_weather-gfs", "_weather-plus", "_weather-exact"] if c in out]
    two = lambda x, y: r"\begin{tabular}[b]{@{}c@{}}" + x + r"\\" + y + r"\end{tabular}"
    head = {"none": two("No", "weather"), "_weather-gfs": two("GFS", "temperature"),
            "_weather-plus": two("GFS/GEFS", "multi-variable"), "_weather-exact": two("Realised", "temperature")}
    f = lambda x: "--" if not np.isfinite(x) else f"{x:.3f}"
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
             " & " + " & ".join(head[c] for c in cols) + r" \\", r"\midrule",
             r"\multicolumn{" + str(len(cols) + 1) + r"}{@{}l}{\emph{Combination rules}} \\"]
    for m, lab in LABEL.items():
        lines.append(lab + " & " + " & ".join(f(out.loc[m, c]) for c in cols) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(len(cols) + 1) + r"}{@{}l}{\emph{Weather candidates on their own}} \\"]
    for m, lab in ALONE.items():
        lines.append(lab + " & " + " & ".join(f(alone[c].get(m, np.nan)) if c in alone else "--" for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(ROOT, "paper", "tables", "global_elliott.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
