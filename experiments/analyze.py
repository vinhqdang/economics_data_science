"""Tables and figures for the U.S. grid application.

Reads results/backtest_<tag>.npz and writes LaTeX tables to paper/tables and
figures to paper/figures.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
LABEL = {"Official": "Official forecast", "EW": "Equal weights", "Select": "Best single (per BA)",
         "Pooled": "Pooled DF weights", "PerBA": "Per-BA DF weights", "Shrink": "Shrinkage to pooled",
         "KMeans2S": "Two-step $k$-means", "FTO-PerBA": "Forecast-then-commit, per BA",
         "FTO-Grouped": "Forecast-then-commit, grouped", "GDMA": "\\textbf{GDMA}", "GDMA-min": "GDMA, minimum hold-out rule",
         "GDMA-soft": "\\textbf{Soft GDMA}"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load(tag):
    d = np.load(os.path.join(ROOT, "results", f"backtest_{tag}.npz"), allow_pickle=True)
    r = {k: d[k] for k in d.files}
    vp = os.path.join(ROOT, "results", f"variants_{tag}.npz")
    if os.path.exists(vp):
        v = np.load(vp, allow_pickle=True)
        assert np.array_equal(v["eval_days"], r["eval_days"])
        r["daily_cost"] = np.concatenate([r["daily_cost"], v["daily_cost"]])
        r["short_hours"] = np.concatenate([r["short_hours"], v["short_hours"]])
        r["methods"] = np.concatenate([r["methods"], v["methods"]])
        r["G_min"] = v["G"]
        r["G_soft"], r["kappa_soft"] = v["G_soft"], v["kappa_soft"]
    # BA-days on which any method lacks a commitment (all candidates missing)
    # are dropped for every method so that all methods are scored on the same cells
    bad = np.isnan(r["daily_cost"]).any(axis=0)
    r["daily_cost"] = np.where(bad[None], 0.0, r["daily_cost"])
    r["n_dropped"] = int(bad.sum())
    return r


def dm_test(a, b, lag=7):
    """Diebold-Mariano test on daily loss differential a - b with Newey-West HAC."""
    x = a - b
    n = len(x)
    xc = x - x.mean()
    v = xc @ xc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (xc[k:] @ xc[:-k]) / n
    t = x.mean() / np.sqrt(v / n)
    return t, 2 * stats.norm.sf(abs(t))


def summary(r):
    meth = list(r["methods"])
    C = r["daily_cost"]                          # (K, N, days)
    sysd = C.sum(axis=1)                         # (K, days) system cost per day
    tot = sysd.sum(axis=1)
    k_ew, k_off, k_g = meth.index("EW"), meth.index("Official"), meth.index("GDMA")
    # scale-free: each BA's cost divided by its mean hourly demand, averaged
    norm = (C.sum(axis=2) / r["scale"][None]).mean(axis=1)
    per_ba = C.sum(axis=2)
    rows = []
    for k, m in enumerate(meth):
        t, p = dm_test(sysd[k], sysd[k_g]) if k != k_g else (np.nan, np.nan)
        rows.append(dict(method=m, rel_ew=tot[k] / tot[k_ew], rel_off=tot[k] / tot[k_off],
                         norm_rel_ew=norm[k] / norm[k_ew],
                         beat_off=(per_ba[k] < per_ba[k_off]).mean(),
                         dm_t=t, dm_p=p,
                         short_rate=r["short_hours"][k].sum() / r["valid_hours"].sum()))
    return pd.DataFrame(rows).set_index("method"), sysd


def main():
    tags = sorted(os.path.basename(p)[9:-4] for p in glob.glob(os.path.join(ROOT, "results", "backtest_*.npz")))
    print("found", tags)
    windows = [t for t in tags if t.endswith("taucalibrated")]
    windows = sorted(windows, key=lambda t: int(t.split("_")[0][1:]))
    res = {t: load(t) for t in tags}

    # ---------------- Table: main results by window -----------------------
    summ = {t: summary(res[t])[0] for t in windows}
    meth = []
    for t in windows:
        meth += [m for m in res[t]["methods"] if m not in meth]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(windows) + "c@{}}", r"\toprule",
             " & \\multicolumn{%d}{c}{System cost relative to the official-forecast policy} & \\\\" % len(windows),
             r"\cmidrule(lr){2-%d}" % (len(windows) + 1),
             "Method & " + " & ".join(f"$W={t.split('_')[0][1:]}$" for t in windows) + r" & Share of BAs \\",
             " & " + " & ".join("days" for _ in windows) + r" & improved$^a$ \\", r"\midrule"]
    main_w = "w28_taucalibrated" if "w28_taucalibrated" in windows else windows[-1]
    for m in meth:
        cells = []
        for t in windows:
            if m not in summ[t].index:
                cells.append("--")
                continue
            s = summ[t].loc[m]
            best = summ[t]["rel_off"].min()
            v = f"{s.rel_off:.4f}"
            if np.isclose(s.rel_off, best):
                v = r"\textbf{" + v + "}"
            star = ""
            if m != "GDMA" and np.isfinite(s.dm_p):
                star = "$^{***}$" if s.dm_p < 0.01 else "$^{**}$" if s.dm_p < 0.05 else "$^{*}$" if s.dm_p < 0.1 else ""
            cells.append(v + star)
        share = f"{summ[main_w].loc[m].beat_off:.2f}" if m in summ[main_w].index else "--"
        lines.append(LABEL[m] + " & " + " & ".join(cells) + f" & {share}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "main.tex"), "w").write("\n".join(lines))

    # ---------------- Table: scale-free and shortfall rates (main window) ------
    s = summ[main_w]
    tau = res[main_w]["tau"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Method & Scale-free cost$^b$ & Shortfall & DM statistic \\",
             r" & (rel.\ to EW) & frequency & vs GDMA$^c$ \\", r"\midrule"]
    for m in s.index:
        dm = "--" if m == "GDMA" else f"{s.loc[m].dm_t:.2f}"
        lines.append(f"{LABEL[m]} & {s.loc[m].norm_rel_ew:.4f} & {s.loc[m].short_rate:.3f} & {dm}" + r" \\")
    lines += [r"\midrule", f"Target (mean $\\tau_i$ across BAs) & & {1 - tau.mean():.3f} & " + r"\\",
              r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "secondary.tex"), "w").write("\n".join(lines))

    # ---------------- Table: sensitivity to the critical ratio ------------------
    sens = [t for t in tags if t.startswith(main_w.split("_")[0] + "_tau") and not t.endswith("calibrated")]
    if sens:
        sens = sorted(sens, key=lambda t: float(t.split("tau")[1]))
        ss = {t: summary(res[t])[0] for t in sens + [main_w]}
        cols = sens + [main_w]
        lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
                 "Method & " + " & ".join(("$\\tau=" + t.split("tau")[1] + "$") if "calibrated" not in t else "Calibrated" for t in cols) + r" \\",
                 r"\midrule"]
        for m in meth:
            cells = []
            for t in cols:
                v = ss[t].loc[m].rel_off
                b = ss[t]["rel_off"].min()
                cells.append((r"\textbf{%.4f}" % v) if np.isclose(v, b) else f"{v:.4f}")
            lines.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "tau.tex"), "w").write("\n".join(lines))

    # ---------------- Figure: cumulative savings over time ---------------------
    r = res[main_w]
    _, sysd = summary(r)
    meth = list(r["methods"])
    days = pd.to_datetime(np.load(os.path.join(ROOT, "data", "processed", "panel.npz"), allow_pickle=True)["days"])
    ev = days[r["eval_days"]]
    k_off = meth.index("Official")
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    styles = {"GDMA": ("#1f4e79", "-", 2.0), "Pooled": ("#c55a11", "--", 1.2),
              "PerBA": ("#548235", ":", 1.4), "FTO-PerBA": ("#7f7f7f", "-.", 1.2),
              "EW": ("#bf9000", (0, (1, 1)), 1.2)}
    for m, (c, ls, lw) in styles.items():
        k = meth.index(m)
        cum = np.cumsum(sysd[k_off] - sysd[k]) / np.cumsum(sysd[k_off])
        ax.plot(ev, 100 * cum, color=c, ls=ls, lw=lw, label=LABEL[m].replace("\\textbf{", "").replace("}", ""))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Cumulative cost saving vs\nofficial-forecast policy (%)")
    ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cumulative.pdf"))
    plt.close(fig)

    # ---------------- Figure: number of segments and co-membership -------------
    G = r["G"]
    lab = r["labels"]
    ba = r["ba"]
    N = len(ba)
    co = np.zeros((N, N))
    for L in lab:
        co += (L[:, None] == L[None, :])
    co /= len(lab)
    order = leaves_list(linkage(1 - co[np.triu_indices(N, 1)], "average"))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    og = pd.to_datetime(days[r["origins"]])
    axs[0].step(og, G, where="post", color="#1f4e79", lw=1)
    axs[0].set_ylabel("Selected number of segments $G$")
    axs[0].set_yticks(range(1, int(G.max()) + 1))
    im = axs[1].imshow(co[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
    axs[1].set_xticks(range(N))
    axs[1].set_yticks(range(N))
    axs[1].set_xticklabels(ba[order], rotation=90, fontsize=4.5)
    axs[1].set_yticklabels(ba[order], fontsize=4.5)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02, label="Share of windows in same segment")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "segments.pdf"))
    plt.close(fig)

    # ---------------- average weights by segment-type --------------------------
    W = r["W"]  # (origins, N, M)
    mnames = list(np.load(os.path.join(ROOT, "data", "processed", "panel.npz"), allow_pickle=True)["methods"])
    avgW = pd.DataFrame(W.mean(axis=0), index=ba, columns=mnames)
    avgW.to_csv(os.path.join(ROOT, "results", f"avg_weights_{main_w}.csv"))

    # ---------------- extreme events ---------------------------------------
    events = {"Winter Storm Uri (10--20 Feb 2021)": ("2021-02-10", "2021-02-20"),
              "Winter Storm Elliott (22--27 Dec 2022)": ("2022-12-22", "2022-12-27"),
              "Heat wave (15 Jul--15 Aug 2023)": ("2023-07-15", "2023-08-15")}
    lines = [r"\begin{tabular}{@{}l" + "c" * len(events) + "@{}}", r"\toprule",
             "Method & " + " & ".join(events) + r" \\", r"\midrule"]
    evs = {}
    for name, (a, b) in events.items():
        mask = (ev >= a) & (ev <= b)
        evs[name] = sysd[:, mask].sum(axis=1) / sysd[k_off, mask].sum()
    for k, m in enumerate(meth):
        cells = []
        for name in events:
            v = evs[name][k]
            cells.append((r"\textbf{%.3f}" % v) if np.isclose(v, evs[name].min()) else f"{v:.3f}")
        lines.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "events.tex"), "w").write("\n".join(lines).replace("Winter Storm ", "").replace(" (", r"\newline (") if False else "\n".join(lines))

    # ---------------- console summary ---------------------------------------
    for t in windows:
        print("\n==", t)
        print(summ[t].round(4).to_string())
    for t in sens:
        print("\n==", t)
        print(summary(res[t])[0][["rel_ew", "rel_off", "dm_p"]].round(4).to_string())
    print("\nG distribution", np.bincount(G))
    for name in events:
        print(name, dict(zip(meth, evs[name].round(3))))
    print("\nsegment kappa / G_km / G_fto means", r["kappa"].mean(), r["G_km"].mean(), r["G_fto"].mean())
    tot_mwh = np.nansum(np.load(os.path.join(ROOT, "data", "processed", "panel.npz"))["Y"][:, r["eval_days"]])
    print("total demand in evaluation MWh", tot_mwh)
    tot = sysd.sum(axis=1)
    print("cost per MWh demand", dict(zip(meth, (tot / tot_mwh).round(5))))


if __name__ == "__main__":
    main()
