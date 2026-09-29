"""Tables and figures for the global (five-continent) application.

Inputs: results/backtest_global_w{W}_taucalibrated.npz, neuralgate_*.npz and the
storage_*.csv files written by storage_siting.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
CONTS = ["North America", "Europe", "Oceania", "Asia", "Africa"]
LABEL = {"Official": "Single best forecaster$^a$", "EW": "Equal weights", "Select": "Best single (per area)",
         "Pooled": "Pooled DF weights", "PerUnit": "Per-area DF weights", "Shrink": "Shrinkage to pooled",
         "KMeans2S": "Two-step $k$-means", "FTO-PerUnit": "Forecast-then-commit, per area",
         "FTO-Grouped": "Forecast-then-commit, grouped", "NeuralGate": "Neural gate (deep MoE)",
         "GDMA": "GDMA", "GDMA-soft": "\\textbf{Soft GDMA}"}
ORDER = ["Official", "EW", "Select", "Pooled", "PerUnit", "Shrink", "KMeans2S", "FTO-PerUnit",
         "FTO-Grouped", "NeuralGate", "GDMA", "GDMA-soft"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def dm(a, b, lag=7):
    x = a - b
    n = len(x)
    xc = x - x.mean()
    v = xc @ xc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (xc[k:] @ xc[:-k]) / n
    t = x.mean() / np.sqrt(v / n)
    return t, 2 * stats.norm.sf(abs(t))


def load(W):
    tag = f"global_w{W}_taucalibrated"
    r = dict(np.load(os.path.join(ROOT, "results", f"backtest_{tag}.npz"), allow_pickle=True))
    ng = os.path.join(ROOT, "results", f"neuralgate_{tag}.npz")
    if os.path.exists(ng):
        g = np.load(ng, allow_pickle=True)
        assert np.array_equal(g["eval_days"], r["eval_days"])
        r["daily_cost"] = np.concatenate([r["daily_cost"], g["daily_cost"]])
        r["short_hours"] = np.concatenate([r["short_hours"], g["short_hours"]])
        r["methods"] = np.concatenate([r["methods"], g["methods"]])
        r["ng_params"] = int(g["n_params"])
    return r


def star(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""


def main():
    res = {W: load(W) for W in (28, 14) if os.path.exists(os.path.join(ROOT, "results", f"backtest_global_w{W}_taucalibrated.npz"))}
    r = res[28]
    meth = list(r["methods"])
    C = r["daily_cost"]                       # (K, N, days)
    cont = r["continent"]
    ref = meth.index("GDMA-soft")
    k0 = meth.index("Official")
    rows = [m for m in ORDER if m in meth]

    # ---------------- Table: cost by continent (W = 28) ---------------------
    cols = ["World"] + CONTS
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
             "Method & " + " & ".join(cols) + r" \\", r"\midrule"]
    tab = {}
    for c in cols:
        mask = np.ones(len(cont), bool) if c == "World" else cont == c
        sysd = C[:, mask].sum(axis=1)
        tot = sysd.sum(axis=1)
        tab[c] = (tot / tot[k0], sysd)
    for m in rows:
        k = meth.index(m)
        cells = []
        for c in cols:
            rel, sysd = tab[c]
            v = f"{rel[k]:.3f}"
            if np.isclose(rel[k], rel[[meth.index(x) for x in rows]].min()):
                v = r"\textbf{" + v + "}"
            if k != ref and sysd[ref].sum() > 0:
                v += star(dm(sysd[k][sysd[ref] > 0], sysd[ref][sysd[ref] > 0])[1])
            cells.append(v)
        lines.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    n_areas = [len(cont)] + [int((cont == c).sum()) for c in CONTS]
    lines += [r"\midrule", "Grid areas & " + " & ".join(map(str, n_areas)) + r" \\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "global_main.tex"), "w").write("\n".join(lines))

    # ---------------- Table: windows W = 14 vs 28 (world) ------------------
    if 14 in res:
        lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Method & $W=14$ & $W=28$ \\", r"\midrule"]
        rel = {}
        for W, rr in res.items():
            s = rr["daily_cost"].sum(axis=(1, 2))
            rel[W] = dict(zip(rr["methods"], s / s[list(rr["methods"]).index("Official")]))
        for m in rows:
            lines.append(LABEL[m] + " & " + " & ".join(f"{rel[W].get(m, np.nan):.3f}" for W in (14, 28)) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "global_windows.tex"), "w").write("\n".join(lines))

    # ---------------- Table: common critical ratio (world, W = 28) ------------
    tp = os.path.join(ROOT, "results", "backtest_global_w28_tau0.9.npz")
    if os.path.exists(tp):
        rt = np.load(tp, allow_pickle=True)
        mt = list(rt["methods"])
        ct = rt["daily_cost"]
        lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
                 "Method & " + " & ".join(cols) + r" \\", r"\midrule"]
        rel_t = {}
        for c in cols:
            mask = np.ones(len(cont), bool) if c == "World" else cont == c
            tot = ct[:, mask].sum(axis=(1, 2))
            rel_t[c] = tot / tot[mt.index("Official")]
        for m in [x for x in rows if x in mt]:
            k = mt.index(m)
            cells = []
            for c in cols:
                v = rel_t[c][k]
                best = min(rel_t[c][mt.index(x)] for x in rows if x in mt)
                cells.append((r"\textbf{%.3f}" % v) if np.isclose(v, best) else f"{v:.3f}")
            lines.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "global_tau.tex"), "w").write("\n".join(lines))

    # ---------------- Table: entry (transfer) ---------------------------
    act = r["active"]                         # (origins, N)
    origins, ev = r["origins"], r["eval_days"]
    entry = np.array([np.argmax(a) if a.any() else -1 for a in act.T])
    late = np.flatnonzero(entry > 0)          # areas that enter after the first origin
    horizon = 8
    mask_days = np.zeros(C.shape[1:], bool)
    for i in late:
        d0 = origins[entry[i]]
        sel = (ev >= d0) & (ev < d0 + 7 * horizon)
        mask_days[i, sel] = True
    ent = (C * mask_days[None]).sum(axis=(1, 2))
    rest = (C * (~mask_days)[None]).sum(axis=(1, 2))
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Method & First 8 weeks after entry$^b$ & All other weeks \\", r"\midrule"]
    for m in rows:
        k = meth.index(m)
        lines.append(f"{LABEL[m]} & {ent[k] / ent[k0]:.3f} & {rest[k] / rest[k0]:.3f}" + r" \\")
    lines += [r"\midrule", f"Grid areas entering & {len(late)} & " + r"\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "global_entry.tex"), "w").write("\n".join(lines))
    entry_codes = pd.Series(r["code"][late]).str[:2].value_counts().to_dict()

    # ---------------- segments by continent -----------------------------------
    labels, G = r["labels"], r["G"]
    multi = G > 1
    co = np.zeros((len(cont), len(cont)))
    cnt = np.zeros_like(co)
    for L, a in zip(labels[multi], act[multi]):
        both = a[:, None] & a[None, :]
        co += (L[:, None] == L[None, :]) & both
        cnt += both
    with np.errstate(all="ignore"):
        co = co / cnt
    within = {c: np.nanmean(co[np.ix_(cont == c, cont == c)][~np.eye((cont == c).sum(), dtype=bool)]) for c in CONTS if (cont == c).sum() > 1}
    across = {}
    for i, a in enumerate(CONTS):
        for b in CONTS[i + 1:]:
            across[f"{a}-{b}"] = np.nanmean(co[np.ix_(cont == a, cont == b)])

    # ---------------- figure: storage frontier ---------------------------------
    pol = {"Official": ("Single best forecaster", "#7f7f7f", "-."), "EW": ("Equal weights", "#bf9000", ":"),
           "Pooled": ("Pooled DF", "#c55a11", "--"), "PerUnit": ("Per-area DF", "#548235", ":"),
           "GDMA-soft": ("Soft GDMA", "#1f4e79", "-")}
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for m, (lab, c, ls) in pol.items():
        p = os.path.join(ROOT, "results", f"storage_frontier_{m}.npy")
        if os.path.exists(p):
            B, cost = np.load(p)
            ax.plot(B, cost / 1e3, color=c, ls=ls, lw=2 if m == "GDMA-soft" else 1.2, label=lab)
    ax.set_xlabel("Battery power installed across the five continents (GW, optimally sited)")
    ax.set_ylabel("Annual commitment cost\n(billion USD)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "storage_frontier.pdf"))
    plt.close(fig)

    # ---------------- storage tables ------------------------------------------
    bp = os.path.join(ROOT, "results", "storage_budget.csv")
    if os.path.exists(bp):
        bud = pd.read_csv(bp)
        eq = pd.read_csv(os.path.join(ROOT, "results", "storage_equivalent.csv")).set_index("policy")
        SL = {"Official": "Single best forecaster", "EW": "Equal weights", "Pooled": "Pooled DF",
              "PerUnit": "Per-area DF", "FTO-PerUnit": "Forecast-then-commit", "GDMA": "GDMA",
              "GDMA-soft": r"\textbf{Soft GDMA}"}
        lines = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
                 r" & Cost without & GW to match & \multicolumn{5}{c}{Siting of a 10 GW budget (GW)} \\",
                 r"\cmidrule(lr){4-8}",
                 r"Commitment rule & storage (\$bn/yr) & soft GDMA$^c$ & N.\ Am. & Europe & Oceania & Asia & Africa \\",
                 r"\midrule"]
        for m in SL:
            b = bud[(bud.budget_gw == 10) & (bud.policy == m)].iloc[0]
            g = eq.loc[m, "gw_to_match_soft"]
            gs = "$>60$" if not np.isfinite(g) else ("--" if m == "GDMA-soft" else f"{g:.0f}")
            lines.append(f"{SL[m]} & {eq.loc[m, 'cost_no_storage_musd'] / 1e3:.2f} & {gs} & "
                         + " & ".join(f"{b[f'gw_{c}']:.1f}" for c in CONTS) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "storage.tex"), "w").write("\n".join(lines))
        lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
                 r"Siting based on & 1 GW & 5 GW & 10 GW & 20 GW \\", r"\midrule"]
        for m in ["Official", "PerUnit", "FTO-PerUnit", "GDMA-soft"]:
            v = [bud[(bud.budget_gw == B) & (bud.policy == m)].value_under_soft_musd.iloc[0] for B in (1, 5, 10, 20)]
            lines.append(SL[m] + " & " + " & ".join(f"{x:.0f}" for x in v) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "siting_value.tex"), "w").write("\n".join(lines))

    # ---------------- console summary ----------------------------------------
    pd.set_option("display.width", 200)
    print(pd.DataFrame({c: dict(zip(meth, tab[c][0])) for c in cols}).loc[rows].round(4))
    print("entry areas", len(late), entry_codes, dict(zip(meth, (ent / ent[k0]).round(3))))
    print("G distribution", np.bincount(G), "mean kappa", r["kappa"].mean().round(2))
    print("within-continent co-membership", {k: round(v, 2) for k, v in within.items()})
    print("across", {k: round(v, 2) for k, v in across.items()})
    if "ng_params" in r:
        print("neural gate parameters", r["ng_params"])
    short = r["short_hours"].sum(axis=(1, 2)) / r["valid_hours"].sum()
    print("shortfall rate", dict(zip(meth, short.round(3))), "target", (1 - r["tau"]).mean().round(3))


if __name__ == "__main__":
    main()
