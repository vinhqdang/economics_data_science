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
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mcs import model_confidence_set  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
CONTS = ["North America", "Europe", "Oceania", "Asia", "Africa"]
LABEL = {"Official": "Reference forecaster$^a$", "EW": "Equal weights", "Select": "Best single (per area)", "QR": "Quantile-regression comb.",
         "Pooled": "Pooled DF weights", "PerUnit": "Per-area DF weights", "Shrink": "Shrinkage to pooled",
         "KMeans2S": "Two-step $k$-means", "FTO-PerUnit": "Forecast-then-commit, per area",
         "FTO-Grouped": "Forecast-then-commit, grouped", "NeuralGate": "Neural gate (deep MoE)",
         "GDMA": "GDMA", "GDMA-soft": "\\textbf{Soft GDMA}", "GDMA-fair": "Fair soft GDMA"}
ORDER = ["Official", "EW", "Select", "Pooled", "PerUnit", "Shrink", "KMeans2S", "QR", "FTO-PerUnit",
         "FTO-Grouped", "NeuralGate", "GDMA", "GDMA-soft", "GDMA-fair"]


def fairness_table(r, path):
    """Distribution of benefits across grid areas: per-area cost relative to the
    reference forecaster, and share of areas worse off than under the reference
    and than under their own best single candidate (chosen in-window)."""
    meth = list(r["methods"])
    C = r["daily_cost"].sum(axis=2)                  # (K, N)
    k0 = meth.index("Official")
    kb = meth.index("Select")
    ok = C[k0] > 0
    ratio = C[:, ok] / C[k0, ok][None]
    ratio_b = C[:, ok] / np.maximum(C[kb, ok][None], 1e-12)
    cont = r["continent"][ok]
    years = (r["valid_hours"][ok].sum(axis=1) / 8760.0)
    poor = years < 2.0
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Per-area cost ratio} & \multicolumn{2}{c}{Areas harmed vs} & Worst & Data-rich & Data-poor \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Method & P10 & Median & P90 & ref. & best single & continent$^d$ & areas & areas$^e$ \\", r"\midrule"]
    out = {}
    for m in [x for x in ORDER if x in meth]:
        k = meth.index(m)
        q = np.percentile(ratio[k], [10, 50, 90])
        harmed = (ratio[k] > 1.0).mean()
        harmed_b = (ratio_b[k] > 1.0).mean()
        worst = max(np.median(ratio[k][cont == c]) for c in CONTS if (cont == c).any())
        rich = C[k, ok][~poor].sum() / C[k0, ok][~poor].sum()
        pr = C[k, ok][poor].sum() / C[k0, ok][poor].sum()
        out[m] = dict(p10=q[0], med=q[1], p90=q[2], harmed=harmed, harmed_best=harmed_b, worst=worst,
                      rich=rich, poor=pr)
        hb = "--" if m == "Select" else f"{harmed_b:.2f}"
        lines.append(f"{LABEL[m]} & {q[0]:.3f} & {q[1]:.3f} & {q[2]:.3f} & {harmed:.2f} & {hb} & {worst:.3f} & {rich:.3f} & {pr:.3f}" + r" \\")
    lines += [r"\midrule", f"Areas & & {int(ok.sum())} & & & & & {int((~poor).sum())} & {int(poor.sum())}" + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines))
    return out
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


MAIN = os.environ.get("MAIN_SUFFIX", "_weather-plus")


def load(W, suffix=None):
    suffix = MAIN if suffix is None else suffix
    tag = f"global_w{W}_taucalibrated{suffix}"
    r = dict(np.load(os.path.join(ROOT, "results", f"backtest_{tag}.npz"), allow_pickle=True))
    ng = os.path.join(ROOT, "results", f"neuralgate_{tag}.npz")
    if os.path.exists(ng):
        g = np.load(ng, allow_pickle=True)
        if not np.array_equal(g["eval_days"], r["eval_days"]):
            print("WARNING: neural-gate file for", tag, "does not match the backtest (stale); skipped")
            return r
        r["daily_cost"] = np.concatenate([r["daily_cost"], g["daily_cost"]])
        r["short_hours"] = np.concatenate([r["short_hours"], g["short_hours"]])
        r["methods"] = np.concatenate([r["methods"], g["methods"]])
        r["ng_params"] = int(g["n_params"])
    return r


def star(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.1 else ""


def main():
    res = {W: load(W) for W in (28, 14) if os.path.exists(os.path.join(ROOT, "results", f"backtest_global_w{W}_taucalibrated{MAIN}.npz"))}
    if 14 not in res and os.path.exists(os.path.join(ROOT, "results", "backtest_global_w14_taucalibrated.npz")):
        res[14] = load(14, "")
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
    # 90% model confidence set on world daily cost (Hansen, Lunde and Nason, 2011)
    kk = [meth.index(m) for m in rows]
    world = tab["World"][1][kk]
    in_mcs, p_mcs = model_confidence_set(world[:, world.sum(axis=0) > 0], alpha=0.10, B=1000, block=7)
    mcs_members = {rows[j]: bool(in_mcs[j]) for j in range(len(rows))}
    print("90% MCS (world daily cost):", [m for m, v in mcs_members.items() if v],
          {rows[j]: round(float(p_mcs[j]), 3) for j in range(len(rows))})
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
            if c == "World" and mcs_members.get(m):
                v += r"$^\dagger$"
            cells.append(v)
        lines.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    n_areas = [len(cont)] + [int((cont == c).sum()) for c in CONTS]
    lines += [r"\midrule", "Grid areas & " + " & ".join(map(str, n_areas)) + r" \\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "global_main.tex"), "w").write("\n".join(lines))

    # ---------------- Table: windows W = 14 vs 28 (world) ------------------
    if 14 in res:
        lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Method & $W=14$ & $W=28$ \\", r"\midrule"]
        rel = {}
        both = all(os.path.exists(os.path.join(ROOT, "results", f"backtest_global_w{W}_taucalibrated{MAIN}.npz"))
                   for W in (14, 28))
        WIN_SUFFIX = MAIN if both else ""          # same candidate set for both windows
        for W in (14, 28):
            rr = load(W, WIN_SUFFIX)
            s = rr["daily_cost"].sum(axis=(1, 2))
            rel[W] = dict(zip(rr["methods"], s / s[list(rr["methods"]).index("Official")]))
        for m in rows:
            lines.append(LABEL[m] + " & " + " & ".join(f"{rel[W][m]:.3f}" if m in rel[W] else "--" for W in (14, 28)) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "global_windows.tex"), "w").write("\n".join(lines))

    # ---------------- Table: common critical ratio (world, W = 28) ------------
    tp = os.path.join(ROOT, "results", f"backtest_global_w28_tau0.9{MAIN}.npz")
    TAU_SUFFIX = MAIN
    if not os.path.exists(tp):
        tp, TAU_SUFFIX = os.path.join(ROOT, "results", "backtest_global_w28_tau0.9.npz"), ""
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

    # ---------------- Table: tail risk and a scarcity-level critical ratio ----
    def tail_stats(rr, names):
        mm = list(rr["methods"])
        wd = rr["daily_cost"].sum(axis=1)                       # (K, days) world daily cost
        k0_ = mm.index("Official")
        n = wd.shape[1] // 7 * 7
        wk = wd[:, :n].reshape(len(mm), -1, 7).sum(axis=2)
        cv = lambda x: np.sort(x)[-max(1, int(0.05 * len(x))):].mean()
        out_ = {}
        for m in names:
            if m in mm:
                k = mm.index(m)
                out_[m] = (wd[k].sum() / wd[k0_].sum(), cv(wd[k]) / cv(wd[k0_]), wk[k].max() / wk[k0_].max())
        return out_
    tail = tail_stats(r, rows)
    t99p = os.path.join(ROOT, "results", f"backtest_global_w28_tau0.99{MAIN}.npz")
    tail99 = tail_stats(np.load(t99p, allow_pickle=True), rows) if os.path.exists(t99p) else {}
    two = lambda x, y: r"\begin{tabular}[b]{@{}c@{}}" + x + r"\\" + y + r"\end{tabular}"
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{Calibrated $\tau_i$ (main)} & \multicolumn{3}{c}{$\tau_i=0.99$ for every area} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             "Method & Total & " + two("CVaR$_{95}$", "daily") + " & " + two("Worst", "week") + " & Total & "
             + two("CVaR$_{95}$", "daily") + " & " + two("Worst", "week") + r" \\", r"\midrule"]
    fm = lambda d, m: " & ".join(f"{x:.3f}" for x in d[m]) if m in d else "-- & -- & --"
    for m in rows:
        if m in tail:
            lines.append(LABEL[m] + " & " + fm(tail, m) + " & " + fm(tail99, m) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "global_tail.tex"), "w").write("\n".join(lines))
    print("tail risk", {m: np.round(v, 3).tolist() for m, v in tail.items()})
    print("tau 0.99", {m: np.round(v, 3).tolist() for m, v in tail99.items()})

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
        p = os.path.join(ROOT, "results", f"storage_frontier{MAIN}_{m}.npy")
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
    SS = MAIN.replace("_", "_", 1)
    bp = os.path.join(ROOT, "results", f"storage_budget{SS}.csv")
    if os.path.exists(bp):
        bud = pd.read_csv(bp)
        eq = pd.read_csv(os.path.join(ROOT, "results", f"storage_equivalent{SS}.csv")).set_index("policy")
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
    op = os.path.join(ROOT, "results", f"storage_oos{SS}.csv")
    if os.path.exists(op):
        oo = pd.read_csv(op)
        lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
                 r"Budget & Sited on soft GDMA & Sited on the reference & Sited on the second \\",
                 r"(GW) & (first half) & (first half) & half itself (oracle) \\", r"\midrule"]
        for _, x in oo.iterrows():
            lines.append(f"{int(x.budget_gw)} & {x['sited_on_GDMA-soft_musd']:.0f} & {x['sited_on_Official_musd']:.0f} & "
                         f"{x.oracle_musd:.0f}" + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "siting_oos.tex"), "w").write("\n".join(lines))
        print("out-of-sample siting", oo.round(1).to_dict("records"))

    fp = os.path.join(ROOT, "results", f"storage_fair{MAIN}.csv")
    if os.path.exists(fp):
        fr = pd.read_csv(fp)
        rel = [c for c in fr.columns if c.startswith("relben_")]
        fr["min_relben"] = fr[rel].min(axis=1)
        lines = [r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
                 r" & \multicolumn{2}{c}{Efficient} & \multicolumn{3}{c}{Proportional floors ($\lambda=1$)} & \multicolumn{3}{c}{Maximin} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
                 r"Budget & Value & Min.\ rel. & Value & Min.\ rel. & PoF & Value & Min.\ rel. & PoF \\",
                 r" (GW) & (\$m/yr) & benefit$^f$ & (\$m/yr) & benefit & & (\$m/yr) & benefit & \\", r"\midrule"]
        for B in sorted(fr.budget_gw.unique()):
            e = fr[(fr.budget_gw == B) & (fr.floor_lambda == 0)].iloc[0]
            f1 = fr[(fr.budget_gw == B) & (fr.floor_lambda == 1)].iloc[0]
            mx = fr[(fr.budget_gw == B) & (fr.floor_lambda == -1)].iloc[0]
            lines.append(f"{B:g} & {e.value_musd:.0f} & {100 * e.min_relben:.1f}\\% & {f1.value_musd:.0f} & "
                         f"{100 * f1.min_relben:.1f}\\% & {100 * f1.price_of_fairness:.1f}\\% & {mx.value_musd:.0f} & "
                         f"{100 * mx.min_relben:.1f}\\% & {100 * mx.price_of_fairness:.1f}\\%" + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(TAB, "storage_fair.tex"), "w").write("\n".join(lines))
        print(fr[["budget_gw", "floor_lambda", "value_musd", "min_relben", "price_of_fairness"] + [c for c in fr.columns if c.startswith("gw_")]].round(3).to_string())

    # ---------------- income gradient -------------------------------------
    inc = pd.read_csv(os.path.join(ROOT, "results", "income_by_area.csv")).set_index("code")
    Ck = r["daily_cost"].sum(axis=2)
    k0 = meth.index("Official")
    okk = Ck[k0] > 0
    codes = r["code"][okk]
    ratio = Ck[:, okk] / Ck[k0, okk][None]
    grp = inc.loc[codes, "income_group"].values
    lgdp = np.log(inc.loc[codes, "gdp_pc_ppp"].values)
    us = np.array([c.startswith("US-") for c in codes])
    pcpi = inc.loc[codes, "us_state_pcpi"].values
    terc = np.full(len(codes), -1)
    qs = np.nanquantile(pcpi[us], [1 / 3, 2 / 3])
    terc[us] = np.digitize(pcpi[us], qs)
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{All 125 areas} & \multicolumn{3}{c}{U.S.\ balancing authorities} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & High & Upper-middle & Rank corr.\ with & Lowest & Highest & Rank corr.\ with \\",
             r" & income & income & GDP p.c.$^g$ & income third & income third & state income$^g$ \\", r"\midrule"]
    inc_out = {}
    for m in [x for x in ORDER if x in meth]:
        k = meth.index(m)
        hi = Ck[k, okk][grp == "High income"].sum() / Ck[k0, okk][grp == "High income"].sum()
        um = Ck[k, okk][grp != "High income"].sum() / Ck[k0, okk][grp != "High income"].sum()
        rg = stats.spearmanr(ratio[k], lgdp).statistic
        lo_t = Ck[k, okk][terc == 0].sum() / Ck[k0, okk][terc == 0].sum()
        hi_t = Ck[k, okk][terc == 2].sum() / Ck[k0, okk][terc == 2].sum()
        ru = stats.spearmanr(ratio[k][us], pcpi[us]).statistic
        inc_out[m] = dict(high=hi, upper_middle=um, rho_gdp=rg, us_low=lo_t, us_high=hi_t, rho_us=ru)
        f = lambda x: "--" if not np.isfinite(x) else f"{x:.3f}"
        lines.append(f"{LABEL[m]} & {hi:.3f} & {um:.3f} & {f(rg)} & {lo_t:.3f} & {hi_t:.3f} & {f(ru)}" + r" \\")
    lines += [r"\midrule", f"Areas & {int((grp == 'High income').sum())} & {int((grp != 'High income').sum())} & & "
              f"{int((terc == 0).sum())} & {int((terc == 2).sum())} & " + r"\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "global_income.tex"), "w").write("\n".join(lines))
    print("income", {m: {k: round(v, 3) for k, v in d.items()} for m, d in inc_out.items()})
    # uncertainty of the rank correlations: bootstrap over areas; and the
    # correlation among data-rich areas only (data length is confounded with income)
    rng = np.random.default_rng(0)
    yrs_i = (r["valid_hours"].sum(axis=1) / 8760.0)[okk]
    for m in ["GDMA-soft", "GDMA-fair", "Pooled", "NeuralGate"]:
        if m not in meth:
            continue
        k = meth.index(m)
        x, g = ratio[k], lgdp
        fin = np.isfinite(x) & np.isfinite(g)
        bs = [stats.spearmanr(x[fin][ix], g[fin][ix]).statistic
              for ix in (rng.integers(0, fin.sum(), fin.sum()) for _ in range(1000))]
        rich = fin & (yrs_i >= 2.0)
        xu, pu = ratio[k][us], pcpi[us]
        fu = np.isfinite(xu) & np.isfinite(pu)
        bu = [stats.spearmanr(xu[fu][ix], pu[fu][ix]).statistic
              for ix in (rng.integers(0, fu.sum(), fu.sum()) for _ in range(1000))]
        print(f"income rho {m}: GDP {stats.spearmanr(x[fin], g[fin]).statistic:.3f} "
              f"90% CI [{np.nanquantile(bs, .05):.2f}, {np.nanquantile(bs, .95):.2f}]; data-rich only "
              f"{stats.spearmanr(x[rich], g[rich]).statistic:.3f} (n={rich.sum()}); U.S. state income "
              f"CI [{np.nanquantile(bu, .05):.2f}, {np.nanquantile(bu, .95):.2f}]")

    # ---------------- fairness audit and candidate-set robustness -------------
    fair = fairness_table(r, os.path.join(TAB, "global_fairness.tex"))
    specs = [(r"No\\weather", ""), (r"GFS\\temperature", "_weather-gfs"),
             (r"GFS/GEFS\\multi-variable", "_weather-plus"), (r"Realised\\temperature", "_weather-exact")]
    specs = [(n + r"\\(main)" if sfx == MAIN else n, sfx) for n, sfx in specs]
    specs = [(r"\begin{tabular}[b]{@{}c@{}}" + n + r"\end{tabular}", sfx) for n, sfx in specs]
    avail = [(n, sfx) for n, sfx in specs if os.path.exists(os.path.join(ROOT, "results", f"backtest_global_w28_taucalibrated{sfx}.npz"))]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(avail) + "@{}}", r"\toprule",
             "Method & " + " & ".join(n for n, _ in avail) + r" \\", r"\midrule"]
    rels = {}
    for n, sfx in avail:
        rr = load(28, sfx)
        tot = rr["daily_cost"].sum(axis=(1, 2))
        mm = list(rr["methods"])
        rels[n] = dict(zip(mm, tot / tot[mm.index("Official")]))
    for m in rows:
        lines.append(LABEL[m] + " & " + " & ".join(f"{rels[n][m]:.3f}" if m in rels[n] else "--" for n, _ in avail) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "global_specs.tex"), "w").write("\n".join(lines))
    print("candidate-set robustness", {n: {m: round(v, 3) for m, v in d.items()} for n, d in rels.items()})
    print("fairness", {m: {k: round(v, 3) for k, v in d.items()} for m, d in fair.items()})

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
