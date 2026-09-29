"""LaTeX table for the Monte Carlo study (reads results/simulation_raw.csv)."""
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
METHODS = ["EW", "Pooled", "PerUnit", "Shrink", "KMeans2S", "GDMA"]
HEAD = ["EW", "Pooled", "Per-unit", "Shrink", "2-step", "GDMA"]


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "simulation_raw.csv"))
    g = df.groupby(["design", "N", "T"])
    mean = g[METHODS].mean()
    extra = g[["G_hat", "ARI", "ARI_km"]].mean()
    reps = g.size().min()
    lines = [r"\begin{tabular}{@{}llr" + "c" * len(METHODS) + "ccc@{}}", r"\toprule",
             r" & & & \multicolumn{%d}{c}{Out-of-sample cost / oracle} & & \multicolumn{2}{c}{ARI} \\" % len(METHODS),
             r"\cmidrule(lr){4-%d}\cmidrule(lr){%d-%d}" % (3 + len(METHODS), 5 + len(METHODS), 6 + len(METHODS)),
             "Design & $N$ & $T$ & " + " & ".join(HEAD) + r" & $\bar G$ & GDMA & 2-step \\", r"\midrule"]
    for design in ["grouped", "homogeneous", "continuous"]:
        first = True
        for (d, N, T), row in mean.loc[[design]].iterrows():
            best = row.min()
            cells = [(r"\textbf{%.3f}" % v) if abs(v - best) < 5e-4 else f"{v:.3f}" for v in row.values]
            e = extra.loc[(d, N, T)]
            ari = f"{e.ARI:.2f} & {e.ARI_km:.2f}" if design == "grouped" else "-- & --"
            name = design.capitalize() if first else ""
            first = False
            lines.append(f"{name} & {N} & {T} & " + " & ".join(cells) + f" & {e.G_hat:.1f} & {ari}" + r" \\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    os.makedirs(os.path.join(ROOT, "paper", "tables"), exist_ok=True)
    open(os.path.join(ROOT, "paper", "tables", "simulation.tex"), "w").write("\n".join(lines))
    print("replications per cell:", reps)
    print(mean.round(3).to_string())
    print(extra.round(2).to_string())
    # win rates of GDMA against per-unit weights
    df["gdma_beats_unit"] = df.GDMA < df.PerUnit
    print(df.groupby(["design", "N", "T"]).gdma_beats_unit.mean().round(2).to_string())


if __name__ == "__main__":
    main()
