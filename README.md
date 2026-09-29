# Grouped Decision-Focused Model Averaging (GDMA)

Research code and manuscript for

> **Grouped Decision-Focused Model Averaging for System-Wide Capacity Commitment: Evidence from the U.S. Power Grid**
> Quang-Vinh Dang, British University Vietnam.
> Prepared for the *Journal of Management Science and Engineering* special issue
> [Economics and Management Empowered by Data Science](https://www.sciencedirect.com/special-issue/330213/economics-and-management-empowered-by-data-science) (deadline 31 Oct 2026).

## The problem

A system planner manages many decision units (here, the 46 balancing authorities of the U.S. grid).
Every day each unit must commit capacity for the next day's hourly demand. A shortfall
costs more than a surplus, and how much more depends on how well the unit can import from
its neighbours. Many candidate commitment rules are available: the operator's own day-ahead
forecast, persistence and calendar rules, regressions, and machine learning.
The question is how to combine them.

## Research gap

| Existing approach | Limitation for system-wide commitment |
|---|---|
| Per-series forecast combination / model averaging (JMA, K-fold CV, forward validation) | weights are noisy with short windows (the forecast-combination puzzle) and are chosen for accuracy, not for decision cost |
| Pooled combination | one rule for units that face different conditions |
| Quantile / general-loss model averaging, CQRA | decision-relevant loss, but one series at a time |
| Data-driven newsvendor, predict-then-optimise | learns decisions directly, but does not combine existing forecasting systems across heterogeneous units |
| Latent-group panel models (Bonhomme–Manresa, C-Lasso) | group regression parameters under squared loss |

No existing method **jointly learns latent segments of decision units and segment-specific
combination weights by minimising the realised, unit-specific decision cost**. That is GDMA.

## Method in one line

```
min over memberships g(i) in {1..G} and weights w_1..w_G in the simplex:
    sum_i sum_t  u_i (y_it - q_it'w_g(i))^+  +  o_i (q_it'w_g(i) - y_it)^+
```

solved by a k-means-type alternation (regret-space k-means++ seeding, pooled convex weight
step), with G chosen by a time-ordered hold-out and the one-standard-error rule.
G = 1 gives pooled weights and G = N gives unit-specific weights.

Theory (paper, Section 4): a finite-sample oracle inequality that separates the price of
learning memberships (log G / T per unit) from the price of learning weights
(G M log(NT) / NT); asymptotic optimality; recovery of the segments; and exact equivalence
with the estimator that knows the segments.

## Repository layout

```
src/gdma/core.py            estimator: losses, simplex solvers, grouped fit, G selection, benchmarks
experiments/extract_eia.py  stream EIA-930 bulk file -> hourly demand, day-ahead forecast, interchange
experiments/build_panel.py  cleaning, 46-BA panel, 7 candidate forecasts (incl. global LightGBM)
experiments/backtest.py     rolling out-of-sample commitment backtest (10 methods)
experiments/simulation.py   Monte Carlo study
experiments/analyze.py      tables and figures for the paper
paper/                      LaTeX manuscript (main.tex) and compiled PDF
results/                    logs, summary tables and simulation output
```

## Reproduce

```bash
pip install numpy scipy pandas scikit-learn lightgbm matplotlib pyarrow
mkdir -p data/raw data/interim data/processed results
curl -L https://www.eia.gov/opendata/bulk/EBA.zip -o data/raw/EBA.zip   # ~700 MB, public, no key
python experiments/extract_eia.py
python experiments/build_panel.py
for w in 7 14 28 56; do python experiments/backtest.py --window $w; done
for t in 0.5 0.8 0.9 0.95; do python experiments/backtest.py --window 28 --tau $t; done
python experiments/simulation.py
python experiments/analyze.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
