# From Forecasts to Batteries: Grouped Decision-Focused Model Averaging

Research code and manuscript for

> **From Forecasts to Batteries: Grouped Decision-Focused Model Averaging for Capacity Commitment and Storage Siting on Five Continents**
> Quang-Vinh Dang, British University Vietnam.
> Prepared for the *Journal of Management Science and Engineering* special issue
> [Economics and Management Empowered by Data Science](https://www.sciencedirect.com/special-issue/330213/economics-and-management-empowered-by-data-science) (deadline 31 October 2026).

The compiled manuscript is `paper/main.pdf`.

## The management problem

Grid operators must commit capacity a day ahead: a shortfall costs more than a surplus, and how much more
depends on how well a grid can import from its neighbours. Planners must also decide where to install
new batteries. Both decisions rest on how the available demand forecasts are combined: operators'
own forecasts, persistence and calendar rules, regressions, a global LightGBM model and the
pretrained foundation model Chronos.

## Research gap

| Existing approach | Limitation |
|---|---|
| Per-series model averaging / forecast combination (JMA, K-fold CV, forward validation) | noisy weights with short windows (forecast-combination puzzle); chosen for accuracy, not decision cost |
| Pooled combination | one rule for systems that face different conditions |
| Quantile / general-loss model averaging, CQRA | decision-relevant loss, but one series at a time |
| Neural mixture-of-experts gates, FFORMA | thousands of parameters, hard to audit, no guarantees for the combined decision |
| Data-driven newsvendor, predict-then-optimise | learns decisions, but does not combine existing forecasting systems across heterogeneous units |
| Latent-group panel models (Bonhomme–Manresa, C-Lasso) | group regression parameters under squared loss |
| Storage siting models | take the uncertainty that storage must absorb as given |

No existing method **jointly learns latent segments of decision units and segment-specific combination
weights by minimising the realised, unit-specific decision cost**, or links the resulting operational
risk to where new storage should go. That is GDMA.

## Method

```
min over memberships g(i) in {1..G} and weights w_1..w_G in the simplex:
    sum_i sum_t  u_i (y_it - q_it'w_g(i))^+  +  o_i (q_it'w_g(i) - y_it)^+
```

- k-means-type alternation with regret-space k-means++ seeding; each weight step is a small convex program
- **soft GDMA**: w_i = (1 - kappa) w_i^unit + kappa w_segment, nesting pooled weights (G = 1),
  unit-specific weights (kappa = 0) and shrinkage
- G by time-ordered hold-out with the one-standard-error rule, then kappa by hold-out
- **storage stage**: value curves of battery power from the learned rule's residual shortfalls,
  greedy (exact) allocation of a storage budget across grid areas

**Theory**: a finite-sample oracle inequality separating the price of learning memberships
(log G / T per unit) from the price of learning weights (G M log(NT) / NT); asymptotic optimality;
segment recovery; exact equivalence with the estimator that knows the segments.

## Main results

| | Soft GDMA saving vs best single forecaster |
|---|---|
| 46 U.S. balancing authorities (vs official forecasts) | 18–20% |
| 125 grid areas, five continents (1.66 TW), with GFS/GEFS weather forecasts | 18.7% (fair version 19.0%) |
| vs decision-focused neural gate (~7,000 parameters) | 10 percentage points |
| grid areas made worse off than their own best forecaster | 4% (fair version: 2%; neural gate: 32%) |
| Winter Storm Elliott, excess cost over U.S. official forecasts | 3.1% (6.9% with temperature forecasts only) |

Without storage, soft GDMA is cheaper than the status quo with 60 GW of optimally sited batteries,
and is worth 5–7 GW relative to standard combination rules. Siting a storage programme on the
status-quo error profile loses up to 15% of its value; maximin-fair siting protects the worst-served
continent at a price of fairness of about 1%. No systematic income gradient in the benefits.
During Winter Storm Elliott a candidate whose safety margin widens with the GEFS ensemble spread is
8% cheaper than the operators' own forecasts on its own.

## Repository layout

```
src/gdma/core.py              estimator: losses, simplex solvers, (soft) grouped fit, G selection, benchmarks
src/gdma/storage.py           battery dispatch, value curves, budget allocation
experiments/extract_eia.py    EIA-930 bulk file -> hourly demand, day-ahead forecast, interchange
experiments/build_panel.py    U.S. panel (46 BAs) and seven candidate forecasts
experiments/backtest.py       U.S. rolling backtest (and GDMA variants with --variants)
experiments/download_global.py  Europe (Energy-Charts), Australia (AEMO), China, Taiwan, Thailand, Algeria
experiments/build_global.py   global panel (125 areas) and candidate forecasts
experiments/chronos_candidate.py  zero-shot Chronos-Bolt forecasts
experiments/backtest_global.py    global rolling backtest
experiments/neural_gate.py    decision-focused neural mixture-of-experts benchmark
experiments/download_weather.py   NASA POWER hourly temperature at 169 load centres
experiments/download_gfs.py   archived NOAA GFS day-ahead temperature forecasts (2021-2026, AWS open data)
experiments/weather_candidate.py  weather-aware LightGBM candidate (GFS forecasts / realised temperature)
experiments/download_weather_extra.py  archived GFS humidity, wind, radiation and GEFS temperature spread
experiments/weather_plus_candidate.py  multi-variable weather candidate and daily ensemble spread
experiments/events_global.py  Winter Storm Elliott comparison and stand-alone candidate costs
experiments/income.py         World Bank / IMF GDP per capita and income groups, BEA state income
experiments/storage_siting.py battery siting (efficient, proportional floors, maximin)
experiments/simulation.py     Monte Carlo study
experiments/analyze.py, analyze_global.py, sim_tables.py   tables and figures
paper/                        LaTeX manuscript and compiled PDF
results/                      logs, summary tables and small result files
```

## Reproduce

```bash
pip install numpy scipy pandas scikit-learn lightgbm matplotlib pyarrow openpyxl torch chronos-forecasting pygrib
mkdir -p data/raw data/interim data/processed results
curl -L https://www.eia.gov/opendata/bulk/EBA.zip -o data/raw/EBA.zip       # public, no key
python experiments/extract_eia.py && python experiments/build_panel.py
for w in 7 14 28 56; do python experiments/backtest.py --window $w; python experiments/backtest.py --window $w --variants; done
for t in 0.5 0.8 0.9 0.95; do python experiments/backtest.py --window 28 --tau $t; python experiments/backtest.py --window 28 --tau $t --variants; done
python experiments/download_global.py && python experiments/build_global.py && python experiments/chronos_candidate.py
python experiments/download_weather.py && python experiments/download_gfs.py
python experiments/weather_candidate.py --source gfs && python experiments/weather_candidate.py --noise 0
python experiments/download_weather_extra.py && python experiments/weather_plus_candidate.py
python experiments/income.py            # needs World Bank, IMF and FRED files in data/raw/income
for w in 28 14; do python experiments/backtest_global.py --window $w; python experiments/neural_gate.py --window $w; done
python experiments/backtest_global.py --window 28 --weather gfs && python experiments/neural_gate.py --window 28 --weather gfs
python experiments/backtest_global.py --window 28 --weather plus && python experiments/neural_gate.py --window 28 --weather plus
python experiments/backtest_global.py --window 28 --weather exact
TAG=global_w28_taucalibrated_weather-plus python experiments/storage_siting.py
python experiments/backtest_global.py --window 28 --tau 0.9
python experiments/storage_siting.py
python experiments/simulation.py
python experiments/analyze.py && python experiments/analyze_global.py && python experiments/events_global.py && python experiments/sim_tables.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
