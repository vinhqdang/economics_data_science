# From Forecasts to Batteries: Grouped Decision-Focused Model Averaging

Research code and manuscript for

> **From Forecasts to Batteries: Grouped Decision-Focused Model Averaging for Capacity Commitment and Storage Siting on Five Continents**
> Quang-Vinh Dang, British University Vietnam.
> Prepared for the *Journal of Management Science and Engineering* special issue
> [Economics and Management Empowered by Data Science](https://www.sciencedirect.com/special-issue/330213/economics-and-management-empowered-by-data-science) (deadline 31 October 2026).

The compiled manuscript is `paper/gdma_storage_siting_manuscript.pdf`.

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

All rules use only the information available at the day-ahead gate closure (demand up to the end of day d-2, official forecast for day d, weather forecasts issued before the closure).

| | Soft GDMA saving |
|---|---|
| 46 U.S. balancing authorities, vs their official forecasts | 13-17% (16.5% at W=28) |
| 125 grid areas, five continents, vs the reference forecaster | 11.6% (fair version 11.9%) |
| 125 grid areas, vs each area's best single candidate | 4.2% |
| vs decision-focused neural gate (~7,000 parameters) | 7 percentage points |
| grid areas made worse off than their own best candidate | 6% (fair version 3%; pooled 24%; neural gate 42%) |
| Winter Storm Elliott, excess cost over the operators' forecasts | 9.5% (20% without weather forecasts) |

The 90% model confidence set contains two-step clustering, GDMA, soft GDMA and fair soft GDMA; the grouped variants
cannot be separated, and decision-focused shrinkage comes close. Forecast-then-commit and an unconstrained
quantile-regression combination do worse, and fail in the tail (costliest week 2-9 times the reference's).
In a stylised storage model the reference rule needs 34 GW of optimally sited batteries to match soft GDMA without
any; siting on the wrong rule's error profile loses 2-4% of a programme's value; maximin-fair siting costs 7-9%.
No systematic income gradient is found, though the data cannot exclude a moderate one.

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
experiments/timing.py         information available at the day-ahead decision
experiments/mcs.py            model confidence set and block-bootstrap intervals
experiments/run_all.py        the whole empirical pipeline in dependency order (restartable)
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
python experiments/extract_eia.py
python experiments/download_global.py && python experiments/download_weather.py && python experiments/download_gfs.py
python experiments/download_weather_extra.py
python experiments/income.py            # needs World Bank, IMF and FRED files in data/raw/income
python experiments/run_all.py --jobs 3  # panels, candidates, all backtests, neural gates, storage siting, simulation
python experiments/backtest_global.py --window 28 --tau 0.99 --weather plus   # scarcity-level critical ratio (tail-risk table)
python experiments/analyze.py && python experiments/analyze_global.py && python experiments/events_global.py && python experiments/sim_tables.py
cd paper && pdflatex gdma_storage_siting_manuscript && bibtex gdma_storage_siting_manuscript && pdflatex gdma_storage_siting_manuscript && pdflatex gdma_storage_siting_manuscript
```
