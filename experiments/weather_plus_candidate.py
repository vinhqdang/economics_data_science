"""Weather+ candidate: a global LightGBM model on archived day-ahead forecasts of
temperature, humidity, wind and solar radiation, and on the GEFS ensemble spread.

Features: those of weather_candidate.py (with GFS temperature forecasts) plus
the operating day's forecast 2-m relative humidity, 10-m wind speed, surface
short-wave radiation and ensemble spread of 2-m temperature, hourly and as daily
means.  All inputs are known on the morning of day d-1.
Outputs: data/processed/global_weather_plus.npy       (N, days, 24) point forecast
         data/processed/global_gefs_spread_daily.npy  (N, days) daily mean spread
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from build_global import LGB_WIN, LGB_EVERY, LGB_MAX_ROWS  # noqa: E402
from weather_candidate import design, gfs_forecasts  # noqa: E402
from download_weather import POINTS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def to_areas(field, g, d, N, Dn, H):
    """(gdays, leads, points) forecasts -> (N, Dn, H) area averages on the panel
    calendar; lead 24+h of the day-(d-1) run gives hour h of day d."""
    pts = [tuple(p) for p in g["points"]]
    pidx = {p: j for j, p in enumerate(pts)}
    leads = g["leads"]
    G = field.shape[0]
    rows = field.transpose(0, 2, 1).reshape(-1, len(leads))
    hourly = np.full((rows.shape[0], H), np.nan)
    ok = np.isfinite(rows).all(axis=1)
    x = 24 + np.arange(H)
    for r in np.flatnonzero(ok):
        hourly[r] = np.interp(x, leads, rows[r])
    hourly = hourly.reshape(G, len(pts), H)
    pos = {str(x): i for i, x in enumerate(d["days"])}
    gi = np.array([pos.get(str(x), -1) for x in g["days"]])
    out = np.full((N, Dn, H), np.nan)
    for i, code in enumerate(d["code"]):
        cols = [pidx[(la, lo)] for la, lo in POINTS[str(code)]]
        with np.errstate(all="ignore"):
            vals = np.nanmean(hourly[:, cols, :], axis=1)
        out[i, gi[gi >= 0]] = vals[gi >= 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    d = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    Y = d["Y"].astype(float)
    N, Dn, H = Y.shape
    first = int(d["first_day"])
    OF = np.where(d["has_official"][:, None, None], d["F"][0].astype(float), np.nan)
    days = pd.to_datetime(d["days"])
    month, dow = days.month.values, days.dayofweek.values
    cont = np.array([["North America", "Europe", "Oceania", "Asia", "Africa"].index(c)
                     for c in d["continent"]], dtype=float)
    W = pd.read_parquet(os.path.join(ROOT, "data", "interim", "weather_t2m.parquet"))
    idx = pd.date_range(d["days"][0], periods=Dn * 24, freq="h", tz="UTC")
    T = W.reindex(idx)[list(d["code"])].values.reshape(Dn, 24, N).transpose(2, 0, 1)
    Tf = gfs_forecasts(d, N, Dn, H)

    g = np.load(os.path.join(ROOT, "data", "interim", "weather_extra.npz"), allow_pickle=True)
    rh = to_areas(g["rh"], g, d, N, Dn, H)
    ws = to_areas(np.sqrt(g["u10"] ** 2 + g["v10"] ** 2), g, d, N, Dn, H)
    sw = to_areas(g["dswrf"], g, d, N, Dn, H)
    sp = to_areas(g["t2m_spread"], g, d, N, Dn, H)
    with np.errstate(all="ignore"):
        sp_daily = np.nanmean(sp, axis=2)
    np.save(os.path.join(ROOT, "data", "processed", "global_gefs_spread_daily.npy"), sp_daily)
    for name, a in [("rh", rh), ("wind", ws), ("dswrf", sw), ("spread", sp)]:
        print(name, "coverage", round(float(np.isfinite(a).any(axis=2).mean()), 3), "mean", round(float(np.nanmean(a)), 2))

    def build(days_):
        X, s = design(Y, OF, Tf, days_, month, dow, cont)
        with np.errstate(all="ignore"):
            prev = np.nanmean(T[:, days_ - 1, :], axis=2, keepdims=True)
        X[:, 12] = np.repeat(prev, H, axis=2).reshape(-1)
        rep = lambda a: np.repeat(np.nanmean(a, axis=2, keepdims=True), H, axis=2)
        with np.errstate(all="ignore"):
            extra = [rh[:, days_], ws[:, days_], sw[:, days_], sp[:, days_],
                     rep(ws[:, days_]), rep(sw[:, days_]), rep(sp[:, days_])]
        X = np.column_stack([X] + [e.reshape(-1) for e in extra])
        return X, s

    params = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1,
                  verbose=-1, seed=0, num_threads=args.threads)
    F = np.full((N, Dn, H), np.nan, dtype=np.float32)
    for d0 in range(first, Dn, LGB_EVERY):
        train = np.arange(max(d0 - LGB_WIN, 21), d0)
        X, s = build(train)
        y = (Y[:, train, :] / s[..., None]).reshape(-1)
        ok = np.flatnonzero(np.isfinite(y) & np.isfinite(X[:, 0]))
        if len(ok) > LGB_MAX_ROWS:
            ok = np.sort(np.random.default_rng(d0).choice(ok, LGB_MAX_ROWS, replace=False))
        model = lgb.train(params, lgb.Dataset(X[ok], y[ok], categorical_feature=[16, 17]),
                          num_boost_round=300)
        test = np.arange(d0, min(d0 + LGB_EVERY, Dn))
        Xt, st = build(test)
        F[:, test, :] = model.predict(Xt).reshape(N, len(test), H) * st[..., None]
        print("  weather+ refit at day", d0, flush=True)
    F = np.maximum(F, 0.0)
    F[:, :first] = np.nan
    np.save(os.path.join(ROOT, "data", "processed", "global_weather_plus.npy"), F)
    ok = np.isfinite(F) & np.isfinite(Y)
    with np.errstate(all="ignore"):
        e = np.abs(F - Y) / np.nanmean(Y, axis=(1, 2))[:, None, None]
    print("weather+ mean scaled abs error", float(np.nanmean(np.where(ok, e, np.nan))))
    imp = model.feature_importance("gain")
    print("gain share of the new weather features", round(float(imp[18:].sum() / imp.sum()), 3))


if __name__ == "__main__":
    main()
