"""Weather-aware candidate: a global LightGBM model with temperature features.

The model adds to the features of the LightGBM candidate in build_global.py the
area's hourly 2-m temperature on the operating day (NASA POWER), its daily mean,
minimum and maximum, heating and cooling degrees (base 18 C), and the previous
day's mean temperature.  With --source gfs all temperatures are archived GFS
forecasts known at decision time (timing.py).  Using realised temperature is an
*ex post* forecast in the sense of Hong and Fan (2016): it bounds from above
what a day-ahead weather forecast could contribute.  With --noise S the operating-day temperatures are
perturbed by a serially correlated error (AR(1), coefficient 0.9 across hours)
with standard deviation S degrees, drawn independently for every area and day,
to mimic an imperfect day-ahead weather forecast.
Output: data/processed/global_weather_noise{S}.npy with shape (N, days, 24).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from build_global import LGB_WIN, LGB_EVERY, LGB_MAX_ROWS  # noqa: E402
from timing import INFO_LAG, nwp_mask  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def design(Y, OF, Tm, days, month, dow, cont):
    N, _, H = Y.shape
    with np.errstate(all="ignore"):
        s = np.nanmean(Y[:, days[:, None] - np.arange(INFO_LAG, INFO_LAG + 7)[None, :], :], axis=(2, 3))
        t = Tm[:, days, :]
        tmean = np.nanmean(t, axis=2, keepdims=True)
        rep = lambda a: np.repeat(a, H, axis=2)
        feats = [
            Y[:, days - INFO_LAG, :] / s[..., None],
            Y[:, days - 7, :] / s[..., None],
            Y[:, days - 14, :] / s[..., None],
            rep(np.nanmax(Y[:, days - INFO_LAG, :], axis=2, keepdims=True)) / s[..., None],
            rep(np.nanmean(Y[:, days - INFO_LAG, :], axis=2, keepdims=True)) / s[..., None],
            OF[:, days, :] / s[..., None],
            t,
            rep(tmean),
            rep(np.nanmin(t, axis=2, keepdims=True)),
            rep(np.nanmax(t, axis=2, keepdims=True)),
            np.maximum(18.0 - t, 0.0),
            np.maximum(t - 18.0, 0.0),
            rep(np.nanmean(Tm[:, days - INFO_LAG, :], axis=2, keepdims=True)),
            np.broadcast_to(np.arange(H), (N, len(days), H)).astype(float),
            np.broadcast_to(dow[days][None, :, None], (N, len(days), H)).astype(float),
            np.broadcast_to(month[days][None, :, None], (N, len(days), H)).astype(float),
            np.broadcast_to(np.arange(N)[:, None, None], (N, len(days), H)).astype(float),
            np.broadcast_to(cont[:, None, None], (N, len(days), H)).astype(float),
        ]
    return np.stack(feats, axis=-1).reshape(-1, len(feats)), s


def gfs_forecasts(d, N, Dn, H):
    """Area-average GFS day-ahead temperature forecasts on the panel calendar;
    lead 24+h hours of the run initialised at 00 UTC of day d-1 gives hour h of
    day d (3-hourly leads linearly interpolated).  NaN before 2021."""
    sys.path.insert(0, os.path.dirname(__file__))
    from download_weather import POINTS
    g = np.load(os.path.join(ROOT, "data", "interim", "gfs_t2m.npz"), allow_pickle=True)
    pts = [tuple(p) for p in g["points"]]
    pidx = {p: j for j, p in enumerate(pts)}
    leads = g["leads"]
    hourly = np.array([np.interp(24 + np.arange(H), leads, row) if np.isfinite(row).all() else np.full(H, np.nan)
                       for row in g["t2m"].transpose(0, 2, 1).reshape(-1, len(leads))])
    hourly = hourly.reshape(len(g["days"]), len(pts), H)          # (gdays, points, H)
    day_pos = {str(x): i for i, x in enumerate(d["days"])}
    Tf = np.full((N, Dn, H), np.nan)
    for i, code in enumerate(d["code"]):
        cols = [pidx[(round(la, 2), round(lo, 2))] if (round(la, 2), round(lo, 2)) in pidx else pidx[(la, lo)]
                for la, lo in POINTS[str(code)]]
        vals = np.nanmean(hourly[:, cols, :], axis=1)
        for gi, day in enumerate(g["days"]):
            if str(day) in day_pos:
                Tf[i, day_pos[str(day)]] = vals[gi]
    Tf[~nwp_mask(d["continent"])] = np.nan        # run not yet available at gate closure
    return Tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--source", default="reanalysis", choices=["reanalysis", "gfs"],
                    help="operating-day temperature: realised (reanalysis, optionally with "
                         "--noise) or archived GFS day-ahead forecasts")
    args = ap.parse_args()
    d = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    Y = d["Y"].astype(float)
    N, Dn, H = Y.shape
    first = int(d["first_day"])
    # the official forecast is candidate 0 where it exists
    OF = np.where(d["has_official"][:, None, None], d["F"][0].astype(float), np.nan)
    days = pd.to_datetime(d["days"])
    month, dow = days.month.values, days.dayofweek.values
    cont = np.array([["North America", "Europe", "Oceania", "Asia", "Africa"].index(c)
                     for c in d["continent"]], dtype=float)
    W = pd.read_parquet(os.path.join(ROOT, "data", "interim", "weather_t2m.parquet"))
    idx = pd.date_range(d["days"][0], periods=Dn * 24, freq="h", tz="UTC")
    T = W.reindex(idx)[list(d["code"])].values.reshape(Dn, 24, N).transpose(2, 0, 1)
    # temperature for the operating day: GFS forecasts, or realised (upper bound)
    Tf = T.copy()
    if args.source == "gfs":
        Tf = gfs_forecasts(d, N, Dn, H)
    if args.noise > 0:
        rng = np.random.default_rng(12345)
        z = rng.standard_normal((N, Dn, H))
        e = np.empty_like(z)
        e[:, :, 0] = z[:, :, 0]
        for h in range(1, H):
            e[:, :, h] = 0.9 * e[:, :, h - 1] + np.sqrt(1 - 0.81) * z[:, :, h]
        Tf = T + args.noise * e

    params = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1,
                  verbose=-1, seed=0, num_threads=args.threads)
    F = np.full((N, Dn, H), np.nan, dtype=np.float32)

    def build(days_):
        # operating-day temperature from Tf.  The previous day's temperature is
        # not yet observed at decision time: with GFS it is the forecast for day
        # d-1 from the run of day d-2; the realised-temperature upper bound uses
        # the reanalysis value.
        X, s = design(Y, OF, Tf, days_, month, dow, cont)
        with np.errstate(all="ignore"):
            src = Tf if args.source == "gfs" else T
            prev = np.nanmean(src[:, days_ - 1, :], axis=2, keepdims=True)
        X[:, 12] = np.repeat(prev, H, axis=2).reshape(-1)
        return X, s

    for d0 in range(first, Dn, LGB_EVERY):
        train = np.arange(max(d0 - INFO_LAG + 1 - LGB_WIN, 21), d0 - INFO_LAG + 1)
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
        print("  weather lightgbm refit at day", d0, flush=True)
    F = np.maximum(F, 0.0)
    F[:, :first] = np.nan
    name = "global_weather_gfs.npy" if args.source == "gfs" else f"global_weather_noise{args.noise:g}.npy"
    np.save(os.path.join(ROOT, "data", "processed", name), F)
    ok = np.isfinite(F) & np.isfinite(Y)
    with np.errstate(all="ignore"):
        e = np.abs(F - Y) / np.nanmean(Y, axis=(1, 2))[:, None, None]
    print(f"weather candidate (noise {args.noise}) mean scaled abs error",
          float(np.nanmean(np.where(ok, e, np.nan))))


if __name__ == "__main__":
    main()
