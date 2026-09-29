"""Clean EIA-930 data and build day-ahead candidate forecasts per balancing authority.

Decision problem: before each (UTC) operating day d, the operator of balancing
authority i commits hourly capacity q_{i,d,h}, h = 0..23, using demand observed
up to the end of day d-1 and its own published day-ahead forecast for day d.

Candidate forecasts (M = 7):
  0 Official   the BA's own day-ahead demand forecast (EIA-930 'DF')
  1 Persist    same hour of day d-1
  2 Weekly     same hour of day d-7
  3 Mean7      mean of the same hour over days d-1..d-7
  4 Profile3w  mean of the same hour on days d-7, d-14, d-21
  5 HourlyReg  BA-and-hour specific regression on (d-1, d-7, official, weekday),
               rolling 56-day window, refitted weekly
  6 LightGBM   one global gradient-boosting model across BAs and hours on
               scaled lags, calendar and the official forecast; refitted every
               28 days on the preceding 365 days
Output: data/processed/panel.npz
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.join(os.path.dirname(__file__), "..")
REGIONS = {"US48", "CAL", "CAR", "CENT", "FLA", "MIDA", "MIDW", "NE", "NW", "NY",
           "SE", "SW", "TEN", "TEX"}
START, END = "2019-01-01", "2026-09-27"
METHODS = ["Official", "Persist", "Weekly", "Mean7", "Profile3w", "HourlyReg", "LightGBM"]
FIRST_DAY = 400          # first target day with every candidate defined (~Feb 2020)
REG_WIN, REG_EVERY = 56, 7
LGB_WIN, LGB_EVERY = 365, 28


def clean(s, ref):
    """Drop non-positive values and values implausibly far from a rolling median."""
    s = s.where(s > 0)
    med = ref.rolling(24 * 7, center=True, min_periods=24).median()
    s = s.where((s > 0.3 * med) & (s < 3.0 * med))
    return s.interpolate(limit=3, limit_area="inside")


def to_cube(df, idx):
    """(hours x BAs) frame -> array (N, days, 24)."""
    a = df.reindex(idx).values
    return a.reshape(-1, 24, a.shape[1]).transpose(2, 0, 1)


def regression_forecast(Y, OF):
    N, Dn, H = Y.shape
    F = np.full(Y.shape, np.nan)
    dow = np.arange(Dn) % 7
    for d0 in range(FIRST_DAY - REG_WIN, Dn, REG_EVERY):
        train = np.arange(max(d0 - REG_WIN, 21), d0)
        test = np.arange(d0, min(d0 + REG_EVERY, Dn))
        if len(train) < 28:
            continue
        for i in range(N):
            for h in range(H):
                def design(days):
                    wk = (dow[days][:, None] == np.arange(1, 7)).astype(float)
                    return np.column_stack([np.ones(len(days)), Y[i, days - 1, h],
                                            Y[i, days - 7, h], OF[i, days, h], wk])
                X, y = design(train), Y[i, train, h]
                ok = np.isfinite(X).all(1) & np.isfinite(y)
                if ok.sum() < 20:
                    continue
                b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                F[i, test, h] = design(test) @ b
    return F


def lgb_design(Y, OF, days, month, dow):
    """Feature matrix for all (BA, day in days, hour) cells, demand scaled by the
    BA's mean over the previous 7 days."""
    N, _, H = Y.shape
    s = np.nanmean(Y[:, days[:, None] - np.arange(1, 8)[None, :], :], axis=(2, 3))  # (N, len)
    feats = [
        Y[:, days - 1, :] / s[..., None],
        Y[:, days - 7, :] / s[..., None],
        Y[:, days - 14, :] / s[..., None],
        np.repeat(np.nanmax(Y[:, days - 1, :], axis=2)[..., None], H, 2) / s[..., None],
        np.repeat(np.nanmean(Y[:, days - 1, :], axis=2)[..., None], H, 2) / s[..., None],
        OF[:, days, :] / s[..., None],
        np.broadcast_to(np.arange(H), (N, len(days), H)).astype(float),
        np.broadcast_to(dow[days][None, :, None], (N, len(days), H)).astype(float),
        np.broadcast_to(month[days][None, :, None], (N, len(days), H)).astype(float),
        np.broadcast_to(np.arange(N)[:, None, None], (N, len(days), H)).astype(float),
    ]
    X = np.stack(feats, axis=-1).reshape(-1, len(feats))
    return X, s


def lightgbm_forecast(Y, OF, month, dow):
    N, Dn, H = Y.shape
    F = np.full(Y.shape, np.nan)
    params = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1,
                  verbose=-1, seed=0, num_threads=4)
    for d0 in range(FIRST_DAY, Dn, LGB_EVERY):
        train = np.arange(max(d0 - LGB_WIN, 21), d0)
        X, s = lgb_design(Y, OF, train, month, dow)
        y = (Y[:, train, :] / s[..., None]).reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X[:, 0])
        model = lgb.train(params, lgb.Dataset(X[ok], y[ok], categorical_feature=[9]),
                          num_boost_round=400)
        test = np.arange(d0, min(d0 + LGB_EVERY, Dn))
        Xt, st = lgb_design(Y, OF, test, month, dow)
        F[:, test, :] = model.predict(Xt).reshape(N, len(test), H) * st[..., None]
        print("  lightgbm refit at day", d0, flush=True)
    return F


def main():
    D = pd.read_parquet(os.path.join(ROOT, "data", "interim", "eia_D.parquet"))
    DF = pd.read_parquet(os.path.join(ROOT, "data", "interim", "eia_DF.parquet"))
    TI = pd.read_parquet(os.path.join(ROOT, "data", "interim", "eia_TI.parquet"))
    idx = pd.date_range(START, END, freq="h", tz="UTC", inclusive="left")
    D, DF, TI = D.reindex(idx), DF.reindex(idx), TI.reindex(idx)
    cols = [c for c in D.columns if c not in REGIONS and c in DF.columns]
    Dc = pd.DataFrame({c: clean(D[c], D[c]) for c in cols})
    DFc = pd.DataFrame({c: clean(DF[c], Dc[c]) for c in cols})
    keep = [c for c in cols if Dc[c].notna().mean() >= 0.97 and DFc[c].notna().mean() >= 0.93
            and Dc[c].mean() >= 50
            and (DFc[c] - Dc[c]).abs().median() / Dc[c].median() < 0.15]
    print("kept", len(keep), "of", len(cols), "BAs; dropped:", sorted(set(cols) - set(keep)))
    Dc, DFc = Dc[keep], DFc[keep]

    Y = to_cube(Dc, idx)
    OF = to_cube(DFc, idx)
    N, Dn, H = Y.shape
    days = pd.date_range(START, periods=Dn, freq="D")
    month, dow = days.month.values, days.dayofweek.values

    F = np.full((len(METHODS), N, Dn, H), np.nan)
    F[0] = OF
    lag = lambda k: np.concatenate([np.full((N, k, H), np.nan), Y[:, :-k]], axis=1)
    F[1] = lag(1)
    F[2] = lag(7)
    F[3] = np.nanmean(np.stack([lag(k) for k in range(1, 8)]), axis=0)
    F[4] = np.nanmean(np.stack([lag(7), lag(14), lag(21)]), axis=0)
    print("regression ...", flush=True)
    F[5] = regression_forecast(Y, OF)
    print("lightgbm ...", flush=True)
    F[6] = lightgbm_forecast(Y, OF, month, dow)
    F[:, :, :FIRST_DAY] = np.nan
    # a missing candidate value is replaced by the mean of the available ones
    fill = np.nanmean(F, axis=0)
    F = np.where(np.isfinite(F), F, fill[None])
    F = np.maximum(F, 0.0)

    # interchange flexibility (pre-sample 2019): range of total interchange
    ti = TI.reindex(columns=keep).loc[:"2019-12-31"]
    dm = Dc.loc[:"2019-12-31"].mean()
    flex = ((ti.quantile(0.95) - ti.quantile(0.05)) / dm).fillna(0.0).values

    names = pd.read_csv(os.path.join(ROOT, "data", "interim", "ba_names.csv"), index_col=0).iloc[:, 0]
    np.savez_compressed(os.path.join(ROOT, "data", "processed", "panel.npz"),
                        Y=Y.astype(np.float32), F=F.astype(np.float32), ba=np.array(keep),
                        names=np.array([names.get(c, c) for c in keep]),
                        days=days.strftime("%Y-%m-%d").values, flex=flex,
                        methods=np.array(METHODS), first_day=FIRST_DAY)
    ok = np.isfinite(Y[:, FIRST_DAY:])
    for m, name in enumerate(METHODS):
        e = np.abs(F[m][:, FIRST_DAY:] - Y[:, FIRST_DAY:]) / np.nanmean(Y[:, FIRST_DAY:], axis=(1, 2))[:, None, None]
        print(f"{name:10s} mean scaled abs error {np.nanmean(np.where(ok, e, np.nan)):.4f}")


if __name__ == "__main__":
    main()
