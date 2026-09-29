"""Build the global panel: hourly demand of 125 grid areas on five continents and
day-ahead candidate forecasts for each of them.

Units
  North America  46 U.S. balancing authorities (EIA-930), 2019-2026, with the
                 operators' own day-ahead forecasts
  Europe         33 national systems (Energy-Charts), 2018-2026
  Oceania        5 NEM regions (AEMO), 2018-2026
  Asia           31 Chinese provinces (2018), 4 Taiwanese grid areas (2018-2022),
                 5 Thai regions (2023-2024)
  Africa         Algeria (Sonelgaz), 2018-Jan 2020
The panel is unbalanced: a unit only enters the evaluation while its data exist.

Candidates are the seven rules of build_panel.py.  Only U.S. units publish an
official forecast; elsewhere candidate 0 is set equal to the global LightGBM
forecast, which leaves the set of attainable combinations unchanged.
Output: data/processed/global_panel.npz
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from build_panel import clean, REGIONS

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "global")
START, END = "2018-01-01", "2026-09-27"
METHODS = ["Official", "Persist", "Weekly", "Mean7", "Profile3w", "HourlyReg", "LightGBM"]
FIRST_DAY = 60
REG_WIN, REG_EVERY = 56, 7
LGB_WIN, LGB_EVERY = 365, 56
LGB_MAX_ROWS = 800_000
CHINA = ["Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia", "Liaoning", "Jilin",
         "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi",
         "Shandong", "Henan", "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan", "Chongqing",
         "Sichuan", "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia",
         "Xinjiang"]
EU_NAMES = dict(at="Austria", be="Belgium", bg="Bulgaria", ch="Switzerland", cz="Czechia",
                de="Germany", dk="Denmark", ee="Estonia", es="Spain", fi="Finland", fr="France",
                gr="Greece", hr="Croatia", hu="Hungary", ie="Ireland", it="Italy", lt="Lithuania",
                lu="Luxembourg", lv="Latvia", nl="Netherlands", no="Norway", pl="Poland",
                pt="Portugal", ro="Romania", rs="Serbia", se="Sweden", si="Slovenia",
                sk="Slovakia", uk="United Kingdom", ba="Bosnia and Herzegovina",
                me="Montenegro", mk="North Macedonia", md="Moldova")


def hourly(s):
    """Any sub-hourly UTC series -> hourly mean (MW = MWh per hour)."""
    return s.sort_index().resample("1h").mean()


def load_us(idx):
    D = pd.read_parquet(os.path.join(ROOT, "data", "interim", "eia_D.parquet")).reindex(idx)
    DF = pd.read_parquet(os.path.join(ROOT, "data", "interim", "eia_DF.parquet")).reindex(idx)
    TI = pd.read_parquet(os.path.join(ROOT, "data", "interim", "eia_TI.parquet")).reindex(idx)
    us = np.load(os.path.join(ROOT, "data", "processed", "panel.npz"), allow_pickle=True)
    keep = list(us["ba"])
    names = [str(n).replace("Demand for ", "").split(", hourly")[0] for n in us["names"]]
    dem = pd.DataFrame({c: clean(D[c], D[c]) for c in keep})
    fc = pd.DataFrame({c: clean(DF[c], dem[c]) for c in keep})
    ti = TI.reindex(columns=keep).loc["2019"]
    flex = ((ti.quantile(0.95) - ti.quantile(0.05)) / dem.loc["2019"].mean()).fillna(0).values
    meta = [dict(code=f"US-{c}", name=n, continent="North America", flex=f)
            for c, n, f in zip(keep, names, flex)]
    return dem.values, fc.values, meta


def load_europe(idx):
    cols, meta = [], []
    for c, name in EU_NAMES.items():
        parts, cb = [], []
        for p in sorted(glob.glob(os.path.join(RAW, "europe", f"{c}_*.json"))):
            d = json.load(open(p))
            t = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
            pt = {x["name"]: x["data"] for x in d["production_types"]}
            if "Load" not in pt:
                continue
            parts.append(pd.Series(pt["Load"], index=t, dtype="float64"))
            if "Cross border electricity trading" in pt:
                cb.append(pd.Series(pt["Cross border electricity trading"], index=t, dtype="float64"))
        if not parts:
            continue
        s = hourly(pd.concat(parts)[lambda x: ~x.index.duplicated()]).reindex(idx)
        s = clean(s, s)
        if s.notna().mean() < 0.5:
            continue
        if cb:
            x = hourly(pd.concat(cb)[lambda x: ~x.index.duplicated()]).reindex(idx)
            first = s.first_valid_index().year
            xf = x.loc[str(first)]
            if xf.notna().mean() < 0.5 or (xf.fillna(0) == 0).mean() > 0.5:
                f = np.nan               # no usable cross-border record
            else:
                f = float((xf.quantile(0.95) - xf.quantile(0.05)) / s.loc[str(first)].mean())
        else:
            f = np.nan
        cols.append(s.values)
        meta.append(dict(code=f"EU-{c.upper()}", name=name, continent="Europe", flex=f))
    return np.column_stack(cols), meta


def load_aemo(idx):
    cols, meta = [], []
    names = dict(NSW1="New South Wales", QLD1="Queensland", VIC1="Victoria",
                 SA1="South Australia", TAS1="Tasmania")
    for reg, name in names.items():
        fs = sorted(glob.glob(os.path.join(RAW, "aemo", f"{reg}_*.csv")))
        df = pd.concat([pd.read_csv(f, usecols=["SETTLEMENTDATE", "TOTALDEMAND"])
                        for f in fs if os.path.getsize(f) > 1000])
        # NEM time is UTC+10 without daylight saving; values are period-ending
        t = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S") - pd.Timedelta("10h")
        s = pd.Series(df["TOTALDEMAND"].values, index=(t - pd.Timedelta("1min")).dt.tz_localize("UTC"))
        s = clean(hourly(s[~s.index.duplicated()]).reindex(idx), hourly(s[~s.index.duplicated()]).reindex(idx))
        cols.append(s.values)
        meta.append(dict(code=f"AU-{reg[:-1]}", name=name, continent="Oceania", flex=np.nan))
    return np.column_stack(cols), meta


def load_asia_africa(idx):
    cols, meta = [], []
    # China, 2018, hour index 1..8760 in Beijing time (UTC+8), MWh per hour
    ch = pd.read_csv(os.path.join(RAW, "china_load.csv"), sep=";", encoding="utf-8-sig")
    t = pd.date_range("2018-01-01 00:00", periods=len(ch), freq="h", tz="UTC") - pd.Timedelta("8h")
    for j, name in enumerate(CHINA):
        s = pd.Series(ch.iloc[:, j + 1].astype(float).values, index=t).reindex(idx)
        cols.append(clean(s, s).values)
        meta.append(dict(code=f"CN-{j + 1:02d}", name=name, continent="Asia", flex=np.nan))
    # Taiwan grid areas, 10-min, Taiwan time (UTC+8); isolated island grid
    tw = pd.read_csv(os.path.join(RAW, "taiwan_loadarea.csv"), parse_dates=["datetime"])
    tt = (tw["datetime"] - pd.Timedelta("8h")).dt.tz_localize("UTC")
    for area in ["north", "central", "south", "east"]:
        s = hourly(pd.Series(tw[area].astype(float).values, index=tt)).reindex(idx)
        cols.append(clean(s, s).values)
        meta.append(dict(code=f"TW-{area[0].upper()}", name=f"Taiwan {area}", continent="Asia", flex=0.0))
    # Thailand regions, hourly, Thai time (UTC+7)
    th = pd.concat([pd.read_csv(os.path.join(RAW, f"thailand_{y}.csv"), encoding="utf-8-sig")
                    for y in (2023, 2024)])
    tt = pd.to_datetime(th["datetime"], format="%d/%m/%Y %H:%M").dt.floor("h")
    tt = (tt - pd.Timedelta("7h")).dt.tz_localize("UTC")
    th_codes = dict(north="N", south="S", metropolitan="BKK", central="C", northeast="NE")
    for reg in ["north", "south", "metropolitan", "central", "northeast"]:
        s = pd.Series(th[f"{reg}_demand"].astype(float).values, index=tt)
        s = s[~s.index.duplicated()].reindex(idx)
        cols.append(clean(s, s).values)
        meta.append(dict(code=f"TH-{th_codes[reg]}", name=f"Thailand {reg}", continent="Asia", flex=np.nan))
    # Algeria, daily rows of 24 hourly values (hour-ending, local time UTC+1)
    al = pd.read_excel(os.path.join(RAW, "algeria.xlsx"), sheet_name="Feuil1")
    # the sheet repeats 274 days verbatim; keep one copy of each date
    al = al.assign(Date=pd.to_datetime(al["Date"])).drop_duplicates("Date").sort_values("Date")
    vals = al.iloc[:, 1:25].values.astype(float).ravel()
    t = (pd.to_datetime(al["Date"]).values[:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    s = pd.Series(vals, index=pd.DatetimeIndex(t).tz_localize("UTC") - pd.Timedelta("1h")).reindex(idx)
    cols.append(clean(s, s).values)
    meta.append(dict(code="DZ", name="Algeria", continent="Africa", flex=np.nan))
    return np.column_stack(cols), meta


def to_cube(a):
    return a.reshape(-1, 24, a.shape[1]).transpose(2, 0, 1)


def regression_forecast(Y, OF, first):
    N, Dn, H = Y.shape
    F = np.full(Y.shape, np.nan)
    dow = np.arange(Dn) % 7
    has_of = np.isfinite(OF).any(axis=(1, 2))
    for d0 in range(first, Dn, REG_EVERY):
        train = np.arange(max(d0 - REG_WIN, 21), d0)
        test = np.arange(d0, min(d0 + REG_EVERY, Dn))
        for i in range(N):
            if not np.isfinite(Y[i, train]).any():
                continue
            for h in range(H):
                def design(days):
                    wk = (dow[days][:, None] == np.arange(1, 7)).astype(float)
                    cols = [np.ones(len(days)), Y[i, days - 1, h], Y[i, days - 7, h]]
                    if has_of[i]:
                        cols.append(OF[i, days, h])
                    return np.column_stack(cols + [wk])
                X, y = design(train), Y[i, train, h]
                ok = np.isfinite(X).all(1) & np.isfinite(y)
                if ok.sum() < 20:
                    continue
                b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                F[i, test, h] = design(test) @ b
    return F


def lgb_design(Y, OF, days, month, dow, cont):
    N, _, H = Y.shape
    with np.errstate(all="ignore"):
        s = np.nanmean(Y[:, days[:, None] - np.arange(1, 8)[None, :], :], axis=(2, 3))
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
            np.broadcast_to(cont[:, None, None], (N, len(days), H)).astype(float),
        ]
    return np.stack(feats, axis=-1).reshape(-1, len(feats)), s


def lightgbm_forecast(Y, OF, month, dow, cont, first):
    N, Dn, H = Y.shape
    F = np.full(Y.shape, np.nan)
    params = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1,
                  verbose=-1, seed=0, num_threads=4)
    for d0 in range(first, Dn, LGB_EVERY):
        train = np.arange(max(d0 - LGB_WIN, 21), d0)
        X, s = lgb_design(Y, OF, train, month, dow, cont)
        y = (Y[:, train, :] / s[..., None]).reshape(-1)
        ok = np.flatnonzero(np.isfinite(y) & np.isfinite(X[:, 0]))
        if len(ok) > LGB_MAX_ROWS:
            ok = np.sort(np.random.default_rng(d0).choice(ok, LGB_MAX_ROWS, replace=False))
        model = lgb.train(params, lgb.Dataset(X[ok], y[ok], categorical_feature=[9, 10]),
                          num_boost_round=300)
        test = np.arange(d0, min(d0 + LGB_EVERY, Dn))
        Xt, st = lgb_design(Y, OF, test, month, dow, cont)
        F[:, test, :] = model.predict(Xt).reshape(N, len(test), H) * st[..., None]
        print("  lightgbm refit at day", d0, "rows", len(ok), flush=True)
    return F


def main():
    idx = pd.date_range(START, END, freq="h", tz="UTC", inclusive="left")
    us_d, us_f, us_m = load_us(idx)
    eu_d, eu_m = load_europe(idx)
    au_d, au_m = load_aemo(idx)
    as_d, as_m = load_asia_africa(idx)
    meta = us_m + eu_m + au_m + as_m
    D = np.column_stack([us_d, eu_d, au_d, as_d])
    OFh = np.column_stack([us_f, np.full((len(idx), D.shape[1] - us_d.shape[1]), np.nan)])
    Y, OF = to_cube(D), to_cube(OFh)
    N, Dn, H = Y.shape
    print("units", N, {c: sum(m["continent"] == c for m in meta) for c in
                       ["North America", "Europe", "Oceania", "Asia", "Africa"]}, flush=True)
    days = pd.date_range(START, periods=Dn, freq="D")
    month, dow = days.month.values, days.dayofweek.values
    cont = np.array([["North America", "Europe", "Oceania", "Asia", "Africa"].index(m["continent"])
                     for m in meta], dtype=float)

    F = np.full((len(METHODS), N, Dn, H), np.nan, dtype=np.float32)
    lag = lambda k: np.concatenate([np.full((N, k, H), np.nan), Y[:, :-k]], axis=1)
    with np.errstate(all="ignore"):
        F[1] = lag(1)
        F[2] = lag(7)
        F[3] = np.nanmean(np.stack([lag(k) for k in range(1, 8)]), axis=0)
        F[4] = np.nanmean(np.stack([lag(7), lag(14), lag(21)]), axis=0)
    print("regression ...", flush=True)
    F[5] = regression_forecast(Y, OF, FIRST_DAY)
    print("lightgbm ...", flush=True)
    F[6] = lightgbm_forecast(Y, OF, month, dow, cont, FIRST_DAY)
    F[0] = np.where(np.isfinite(OF), OF, F[6])
    F[:, :, :FIRST_DAY] = np.nan
    with np.errstate(all="ignore"):
        fill = np.nanmean(F, axis=0)
    F = np.where(np.isfinite(F), F, fill[None])
    F = np.maximum(F, 0.0)
    # a forecast is only kept where the unit has demand data around that day
    alive = np.isfinite(Y).any(axis=2)
    F[:, ~alive] = np.nan

    np.savez_compressed(os.path.join(ROOT, "data", "processed", "global_panel.npz"),
                        Y=Y.astype(np.float32), F=F.astype(np.float32),
                        code=np.array([m["code"] for m in meta]), names=np.array([m["name"] for m in meta]),
                        continent=np.array([m["continent"] for m in meta]),
                        flex=np.array([m["flex"] for m in meta], dtype=float),
                        has_official=np.isfinite(OF).any(axis=(1, 2)),
                        days=days.strftime("%Y-%m-%d").values, methods=np.array(METHODS),
                        first_day=FIRST_DAY)
    ok = np.isfinite(Y[:, FIRST_DAY:])
    for m, name in enumerate(METHODS):
        with np.errstate(all="ignore"):
            e = np.abs(F[m][:, FIRST_DAY:] - Y[:, FIRST_DAY:]) / np.nanmean(Y, axis=(1, 2))[:, None, None]
        print(f"{name:10s} mean scaled abs error {np.nanmean(np.where(ok, e, np.nan)):.4f}")
    cov = pd.DataFrame(dict(code=[m["code"] for m in meta], continent=[m["continent"] for m in meta],
                            share=np.isfinite(Y).mean(axis=(1, 2)),
                            first=[days[np.argmax(a)].date() if a.any() else None for a in alive],
                            last=[days[len(a) - 1 - np.argmax(a[::-1])].date() if a.any() else None for a in alive]))
    cov.to_csv(os.path.join(ROOT, "results", "global_coverage.csv"), index=False)
    print(cov.groupby("continent").agg(n=("code", "size"), share=("share", "mean")))


if __name__ == "__main__":
    main()
