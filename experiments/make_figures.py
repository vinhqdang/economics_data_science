"""Maps and process figures for the paper.

fig_pipeline.pdf   data -> candidates -> GDMA -> commitments -> storage siting, and the day-ahead timeline
fig_map_gain.pdf   cost of soft GDMA relative to the reference forecaster in each of the 125 areas
fig_map_storage.pdf  value of the first MW of battery in each area under soft GDMA operations
fig_elliott.pdf    Winter Storm Elliott: daily cost of the U.S. balancing authorities and temperature
Natural Earth country polygons (public domain) are read from data/geo.
"""
import json
import os
import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "paper", "figures")
GEO = os.path.join(ROOT, "data", "geo", "ne_110m_admin_0_countries.geojson")
URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
TAG = "global_w28_taucalibrated_weather-plus"
CONT_COL = {"North America": "#1f77b4", "Europe": "#2ca02c", "Oceania": "#9467bd", "Asia": "#d62728",
            "Africa": "#ff7f0e"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs(FIG, exist_ok=True)


def centroids():
    src = open(os.path.join(ROOT, "experiments", "download_weather.py")).read()
    block = src[src.index("POINTS = {"):src.index("\n}\n", src.index("POINTS = {"))]
    out = {}
    for code, pts in re.findall(r'"([A-Z]{2}(?:-[A-Za-z0-9]+)?)": \[((?:\([^)]*\),? ?)+)\]', block):
        xy = np.array(re.findall(r"\(([-\d.]+), ([-\d.]+)\)", pts), float)
        out[code] = (xy[:, 0].mean(), xy[:, 1].mean())          # lat, lon
    return out


def polygons():
    if not os.path.exists(GEO):
        os.makedirs(os.path.dirname(GEO), exist_ok=True)
        urllib.request.urlretrieve(URL, GEO)
    rings = []
    for f in json.load(open(GEO))["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            rings.append(np.array(p[0]))
    return rings


def basemap(ax, extent):
    for r in polygons():
        ax.add_patch(Polygon(r, closed=True, fc="#eeeeee", ec="#bdbdbd", lw=0.3, zorder=0))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(extent[2:]))))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def area_table():
    r = np.load(os.path.join(ROOT, "results", f"backtest_{TAG}.npz"), allow_pickle=True)
    meth = [str(m) for m in r["methods"]]
    C = r["daily_cost"]
    ok = np.isfinite(C[meth.index("Official")]) & np.isfinite(C[meth.index("GDMA-soft")])
    ratio = (np.where(ok, C[meth.index("GDMA-soft")], 0).sum(1) / np.where(ok, C[meth.index("Official")], 0).sum(1))
    pn = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    code = [str(c) for c in r["code"]]
    mean_mw = np.nanmean(pn["Y"].astype(float), axis=(1, 2))
    xy = centroids()
    df = pd.DataFrame({"code": code, "continent": r["continent"], "ratio": ratio, "mw": mean_mw})
    df["lat"] = [xy.get(c, (np.nan, np.nan))[0] for c in code]
    df["lon"] = [xy.get(c, (np.nan, np.nan))[1] for c in code]
    miss = df[df.lat.isna()].code.tolist()
    assert not miss, miss
    return df, r, meth


def scatter_map(ax, df, col, norm, cmap, size_col="mw", smin=12, smax=140):
    s = smin + (smax - smin) * np.sqrt(df[size_col] / df[size_col].max())
    return ax.scatter(df.lon, df.lat, c=df[col], s=s, cmap=cmap, norm=norm, ec="k", lw=0.25, zorder=3)


def map_gain(df):
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.04, wspace=0.04)
    norm = TwoSlopeNorm(vmin=0.55, vcenter=1.0, vmax=1.25)
    ax = fig.add_subplot(gs[0, :])
    basemap(ax, (-170, 180, -50, 72))
    sc = scatter_map(ax, df, "ratio", norm, "RdYlGn_r")
    ax.set_title("(a) World", loc="left", fontsize=8)
    for k, (title, ext, col) in enumerate([("(b) United States", (-125, -66, 24, 50), 0),
                                           ("(c) Europe", (-11, 33, 35, 68), 1),
                                           ("(d) East Asia", (97, 128, 18, 48), 2)]):
        a = fig.add_subplot(gs[1, col])
        basemap(a, ext)
        scatter_map(a, df, "ratio", norm, "RdYlGn_r", smin=10, smax=90)
        a.set_title(title, loc="left", fontsize=8)
    cb = fig.colorbar(sc, ax=fig.axes, orientation="horizontal", fraction=0.035, pad=0.03, aspect=45)
    cb.set_label("Cost of soft GDMA / cost of the reference forecaster (lower is better; green = soft GDMA cheaper)")
    fig.text(0.01, 0.005, "Marker area proportional to the square root of mean demand. Marker position: mean of the load centres "
             "used for the area's weather.", fontsize=6.5, color="#555555")
    fig.savefig(os.path.join(FIG, "fig_map_gain.pdf"), bbox_inches="tight")
    plt.close(fig)


def map_storage(df):
    sp = pd.read_csv(os.path.join(ROOT, "results", "storage_per_area_weather-plus.csv")).set_index("code")
    df = df.copy()
    df["mv"] = sp.loc[df.code, "mv1_soft"].values / 1e3
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.28, wspace=0.28)
    norm = plt.Normalize(vmin=0, vmax=df.mv.quantile(0.98))
    a0 = fig.add_subplot(gs[0, :])
    basemap(a0, (-170, 180, -50, 72))
    sc = scatter_map(a0, df, "mv", norm, "viridis", smin=14, smax=110)
    a0.set_title("(a) Value of the first MW of battery under soft GDMA operations", loc="left", fontsize=8)
    cb = fig.colorbar(sc, ax=a0, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label("Thousand USD per MW-year (higher is better)")
    ax = fig.add_subplot(gs[1, 0])
    for c, g in df.groupby("continent"):
        ax.scatter(g.mw / 1e3, g.mv, s=14, color=CONT_COL[c], label=c, alpha=0.85, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Mean demand (GW, log scale)")
    ax.set_ylabel("Value of the first MW\n(thousand USD per year)")
    ax.set_title("(b) Value and area size", loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist([df.mv[df.continent == c] for c in CONT_COL if (df.continent == c).any()], bins=12, stacked=True,
             color=[CONT_COL[c] for c in CONT_COL if (df.continent == c).any()])
    ax2.set_xlabel("Value of the first MW (thousand USD per year)")
    ax2.set_ylabel("Number of areas")
    ax2.set_title("(c) Distribution", loc="left", fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_map_storage.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_elliott(r, meth):
    cal = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)["days"]
    days = pd.to_datetime(cal[r["eval_days"]])
    us = np.asarray(r["continent"]) == "North America"
    C = r["daily_cost"][:, us]
    ok = np.isfinite(C).all(axis=0)
    win = (days >= "2022-12-14") & (days <= "2022-12-31")
    d = days[win]
    tot = lambda m: np.where(ok, np.nan_to_num(C[meth.index(m)]), 0)[:, win].sum(0)
    off, soft = tot("Official"), tot("GDMA-soft")
    W = pd.read_parquet(os.path.join(ROOT, "data", "interim", "weather_t2m.parquet")).resample("D").mean()
    W.index = W.index.tz_localize(None) if W.index.tz is not None else W.index
    codes = [str(c) for c in r["code"]][:]
    cols = [c for c, u in zip(codes, us) if u and c in W.columns]
    temp = W.reindex(d)[cols].mean(axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.08))
    ax[0].axvspan(pd.Timestamp("2022-12-21"), pd.Timestamp("2022-12-26 23:59"), color="#fdd0a2", alpha=0.6, lw=0)
    ax[0].plot(d, off / off.mean(), color="k", lw=1.4, label="Operators' official forecasts")
    ax[0].plot(d, soft / off.mean(), color="#d62728", lw=1.4, label="Soft GDMA")
    ax[0].set_ylabel("Daily cost, 46 U.S. BAs\n(relative to official mean;\nlower is better)")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax[0].text(pd.Timestamp("2022-12-26 12:00"), 2.8, "Elliott window\n21-26 Dec", ha="left", va="top", fontsize=7)
    ax[1].axvspan(pd.Timestamp("2022-12-21"), pd.Timestamp("2022-12-26 23:59"), color="#fdd0a2", alpha=0.6, lw=0)
    ax[1].plot(d, temp, color="#1f77b4", lw=1.4)
    ax[1].set_ylabel("Mean daily temperature,\nU.S. load centres (deg C)")
    import matplotlib.dates as mdates
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d Dec'))
    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    fig.savefig(os.path.join(FIG, "fig_elliott.pdf"), bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, fs=7.2, ec="#444444", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#444444"))


def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    # row 1: data
    ax.text(0.05, 6.95, "A. From public data to storage siting", fontsize=8.5, fontweight="bold", va="center")
    box(ax, 0.1, 5.55, 2.9, 1.1, "Hourly demand, 125 grid areas\nEIA-930, Energy-Charts, AEMO,\nZenodo (China, Taiwan, Thailand),\nMendeley (Algeria)", "#deebf7")
    box(ax, 3.55, 5.55, 2.9, 1.1, "Weather\nNASA POWER (temperature)\nGFS and GEFS day-ahead forecasts\n(humidity, wind, radiation, spread)", "#deebf7")
    box(ax, 7.0, 5.55, 2.9, 1.1, "Context\nIncome (World Bank, state\nper-capita income for the U.S.)\nCost asymmetry of each area", "#deebf7")
    # row 2: candidates
    box(ax, 0.1, 3.95, 9.8, 1.0, "11 candidate demand forecasts: operator's official, persistence, weekly, 7-day mean, 3-week profile,\nhourly regression, LightGBM, Chronos foundation model, three weather-driven models", "#e5f5e0")
    for x in (1.55, 5.0, 8.45):
        arrow(ax, x, 5.55, x, 4.95)
    # row 3: GDMA
    box(ax, 0.1, 2.2, 4.7, 1.35, "GDMA: learn jointly\n- latent segments of grid areas\n- combination weights on the simplex\nby minimising each area's own\ncommitment cost (rolling, weekly)", "#fff7bc", bold=False)
    box(ax, 5.2, 2.2, 4.7, 1.35, "Soft GDMA and fair GDMA\n- shrink each area to its segment\n- number of segments by 1-SE rule\n- fair version counts every area's\n  relative gain equally", "#fff7bc")
    arrow(ax, 4.81, 2.9, 5.19, 2.9)
    arrow(ax, 2.45, 3.95, 2.45, 3.56)
    # row 4: outputs
    box(ax, 0.1, 0.5, 2.9, 1.2, "Day-ahead commitments\nout of sample, 2018-2026\nDM tests, MCS, fairness audit", "#fde0dd")
    box(ax, 3.55, 0.5, 2.9, 1.2, "Residual shortfall and\nsurplus profile of each area", "#fde0dd")
    box(ax, 7.0, 0.5, 2.9, 1.2, "Storage siting: efficient,\nfloors, maximin;\nout-of-sample check", "#fde0dd")
    arrow(ax, 1.55, 2.2, 1.55, 1.71)
    arrow(ax, 3.0, 1.1, 3.54, 1.1)
    arrow(ax, 6.46, 1.1, 6.99, 1.1)
    fig.savefig(os.path.join(FIG, "fig_pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)

    # timeline of the information set
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.text(0.05, 2.8, "B. Information set for the commitment of UTC day d", fontsize=8.5, fontweight="bold", va="center")
    xs = {"d-2": (0.3, 3.2), "d-1": (3.4, 6.3), "d": (6.5, 9.4)}
    for lab, (a, b) in xs.items():
        ax.add_patch(FancyBboxPatch((a, 0.75), b - a, 0.75, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc={"d-2": "#c7e9c0", "d-1": "#fff7bc", "d": "#fcbba1"}[lab], ec="#444444", lw=0.8))
        ax.text((a + b) / 2, 1.125, f"Day {lab}", ha="center", va="center", fontsize=8)
    for x, t1, t2 in [(1.75, "Latest demand data used", "(everything up to the end of d-2)"),
                      (4.85, "Morning: gate closes,", "commitment fixed; 00 UTC NWP run usable"),
                      (7.95, "Demand realised,", "cost scored")]:
        ax.text(x, 0.5, t1, ha="center", fontsize=6.8)
        ax.text(x, 0.2, t2, ha="center", fontsize=6.3, color="#444444")
    arrow(ax, 3.3, 1.9, 3.3, 1.55)
    ax.text(3.3, 2.1, "gate closure", ha="center", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_timeline.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    df, r, meth = area_table()
    print(df.groupby("continent").ratio.describe().round(3))
    map_gain(df)
    map_storage(df)
    fig_elliott(r, meth)
