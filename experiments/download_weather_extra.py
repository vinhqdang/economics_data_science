"""Additional archived day-ahead weather forecasts for the 169 load centres.

GFS (1 degree, NOAA Open Data on AWS, from January 2021), run of 00 UTC on day
d-1, leads 24..48 h: 2-m relative humidity, 10-m wind components and surface
downward short-wave radiation.
GEFS (GEFSv12, 0.5 degree, from October 2020), same run and leads: ensemble
standard deviation (spread) of 2-m temperature, a direct measure of how
uncertain tomorrow's weather is.
For every file the .idx index is read once and only the required fields are
fetched by byte range.
Output: data/interim/weather_extra.npz with arrays (days, 9 leads, points).
"""
import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd
import pygrib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from download_gfs import curl, LEADS, PTS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
GFS_FIELDS = {"rh": ":RH:2 m above ground:", "u10": ":UGRD:10 m above ground:",
              "v10": ":VGRD:10 m above ground:", "dswrf": ":DSWRF:surface:"}
GEFS_FIELDS = {"t2m_spread": ":TMP:2 m above ground:"}


def fetch_fields(base, matches):
    """Return {name: 2-D field} for the GRIB messages whose index line contains
    the given match strings (first occurrence)."""
    idx = curl([base + ".idx"])
    if idx is None:
        return None
    lines = idx.decode().splitlines()
    out = {}
    for name, key in matches.items():
        for j, ln in enumerate(lines):
            if key in ln:
                start = int(ln.split(":")[1])
                end = int(lines[j + 1].split(":")[1]) - 1 if j + 1 < len(lines) else ""
                for attempt in range(3):
                    data = curl(["-r", f"{start}-{end}", base])
                    if data is None:
                        break
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".grb2") as f:
                            f.write(data)
                            f.flush()
                            out[name] = pygrib.open(f.name)[1].values.astype(np.float32)
                        break
                    except Exception:
                        time.sleep(1 + attempt)
                break
    return out


def bilinear(field, lat, lon, res):
    r = (90.0 - lat) / res
    c = (lon % 360.0) / res
    nr, nc = field.shape
    r0, c0 = int(np.floor(r)), int(np.floor(c)) % nc
    dr, dc = r - np.floor(r), c - np.floor(c)
    r1, c1 = min(r0 + 1, nr - 1), (c0 + 1) % nc
    return ((1 - dr) * (1 - dc) * field[r0, c0] + (1 - dr) * dc * field[r0, c1]
            + dr * (1 - dc) * field[r1, c0] + dr * dc * field[r1, c1])


def run(kind, days, fields, res, url_of):
    ck = os.path.join(ROOT, "data", "interim", f"weather_extra_{kind}_partial.npz")
    if os.path.exists(ck):
        z = np.load(ck)
        arrs = {k: z[k] for k in fields}
    else:
        arrs = {k: np.full((len(days), len(LEADS), len(PTS)), np.nan, dtype=np.float32) for k in fields}
    first = next(iter(fields))
    tasks = [(i, k) for i in range(len(days)) for k in range(len(LEADS)) if np.isnan(arrs[first][i, k]).all()]
    print(kind, "files to fetch:", len(tasks), flush=True)

    def work(t):
        i, k = t
        for base in url_of(days[i] - pd.Timedelta(days=1), LEADS[k]):
            got = fetch_fields(base, fields)
            if got:
                for name, fld in got.items():
                    arrs[name][i, k] = [bilinear(fld, la, lo, res) for la, lo in PTS]
                return True
        return False

    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 16))) as ex:
        for n, res_ in enumerate(ex.map(work, tasks)):
            ok += res_
            if n % 2000 == 0:
                print(f"{kind} {n}/{len(tasks)} files, {ok} ok, {time.time() - t0:.0f}s", flush=True)
                np.savez(ck, **arrs)
    np.savez(ck, **arrs)
    return arrs


def main():
    days = pd.date_range("2020-10-02", "2026-09-26", freq="D")
    gfs = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    gefs = "https://noaa-gefs-pds.s3.amazonaws.com"

    def gfs_urls(run_day, lead):
        ymd = run_day.strftime("%Y%m%d")
        return [f"{gfs}/gfs.{ymd}/00/{sub}gfs.t00z.pgrb2.1p00.f{lead:03d}" for sub in ("atmos/", "")]

    def gefs_urls(run_day, lead):
        ymd = run_day.strftime("%Y%m%d")
        return [f"{gefs}/gefs.{ymd}/00/atmos/pgrb2ap5/gespr.t00z.pgrb2a.0p50.f{lead:03d}"]

    a = run("gefs", days, GEFS_FIELDS, 0.5, gefs_urls)
    b = run("gfs", days, GFS_FIELDS, 1.0, gfs_urls)
    np.savez_compressed(os.path.join(ROOT, "data", "interim", "weather_extra.npz"),
                        days=days.strftime("%Y-%m-%d").values, leads=np.array(LEADS),
                        points=np.array(PTS), **a, **b)
    for k, v in {**a, **b}.items():
        print(k, "retrieved share", float(np.isfinite(v).any(axis=2).mean()), "mean", float(np.nanmean(v)))


if __name__ == "__main__":
    main()
