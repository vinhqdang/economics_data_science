"""Archived day-ahead temperature forecasts from NOAA's Global Forecast System.

For every operating day d (UTC) from 2 January 2021 we take the GFS run
initialised at 00 UTC on day d-1 and its 2-m temperature forecasts for lead
times 24, 27, ..., 48 hours, which cover day d.  Such forecasts are available
to an operator on the morning of day d-1, before day-ahead commitments are
made.  Only the 2-m temperature field is downloaded from each 1-degree GRIB2
file (byte-range request located through the .idx index) from the NOAA Open
Data Dissemination archive on AWS (noaa-gfs-bdp-pds), which starts in January
2021.  Values are bilinearly interpolated to the 169 load centres of
download_weather.py.
Output: data/interim/gfs_t2m.npz with forecast temperature (days, 9 leads, 169 points).
"""
import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import pandas as pd
import pygrib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from download_weather import POINTS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
LEADS = list(range(24, 49, 3))
PTS = sorted({p for pts in POINTS.values() for p in pts})


def curl(args):
    for k in range(4):
        r = subprocess.run(["curl", "-sS", "--max-time", "60", "-w", "%{http_code}"] + args,
                           capture_output=True)
        code = r.stdout[-3:].decode(errors="ignore")
        if code in ("200", "206"):
            return r.stdout[:-3]
        if code == "404":
            return None
        time.sleep(2 ** k)
    return None


def fetch(run_day, lead):
    """2-m temperature field (K) of the 00 UTC run of run_day at `lead` hours."""
    ymd = run_day.strftime("%Y%m%d")
    for sub in ("atmos/", ""):
        base = f"{BUCKET}/gfs.{ymd}/00/{sub}gfs.t00z.pgrb2.1p00.f{lead:03d}"
        idx = curl([base + ".idx"])
        if idx is None:
            continue
        lines = idx.decode().splitlines()
        for j, ln in enumerate(lines):
            if ":TMP:2 m above ground:" in ln:
                start = int(ln.split(":")[1])
                end = int(lines[j + 1].split(":")[1]) - 1 if j + 1 < len(lines) else ""
                for attempt in range(3):          # re-fetch truncated transfers
                    data = curl(["-r", f"{start}-{end}", base])
                    if data is None:
                        return None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".grb2") as f:
                            f.write(data)
                            f.flush()
                            return pygrib.open(f.name)[1].values.astype(np.float32)
                    except Exception:
                        time.sleep(1 + attempt)
                return None
    return None


def bilinear(field, lat, lon):
    """field on the 1-degree grid: rows 90..-90, columns 0..359."""
    r = 90.0 - lat
    c = lon % 360.0
    r0, c0 = int(np.floor(r)), int(np.floor(c)) % 360
    dr, dc = r - np.floor(r), c - np.floor(c)
    r1, c1 = min(r0 + 1, 180), (c0 + 1) % 360
    return ((1 - dr) * (1 - dc) * field[r0, c0] + (1 - dr) * dc * field[r0, c1]
            + dr * (1 - dc) * field[r1, c0] + dr * dc * field[r1, c1])


def main():
    days = pd.date_range("2021-01-02", "2026-09-26", freq="D")
    ck = os.path.join(ROOT, "data", "interim", "gfs_t2m_partial.npy")
    out = np.load(ck) if os.path.exists(ck) else np.full((len(days), len(LEADS), len(PTS)), np.nan, dtype=np.float32)
    tasks = [(i, k) for i in range(len(days)) for k in range(len(LEADS)) if np.isnan(out[i, k]).all()]
    print("fields to fetch:", len(tasks), flush=True)

    def work(t):
        i, k = t
        fld = fetch(days[i] - pd.Timedelta(days=1), LEADS[k])
        if fld is not None:
            out[i, k] = [bilinear(fld, la, lo) - 273.15 for la, lo in PTS]
        return fld is not None

    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 16))) as ex:
        for n, res in enumerate(ex.map(work, tasks)):
            ok += res
            if n % 2000 == 0:
                print(f"{n}/{len(tasks)} fields, {ok} ok, {time.time() - t0:.0f}s", flush=True)
                np.save(ck, out)
    np.savez_compressed(os.path.join(ROOT, "data", "interim", "gfs_t2m.npz"), t2m=out,
                        days=days.strftime("%Y-%m-%d").values, leads=np.array(LEADS),
                        points=np.array(PTS))
    print("done; share of fields retrieved", ok / len(tasks))


if __name__ == "__main__":
    main()
