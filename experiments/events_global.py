"""Cost of every combination rule during Winter Storm Elliott (22-27 December
2022) on the U.S. balancing authorities of the global panel, relative to the
official day-ahead forecasts, for each candidate set."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVENT = ("2022-12-22", "2022-12-27")


def event_ratios(suffix):
    p = os.path.join(ROOT, "results", f"backtest_global_w28_taucalibrated{suffix}.npz")
    if not os.path.exists(p):
        return None
    r = np.load(p, allow_pickle=True)
    cal = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)["days"]
    ev = pd.to_datetime(cal[r["eval_days"]])
    days = (ev >= EVENT[0]) & (ev <= EVENT[1])
    us = r["continent"] == "North America"
    c = r["daily_cost"][:, us][:, :, days]
    ok = np.isfinite(c).all(axis=0)
    tot = np.where(ok, c, 0.0).sum(axis=(1, 2))
    m = list(r["methods"])
    return pd.Series(tot / tot[m.index("Official")], index=m)


def main():
    specs = sys.argv[1:] or ["", "_weather-gfs", "_weather-plus", "_weather-exact"]
    res = {s or "none": event_ratios(s) for s in specs}
    out = pd.DataFrame({k: v for k, v in res.items() if v is not None})
    print(out.round(3).to_string())
    out.to_csv(os.path.join(ROOT, "results", "elliott_global.csv"))


if __name__ == "__main__":
    main()
