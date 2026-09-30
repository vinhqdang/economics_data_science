"""Zero-shot foundation-model candidate: Chronos-Bolt (Ansari et al., 2024).

For every grid area and operating day d, the pretrained model receives the
28 days of hourly demand known at decision time (672 values, up to the end of
day d-2, see timing.py) and forecasts 48 hours ahead; the last 24 hours are the
forecast for day d.  No fine-tuning is done,
so the forecast uses no information from the evaluation period.
Output: data/processed/global_chronos.npy with shape (N, days, 24).
"""
import os
import time
import numpy as np
import sys
import torch
from chronos import BaseChronosPipeline

sys.path.insert(0, os.path.dirname(__file__))
from timing import INFO_LAG  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONTEXT_DAYS = 28
BATCH = 512
MODEL = os.environ.get("CHRONOS_MODEL", "amazon/chronos-bolt-small")


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    d = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    Y = d["Y"].astype(np.float32)
    first = int(d["first_day"])
    N, Dn, H = Y.shape
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu", torch_dtype=torch.float32)
    out = np.full((N, Dn, H), np.nan, dtype=np.float32)
    jobs = []
    for i in range(N):
        for day in range(max(first, CONTEXT_DAYS + INFO_LAG), Dn):
            ctx = Y[i, day - INFO_LAG + 1 - CONTEXT_DAYS:day - INFO_LAG + 1].ravel()
            # need most of the context and demand data on the target day
            if np.isfinite(ctx).mean() >= 0.75 and np.isfinite(Y[i, day]).any():
                jobs.append((i, day))
    print("forecasts to produce:", len(jobs), flush=True)
    t0 = time.time()
    for b in range(0, len(jobs), BATCH):
        chunk = jobs[b:b + BATCH]
        ctx = np.stack([Y[i, day - INFO_LAG + 1 - CONTEXT_DAYS:day - INFO_LAG + 1].ravel() for i, day in chunk])
        # Chronos handles missing values given as NaN
        _, mean = pipe.predict_quantiles(torch.tensor(ctx), prediction_length=H * INFO_LAG, quantile_levels=[0.5])
        med = np.asarray(mean).reshape(len(chunk), -1)[:, H * (INFO_LAG - 1):H * INFO_LAG]
        for (i, day), row in zip(chunk, med):
            out[i, day] = row
        if (b // BATCH) % 50 == 0:
            done = b + len(chunk)
            print(f"{done}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(ROOT, "data", "processed", "global_chronos.npy"), np.maximum(out, 0.0))
    ok = np.isfinite(out) & np.isfinite(Y)
    with np.errstate(all="ignore"):
        e = np.abs(out - Y) / np.nanmean(Y, axis=(1, 2))[:, None, None]
    print("Chronos mean scaled abs error", float(np.nanmean(np.where(ok, e, np.nan))))


if __name__ == "__main__":
    main()
