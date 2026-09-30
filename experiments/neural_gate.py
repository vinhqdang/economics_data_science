"""Deep-learning benchmark: a decision-focused neural mixture-of-experts gate.

For unit i at weekly origin d0 a multilayer perceptron maps features z_i(d0)
to combination weights w_i = softmax(g_theta(z_i)) on the simplex, and the unit
commits q = Q w_i for the following week.  The network is trained end to end
on the smoothed newsvendor cost of those commitments, using only weeks whose
outcomes are known at the time of training (the previous 52 weekly origins),
and is re-trained every 4 weeks with a warm start.  This is the
covariate-dependent weighting of mixture-of-experts model averaging (Gong, He
and Zhang, 2026) and of FFORMA (Montero-Manso et al., 2020), trained on the
decision loss instead of forecast error.

Features (all computed from the 28 days before d0): for each candidate, its
mean scaled newsvendor cost over the last 7 and 28 days and its mean absolute
relative forecast error over 28 days; the unit's critical ratio, log mean
demand, continent, whether an official forecast exists, and the season.
usage: python neural_gate.py --window 28
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from gdma.core import newsvendor_cost  # noqa: E402
from backtest import STEP, RESID_DAYS  # noqa: E402
from backtest_global import load_global, MAX_WINDOW  # noqa: E402

CONTS = ["North America", "Europe", "Oceania", "Asia", "Africa"]
HIST_ORIGINS = 52
RETRAIN_EVERY = 4


class Gate(torch.nn.Module):
    def __init__(self, k, m, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(k, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(hidden, m))

    def forward(self, z):
        return torch.softmax(self.net(z), dim=-1)


def daily_stats(P, F, Y, u, o, scale):
    """Per candidate, unit and day: mean scaled newsvendor cost and mean
    absolute relative forecast error over the 24 hours (M, N, days)."""
    M, N, Dn, H = P.shape
    cday = np.full((M, N, Dn), np.nan)
    rday = np.full((M, N, Dn), np.nan)
    with np.errstate(all="ignore"):
        for m in range(M):
            c = newsvendor_cost(Y, P[m], u[:, None, None], o[:, None, None]) / scale[:, None, None]
            cday[m] = np.nanmean(c, axis=2)
            rday[m] = np.nanmean(np.abs(Y - F[m]), axis=2) / scale[:, None]
    return cday, rday


def features(cday, rday, tau, scale, cont, has_off, d0, doy):
    """z (N, k) from the 28 days before origin d0."""
    N = cday.shape[1]
    with np.errstate(all="ignore"):
        c7 = np.nanmean(cday[:, :, d0 - 7:d0], axis=2).T
        c28 = np.nanmean(cday[:, :, d0 - 28:d0], axis=2).T
        rel = np.nanmean(rday[:, :, d0 - 28:d0], axis=2).T
    z = np.column_stack([np.log1p(np.nan_to_num(c7, nan=1.0)), np.log1p(np.nan_to_num(c28, nan=1.0)),
                         np.nan_to_num(rel, nan=1.0), tau, np.log(scale),
                         (cont[:, None] == np.array(CONTS)[None, :]).astype(float),
                         has_off.astype(float),
                         np.full(N, np.sin(2 * np.pi * doy[d0] / 365.25)),
                         np.full(N, np.cos(2 * np.pi * doy[d0] / 365.25))])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--tau", default="calibrated")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--weather", default="none", choices=["none", "exact", "noise2", "gfs", "plus"])
    args = ap.parse_args()
    tag = f"global_w{args.window}_tau{args.tau}" + ("" if args.weather == "none" else f"_weather-{args.weather}")
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))

    d, Y, F, P, u, o, tau, cand, first = load_global(args.tau, weather=args.weather)
    M, N, Dn, H = P.shape
    with np.errstate(all="ignore"):
        scale = np.nanmean(Y, axis=(1, 2))
    cont = d["continent"]
    has_off = d["has_official"]
    doy = np.array([int(x[5:7]) * 30.4 + int(x[8:10]) for x in d["days"]])
    start = first + RESID_DAYS + MAX_WINDOW
    origins = np.arange(start, Dn, STEP)

    cday, rday = daily_stats(P, F, Y, u, o, scale)
    Z = {}

    def z_at(k):
        if k not in Z:
            Z[k] = features(cday, rday, tau, scale, cont, has_off, origins[k], doy)
        return Z[k]

    def week(k):
        """Commitment candidates and demand of the week after origin k."""
        days = np.arange(origins[k], min(origins[k] + STEP, Dn))
        Q = P[:, :, days, :].reshape(M, N, -1).transpose(1, 2, 0)   # (N, T, M)
        y = Y[:, days, :].reshape(N, -1)
        ok = np.isfinite(y) & np.isfinite(Q).all(axis=2)
        return np.where(ok[..., None], Q, 0.0), np.where(ok, y, 0.0), ok

    zdim = z_at(0).shape[1]
    gate = Gate(zdim, M)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    mu, sd = None, None
    orders = np.full((N, Dn, H), np.nan, dtype=np.float32)
    ut, ot = torch.tensor(u, dtype=torch.float32), torch.tensor(o, dtype=torch.float32)
    st = torch.tensor(scale, dtype=torch.float32)
    t0 = time.time()
    for k, d0 in enumerate(origins):
        # re-train on the previous HIST_ORIGINS weeks whose outcomes are observed
        past = [j for j in range(max(0, k - HIST_ORIGINS), k)]
        if past and (k % RETRAIN_EVERY == 0 or k < RETRAIN_EVERY):
            zs = np.stack([z_at(j) for j in past])                  # (J, N, k)
            mu, sd = zs.reshape(-1, zdim).mean(0), zs.reshape(-1, zdim).std(0) + 1e-6
            Qs, ys, oks = zip(*[week(j) for j in past])
            Zt = torch.tensor((zs - mu) / sd, dtype=torch.float32)
            Qt = torch.tensor(np.stack(Qs), dtype=torch.float32)    # (J, N, T, M)
            yt = torch.tensor(np.stack(ys), dtype=torch.float32)
            okt = torch.tensor(np.stack(oks), dtype=torch.float32)
            eps = 0.02 * st[None, :, None]
            for _ in range(args.epochs if k >= RETRAIN_EVERY else 30):
                w = gate(Zt)                                        # (J, N, M)
                q = (Qt * w[:, :, None, :]).sum(-1)
                r = yt - q
                tau_t = (ut / (ut + ot))[None, :, None]
                smooth = tau_t * r + eps * torch.nn.functional.softplus(-r / eps)
                loss = ((ut + ot)[None, :, None] * smooth * okt).sum() / (okt.sum() * st.mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
        out = np.arange(d0, min(d0 + STEP, Dn))
        if mu is None:
            w = np.full((N, M), 1.0 / M)
        else:
            with torch.no_grad():
                w = gate(torch.tensor((z_at(k) - mu) / sd, dtype=torch.float32)).numpy()
        orders[:, out] = np.einsum("mnlh,nm->nlh", np.nan_to_num(P[:, :, out, :]), w)
        if k % 40 == 0:
            print(f"[{tag} neural gate] origin {k + 1}/{len(origins)} {time.time() - t0:.0f}s "
                  f"mean weights {np.round(w.mean(0), 2)}", flush=True)

    ev = np.arange(start, Dn)
    yy = Y[:, ev, :]
    valid = np.isfinite(yy) & np.isfinite(P[:, :, ev, :]).all(axis=0)
    cost = np.where(valid, newsvendor_cost(np.where(valid, yy, 0.0), np.nan_to_num(orders[:, ev, :]),
                                           u[:, None, None], o[:, None, None]), 0.0)
    short = np.where(valid, orders[:, ev, :] < yy, False)
    np.savez_compressed(os.path.join(ROOT, "results", f"neuralgate_{tag}.npz"),
                        daily_cost=cost.sum(axis=2)[None], short_hours=short.sum(axis=2)[None],
                        methods=np.array(["NeuralGate"]), eval_days=ev,
                        n_params=sum(p.numel() for p in gate.parameters()))
    np.save(os.path.join(ROOT, "results", f"neuralgate_orders_{tag}.npy"), orders[:, ev, :])
    print(f"[{tag} neural gate] done, parameters: {sum(p.numel() for p in gate.parameters())}")


if __name__ == "__main__":
    main()
