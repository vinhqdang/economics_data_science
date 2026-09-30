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

Features (all computed from the 28 days known at d0, i.e. ending at
d0 - INFO_LAG): for each candidate, its mean scaled newsvendor cost over the
last 7 and 28 days and its mean absolute relative forecast error over 28 days;
the unit's critical ratio, log trailing mean demand, the share of those days
with data, continent, whether an official forecast exists, and the season.
Training uses only weeks whose outcomes are fully observed at d0; the last
four of them are held out for early stopping, and the weights of two
independently initialised networks are averaged.
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
from timing import INFO_LAG, trailing_mean  # noqa: E402
from backtest_global import load_global, MAX_WINDOW  # noqa: E402

CONTS = ["North America", "Europe", "Oceania", "Asia", "Africa"]
HIST_ORIGINS = 52
RETRAIN_EVERY = 4
VAL_WEEKS = 4
SEEDS = (0, 1)


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
            c = newsvendor_cost(Y, P[m], u[:, None, None], o[:, None, None]) / scale[:, :, None]
            cday[m] = np.nanmean(c, axis=2)
            rday[m] = np.nanmean(np.abs(Y - F[m]), axis=2) / scale
    return cday, rday


def features(cday, rday, tau, scale, cont, has_off, d0, doy):
    """z (N, k) from the 28 days known at origin d0 (ending at d0 - INFO_LAG)."""
    N = cday.shape[1]
    e = d0 - INFO_LAG + 1
    with np.errstate(all="ignore"):
        c7 = np.nanmean(cday[:, :, e - 7:e], axis=2).T
        c28 = np.nanmean(cday[:, :, e - 28:e], axis=2).T
        rel = np.nanmean(rday[:, :, e - 28:e], axis=2).T
        have = np.isfinite(cday[0, :, e - 28:e]).mean(axis=1)
        lsc = np.log(np.nan_to_num(scale[:, e - 1], nan=np.nanmedian(scale[:, e - 1])) + 1e-8)
    z = np.column_stack([np.log1p(np.nan_to_num(c7, nan=1.0)), np.log1p(np.nan_to_num(c28, nan=1.0)),
                         np.nan_to_num(rel, nan=1.0), tau, lsc, have,
                         (cont[:, None] == np.array(CONTS)[None, :]).astype(float),
                         has_off.astype(float),
                         np.full(N, np.sin(2 * np.pi * doy[d0] / 365.25)),
                         np.full(N, np.cos(2 * np.pi * doy[d0] / 365.25))])
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--tau", default="calibrated")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weather", default="none", choices=["none", "exact", "noise2", "gfs", "plus"])
    args = ap.parse_args()
    tag = f"global_w{args.window}_tau{args.tau}" + ("" if args.weather == "none" else f"_weather-{args.weather}")
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))

    d, Y, F, P, u, o, tau, cand, first = load_global(args.tau, weather=args.weather)
    M, N, Dn, H = P.shape
    with np.errstate(all="ignore"):
        scale = trailing_mean(Y, 28)           # (N, days), causal scale of day t
    cont = d["continent"]
    has_off = d["has_official"]
    doy = np.array([int(x[5:7]) * 30.4 + int(x[8:10]) for x in d["days"]])
    start = first + RESID_DAYS + MAX_WINDOW + INFO_LAG
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
    gates = []
    for sd_ in SEEDS:
        torch.manual_seed(sd_)
        gates.append(Gate(zdim, M))
    mu, sdv = None, None
    orders = np.full((N, Dn, H), np.nan, dtype=np.float32)
    ut, ot = torch.tensor(u, dtype=torch.float32), torch.tensor(o, dtype=torch.float32)
    tau_t = (ut / (ut + ot))[None, :, None]

    def loss_fn(g, Zt, Qt, yt, okt, st):
        w = g(Zt)                                           # (J, N, M)
        q = (Qt * w[:, :, None, :]).sum(-1)
        r = yt - q
        eps = 0.02 * st[:, :, None]
        smooth = tau_t * r + eps * torch.nn.functional.softplus(-r / eps)
        return ((ut + ot)[None, :, None] * smooth * okt).sum() / (okt.sum() * st.mean() + 1e-9)

    t0 = time.time()
    n_steps = []
    for k, d0 in enumerate(origins):
        # weeks whose outcomes are fully observed at d0: origin j covers days
        # origins[j] .. origins[j] + 6, which must end by d0 - INFO_LAG
        last = [j for j in range(k) if origins[j] + STEP - 1 <= d0 - INFO_LAG]
        past = last[-HIST_ORIGINS:]
        if len(past) >= 2 and (k % RETRAIN_EVERY == 0 or mu is None):
            zs = np.stack([z_at(j) for j in past])                  # (J, N, k)
            mu, sdv = zs.reshape(-1, zdim).mean(0), zs.reshape(-1, zdim).std(0) + 1e-6
            Qs, ys, oks = zip(*[week(j) for j in past])
            Zt = torch.tensor((zs - mu) / sdv, dtype=torch.float32)
            Qt = torch.tensor(np.stack(Qs), dtype=torch.float32)    # (J, N, T, M)
            yt = torch.tensor(np.stack(ys), dtype=torch.float32)
            okt = torch.tensor(np.stack(oks), dtype=torch.float32)
            st = torch.tensor(np.nan_to_num(np.stack([scale[:, origins[j]] for j in past]), nan=1.0),
                              dtype=torch.float32)                  # (J, N)
            nv = min(VAL_WEEKS, len(past) // 4)
            tr, va = slice(0, len(past) - nv), slice(len(past) - nv, len(past))
            for g in gates:
                opt = torch.optim.Adam(g.parameters(), lr=args.lr, weight_decay=1e-4)
                best, best_state, best_ep = np.inf, None, 0
                for ep in range(args.epochs):
                    loss = loss_fn(g, Zt[tr], Qt[tr], yt[tr], okt[tr], st[tr])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if nv > 0 and ep % 10 == 9:
                        with torch.no_grad():
                            v = float(loss_fn(g, Zt[va], Qt[va], yt[va], okt[va], st[va]))
                        if v < best:
                            best, best_ep = v, ep
                            best_state = {kk: vv.clone() for kk, vv in g.state_dict().items()}
                if best_state is not None:
                    g.load_state_dict(best_state)
                n_steps.append(best_ep + 1)
        out = np.arange(d0, min(d0 + STEP, Dn))
        if mu is None:
            w = np.full((N, M), 1.0 / M)
        else:
            with torch.no_grad():
                zt = torch.tensor((z_at(k) - mu) / sdv, dtype=torch.float32)
                w = np.mean([g(zt).numpy() for g in gates], axis=0)
        orders[:, out] = np.einsum("mnlh,nm->nlh", np.nan_to_num(P[:, :, out, :]), w)
        if k % 40 == 0:
            print(f"[{tag} neural gate] origin {k + 1}/{len(origins)} {time.time() - t0:.0f}s "
                  f"mean weights {np.round(w.mean(0), 2)} median steps {np.median(n_steps) if n_steps else 0}",
                  flush=True)

    ev = np.arange(start, Dn)
    yy = Y[:, ev, :]
    valid = np.isfinite(yy) & np.isfinite(P[:, :, ev, :]).all(axis=0)
    cost = np.where(valid, newsvendor_cost(np.where(valid, yy, 0.0), np.nan_to_num(orders[:, ev, :]),
                                           u[:, None, None], o[:, None, None]), 0.0)
    short = np.where(valid, orders[:, ev, :] < yy, False)
    np.savez_compressed(os.path.join(ROOT, "results", f"neuralgate_{tag}.npz"),
                        daily_cost=cost.sum(axis=2)[None], short_hours=short.sum(axis=2)[None],
                        methods=np.array(["NeuralGate"]), eval_days=ev,
                        n_params=sum(p.numel() for p in gates[0].parameters()))
    np.save(os.path.join(ROOT, "results", f"neuralgate_orders_{tag}.npy"), orders[:, ev, :])
    print(f"[{tag} neural gate] done, parameters per network: {sum(p.numel() for p in gates[0].parameters())}, "
          f"median early-stopping step {np.median(n_steps)}")


if __name__ == "__main__":
    main()
