"""Rolling out-of-sample evaluation of day-ahead capacity-commitment policies.

usage: python backtest.py --window 28 [--tau calibrated|0.9] [--tag name]

Every 7 days each method re-estimates its combination weights from the most
recent `window` days and commits hourly capacity for the next 7 days.  Only
information available before an operating day is used for that day.
"""
import argparse
import os
import sys
import time
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from gdma.core import (newsvendor_cost, fit_weights, fit_per_unit, select_grouped,  # noqa: E402
                       fit_kmeans_two_step, cost_matrix, fit_grouped,
                       select_soft_grouped)

RESID_DAYS = 28      # window for the residual quantile of each candidate policy
STEP = 7             # re-estimation frequency (days)
MAX_WINDOW = 56      # common start of the evaluation sample for every window
G_MAX = 6


def calibrate_costs(flex, tau_spec):
    """Unit overage cost o = 1; underage cost from the critical ratio tau_i.

    'calibrated': tau_i falls linearly from 0.95 for the least import-flexible
    BA to 0.80 for the most flexible one (ranked interchange flexibility).
    """
    N = len(flex)
    if tau_spec == "calibrated":
        r = (np.argsort(np.argsort(flex)) / (N - 1))
        tau = 0.95 - 0.15 * r
    else:
        tau = np.full(N, float(tau_spec))
    o = np.ones(N)
    u = tau / (1 - tau)
    return u, o, tau


REL_FLOOR = 0.2     # floor of the relative-error denominator, share of mean demand


def candidate_policies(Y, F, tau, first):
    """q^{(m)}_{i,d,h} = f^{(m)}_{i,d,h} (1 + Qhat_tau_i(relative residuals of m
    over days d-28..d-1)).  Relative errors are taken with respect to
    max(f, REL_FLOOR * mean demand of the BA), so that an implausibly small
    forecast cannot produce an unbounded margin."""
    M, N, Dn, H = F.shape
    P = np.full(F.shape, np.nan)
    floor = REL_FLOOR * np.nanmean(Y[:, first:], axis=(1, 2))
    rel = (Y[None] - F) / np.maximum(F, floor[None, :, None, None])
    for d in range(first + RESID_DAYS, Dn):
        win = rel[:, :, d - RESID_DAYS:d, :].reshape(M, N, -1)
        for i in range(N):
            qi = np.nanquantile(win[:, i], tau[i], axis=1)          # (M,)
            P[:, i, d, :] = F[:, i, d, :] * (1.0 + qi[:, None])
    return np.maximum(P, 0.0)


def window_arrays(A, Y, days):
    """Stack candidate arrays (M,N,Dn,H) over `days` into (N, T, M) and (N, T);
    hours with missing demand are zeroed so that they carry no loss."""
    Q = A[:, :, days, :].reshape(A.shape[0], A.shape[1], -1).transpose(1, 2, 0)
    y = Y[:, days, :].reshape(Y.shape[0], -1)
    miss = ~np.isfinite(y) | ~np.isfinite(Q).all(axis=2)
    Q = np.where(miss[..., None], 0.0, Q)
    y = np.where(miss, 0.0, y)
    return Q, y


def fto_orders(F, Y, W, labels, tau, days_win, days_out):
    """Forecast-then-order: combine point forecasts with weights W[labels], then
    add the tau-quantile of the combined forecast's relative residuals over the
    last RESID_DAYS days of the window."""
    N = Y.shape[0]
    rd = days_win[-RESID_DAYS:]
    out = np.empty((N, len(days_out), Y.shape[2]))
    for i in range(N):
        w = W[labels[i]]
        fc = np.tensordot(w, F[:, i, rd, :], axes=1)
        floor = REL_FLOOR * np.nanmean(Y[i])
        rel = ((Y[i, rd, :] - fc) / np.maximum(fc, floor)).ravel()
        qi = np.nanquantile(rel, tau[i])
        out[i] = np.tensordot(w, F[:, i, days_out, :], axes=1) * (1 + qi)
    return np.maximum(out, 0.0)


METHOD_NAMES = ["Official", "EW", "Select", "Pooled", "PerBA", "Shrink", "KMeans2S",
                "FTO-PerBA", "FTO-Grouped", "GDMA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--tau", default="calibrated")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_init", type=int, default=4)
    ap.add_argument("--variants", action="store_true",
                    help="only run GDMA variants (G chosen by the hold-out minimum)")
    args = ap.parse_args()
    tag = args.tag or f"w{args.window}_tau{args.tau}"
    if args.variants:
        return run_variants(args, tag)

    d = np.load(os.path.join(ROOT, "data", "processed", "panel.npz"), allow_pickle=True)
    Y, F = d["Y"].astype(float), d["F"].astype(float)
    first = int(d["first_day"])
    M, N, Dn, H = F.shape
    u, o, tau = calibrate_costs(d["flex"], args.tau)
    P = candidate_policies(Y, F, tau, first)
    scale = np.nanmean(Y[:, first:], axis=(1, 2))

    start = first + RESID_DAYS + MAX_WINDOW
    origins = np.arange(start, Dn, STEP)
    K = len(METHOD_NAMES)
    orders = np.full((K, N, Dn, H), np.nan)
    hist = dict(G=[], labels=[], W=[], G_fto=[], kappa=[], G_km=[])
    prev_labels = None
    t0 = time.time()
    for n, d0 in enumerate(origins):
        win = np.arange(d0 - args.window, d0)
        out = np.arange(d0, min(d0 + STEP, Dn))
        Q, y = window_arrays(P, Y, win)
        Qo = P[:, :, out, :]                                   # (M, N, len, H)
        comb = lambda Wn: np.einsum("mnlh,nm->nlh", Qo, Wn)

        orders[0][:, out] = Qo[0]
        orders[1][:, out] = Qo.mean(axis=0)
        C_single = newsvendor_cost(y[..., None], Q, u[:, None, None], o[:, None, None]).sum(1)
        orders[2][:, out] = Qo[C_single.argmin(1), np.arange(N)]
        pooled = fit_grouped(Q, y, 1, u, o, scale=scale)["W"][0]
        orders[3][:, out] = comb(np.tile(pooled, (N, 1)))
        W_unit = fit_per_unit(Q, y, u, o, scale=scale)
        orders[4][:, out] = comb(W_unit)

        # shrinkage of unit weights towards the pooled weights; kappa by hold-out
        T1 = int(np.ceil(2 / 3 * Q.shape[1]))
        Wa = fit_per_unit(Q[:, :T1], y[:, :T1], u, o, scale=scale)
        pa = fit_grouped(Q[:, :T1], y[:, :T1], 1, u, o, scale=scale)["W"][0]
        kappas = np.linspace(0, 1, 5)
        sc = [np.diag(cost_matrix(Q[:, T1:], y[:, T1:], (1 - k) * Wa + k * pa, u, o)).sum()
              for k in kappas]
        kap = kappas[int(np.argmin(sc))]
        orders[5][:, out] = comb((1 - kap) * W_unit + kap * pooled)

        # two-step: k-means on unit weights, G by the same hold-out rule
        sc = []
        for G in range(1, G_MAX + 1):
            f = fit_kmeans_two_step(Q[:, :T1], y[:, :T1], G, u, o, scale=scale, W_unit=Wa)
            sc.append(cost_matrix(Q[:, T1:], y[:, T1:], f["W"], u, o)[np.arange(N), f["labels"]].sum())
        Gk = int(np.argmin(sc)) + 1
        km = fit_kmeans_two_step(Q, y, Gk, u, o, scale=scale, W_unit=W_unit)
        orders[6][:, out] = comb(km["W"][km["labels"]])

        # forecast-then-order: squared-error weights on point forecasts
        Fq, fy = window_arrays(F, Y, win)
        Wf = fit_per_unit(Fq, fy, loss="squared", scale=scale)
        orders[7][:, out] = fto_orders(F, Y, Wf, np.arange(N), tau, win, out)
        gf = select_grouped(Fq, fy, G_MAX, loss="squared", scale=scale, n_init=args.n_init)
        orders[8][:, out] = fto_orders(F, Y, gf["W"], gf["labels"], tau, win, out)

        # proposed: grouped decision-focused model averaging
        g = select_grouped(Q, y, G_MAX, u, o, scale=scale, n_init=args.n_init, init_labels=prev_labels)
        prev_labels = g["labels"]
        orders[9][:, out] = comb(g["W"][g["labels"]])

        hist["G"].append(g["G"]); hist["labels"].append(g["labels"]); hist["W"].append(g["W"][g["labels"]])
        hist["G_fto"].append(gf["G"]); hist["kappa"].append(kap); hist["G_km"].append(Gk)
        if n % 20 == 0:
            print(f"[{tag}] origin {n + 1}/{len(origins)}  G={g['G']}  kappa={kap:.2f}  "
                  f"Gkm={Gk}  Gfto={gf['G']}  {time.time() - t0:.0f}s", flush=True)

    ev = np.arange(start, Dn)
    yy = Y[:, ev, :]
    valid = np.isfinite(yy)
    cost = np.where(valid[None], newsvendor_cost(yy[None], orders[:, :, ev, :],
                                                  u[None, :, None, None], o[None, :, None, None]), 0.0)
    short = np.where(valid[None], orders[:, :, ev, :] < yy[None], False)
    np.savez_compressed(os.path.join(ROOT, "results", f"backtest_{tag}.npz"),
                        daily_cost=cost.sum(axis=3), short_hours=short.sum(axis=3),
                        valid_hours=valid.sum(axis=2), methods=np.array(METHOD_NAMES),
                        eval_days=ev, tau=tau, u=u, o=o, scale=scale, ba=d["ba"],
                        G=np.array(hist["G"]), labels=np.array(hist["labels"]),
                        W=np.array(hist["W"]), G_fto=np.array(hist["G_fto"]),
                        kappa=np.array(hist["kappa"]), G_km=np.array(hist["G_km"]),
                        origins=origins)
    tot = cost.sum(axis=(1, 2, 3))
    for k, name in enumerate(METHOD_NAMES):
        print(f"[{tag}] {name:12s} total cost / EW = {tot[k] / tot[1]:.4f}")


def run_variants(args, tag):
    """GDMA with G chosen by the plain hold-out minimum (rule='min') and soft
    GDMA with (G, kappa) chosen jointly by the hold-out."""
    d = np.load(os.path.join(ROOT, "data", "processed", "panel.npz"), allow_pickle=True)
    Y, F = d["Y"].astype(float), d["F"].astype(float)
    first = int(d["first_day"])
    M, N, Dn, H = F.shape
    u, o, tau = calibrate_costs(d["flex"], args.tau)
    P = candidate_policies(Y, F, tau, first)
    scale = np.nanmean(Y[:, first:], axis=(1, 2))
    start = first + RESID_DAYS + MAX_WINDOW
    origins = np.arange(start, Dn, STEP)
    names = ["GDMA-min", "GDMA-soft"]
    orders = np.full((2, N, Dn, H), np.nan)
    Gs, Gsoft, ksoft, prev, prev2 = [], [], [], None, None
    for n, d0 in enumerate(origins):
        win = np.arange(d0 - args.window, d0)
        out = np.arange(d0, min(d0 + STEP, Dn))
        Q, y = window_arrays(P, Y, win)
        Qo = P[:, :, out, :]
        g = select_grouped(Q, y, G_MAX, u, o, scale=scale, n_init=args.n_init,
                           init_labels=prev, rule="min")
        prev = g["labels"]
        orders[0][:, out] = np.einsum("mnlh,nm->nlh", Qo, g["W"][g["labels"]])
        Gs.append(g["G"])
        sg = select_soft_grouped(Q, y, G_MAX, u, o, scale=scale, n_init=args.n_init,
                                 init_labels=prev2)
        prev2 = sg["labels"]
        orders[1][:, out] = np.einsum("mnlh,nm->nlh", Qo, sg["W_units"])
        Gsoft.append(sg["G"])
        ksoft.append(sg["kappa"])
        if n % 40 == 0:
            print(f"[{tag} variants] origin {n + 1}/{len(origins)} G={g['G']} "
                  f"soft G={sg['G']} kappa={sg['kappa']:.2f}", flush=True)
    ev = np.arange(start, Dn)
    yy = Y[:, ev, :]
    valid = np.isfinite(yy)
    cost = np.where(valid[None], newsvendor_cost(yy[None], orders[:, :, ev, :],
                                                  u[None, :, None, None], o[None, :, None, None]), 0.0)
    short = np.where(valid[None], orders[:, :, ev, :] < yy[None], False)
    np.savez_compressed(os.path.join(ROOT, "results", f"variants_{tag}.npz"),
                        daily_cost=cost.sum(axis=3), short_hours=short.sum(axis=3),
                        methods=np.array(names), eval_days=ev, G=np.array(Gs),
                        G_soft=np.array(Gsoft), kappa_soft=np.array(ksoft))
    print(f"[{tag} variants] done")


if __name__ == "__main__":
    main()
