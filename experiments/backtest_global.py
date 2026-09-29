"""Rolling out-of-sample commitment backtest on the global (five-continent) panel.

usage: python backtest_global.py --window 28 [--tau calibrated|0.9]

Same protocol as backtest.py: every 7 days each method re-estimates its
combination from the last `window` days and commits hourly capacity for the
next 7 days.  The panel is unbalanced; a unit contributes to estimation and
evaluation only on days with demand data.  Hourly commitments of the main
methods are stored for the battery-siting stage.
"""
import argparse
import os
import sys
import time
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from gdma.core import (newsvendor_cost, fit_per_unit, fit_grouped, select_grouped,  # noqa: E402
                       select_soft_grouped, fit_kmeans_two_step, cost_matrix)
from backtest import (candidate_policies, window_arrays, fto_orders, RESID_DAYS,  # noqa: E402
                      STEP, G_MAX)

METHOD_NAMES = ["Official", "EW", "Select", "Pooled", "PerUnit", "Shrink", "KMeans2S",
                "FTO-PerUnit", "FTO-Grouped", "GDMA", "GDMA-soft"]
SAVE_ORDERS = ["Official", "EW", "Pooled", "PerUnit", "Shrink", "FTO-PerUnit", "GDMA", "GDMA-soft"]
MAX_WINDOW = 56


def fast_candidate_policies(Y, F, tau, first):
    """Vectorised version of backtest.candidate_policies (identical output)."""
    from backtest import REL_FLOOR
    M, N, Dn, H = F.shape
    P = np.full(F.shape, np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        floor = REL_FLOOR * np.nanmean(Y, axis=(1, 2))
        rel = (Y[None] - F) / np.maximum(F, floor[None, :, None, None])
        ii = np.arange(N)
        for d in range(first + RESID_DAYS, Dn):
            win = rel[:, :, d - RESID_DAYS:d, :].reshape(M, N, -1)
            q = np.nanquantile(win, tau, axis=2)            # (N_tau, M, N)
            qi = q[ii, :, ii]                                # (N, M)
            P[:, :, d, :] = F[:, :, d, :] * (1.0 + qi.T[:, :, None])
    return np.maximum(P, 0.0)


def calibrate_costs(flex, tau_spec):
    """Critical ratios from interchange flexibility where it is measured:
    tau falls linearly in the flexibility rank from 0.95 (least flexible) to
    0.80 (most flexible); units without interchange data get 0.875."""
    N = len(flex)
    if tau_spec == "calibrated":
        tau = np.full(N, 0.875)
        known = np.isfinite(flex)
        r = np.argsort(np.argsort(flex[known])) / (known.sum() - 1)
        tau[known] = 0.95 - 0.15 * r
    else:
        tau = np.full(N, float(tau_spec))
    return tau / (1 - tau), np.ones(N), tau


WEATHER_FILES = {"exact": ("Weather", "global_weather_noise0.npy"),
                 "noise2": ("Weather", "global_weather_noise2.npy")}


def load_global(tau_spec, with_chronos=True, weather="none"):
    """Global panel with (optionally) the Chronos candidate appended, the cost
    parameters and the candidate commitments (cached on disk per tau_spec)."""
    d = np.load(os.path.join(ROOT, "data", "processed", "global_panel.npz"), allow_pickle=True)
    Y, F = d["Y"].astype(float), d["F"].astype(float)
    methods = list(d["methods"])
    cp = os.path.join(ROOT, "data", "processed", "global_chronos.npy")
    if with_chronos and os.path.exists(cp):
        C = np.load(cp).astype(float)
        with np.errstate(all="ignore"):
            fill = np.nanmean(F, axis=0)
        C = np.where(np.isfinite(C), C, fill)
        C[~np.isfinite(F[0])] = np.nan
        F = np.concatenate([F, C[None]], axis=0)
        methods.append("Chronos")
    if weather != "none":
        name, fn = WEATHER_FILES[weather]
        Wc = np.load(os.path.join(ROOT, "data", "processed", fn)).astype(float)
        with np.errstate(all="ignore"):
            fill = np.nanmean(F, axis=0)
        Wc = np.where(np.isfinite(Wc), Wc, fill)
        Wc[~np.isfinite(F[0])] = np.nan
        F = np.concatenate([F, Wc[None]], axis=0)
        methods.append(name)
    first = int(d["first_day"])
    u, o, tau = calibrate_costs(d["flex"], tau_spec)
    wtag = "" if weather == "none" else f"_weather-{weather}"
    pp = os.path.join(ROOT, "data", "processed", f"global_policies_{len(methods)}_{tau_spec}{wtag}.npy")
    if os.path.exists(pp):
        P = np.load(pp).astype(float)
    else:
        P = fast_candidate_policies(Y, F, tau, first)
        np.save(pp, P)
        P = P.astype(float)
    return d, Y, F, P, u, o, tau, methods, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--tau", default="calibrated")
    ap.add_argument("--n_init", type=int, default=4)
    ap.add_argument("--weather", default="none", choices=["none", "exact", "noise2"])
    args = ap.parse_args()
    tag = f"global_w{args.window}_tau{args.tau}" + ("" if args.weather == "none" else f"_weather-{args.weather}")

    d, Y, F, P, u, o, tau, cand, first = load_global(args.tau, weather=args.weather)
    M, N, Dn, H = F.shape
    print("candidates:", cand, flush=True)
    with np.errstate(all="ignore"):
        scale = np.nanmean(Y, axis=(1, 2))

    start = first + RESID_DAYS + MAX_WINDOW
    origins = np.arange(start, Dn, STEP)
    K = len(METHOD_NAMES)
    orders = np.full((K, N, Dn, H), np.nan, dtype=np.float32)
    hist = dict(G=[], labels=[], W=[], kappa=[], kappa_shrink=[], G_km=[], G_fto=[], active=[])
    prev = None
    t0 = time.time()
    for n, d0 in enumerate(origins):
        win = np.arange(d0 - args.window, d0)
        out = np.arange(d0, min(d0 + STEP, Dn))
        Q, y = window_arrays(P, Y, win)
        Qo = np.nan_to_num(P[:, :, out, :])
        active = (y != 0).sum(axis=1) >= 24          # at least one day of data
        comb = lambda Wn: np.einsum("mnlh,nm->nlh", Qo, Wn)
        a = np.flatnonzero(active)
        Qa_, ya_ = Q[a], y[a]
        ua, oa, sa = u[a], o[a], scale[a]

        def full(Wa, fill=None):
            Wn = np.full((N, M), 1.0 / M) if fill is None else np.tile(fill, (N, 1))
            Wn[a] = Wa
            return Wn

        orders[0][:, out] = Qo[0]
        orders[1][:, out] = Qo.mean(axis=0)
        Cs = newsvendor_cost(ya_[..., None], Qa_, ua[:, None, None], oa[:, None, None]).sum(1)
        sel = np.eye(M)[Cs.argmin(1)]
        orders[2][:, out] = comb(full(sel))
        pooled = fit_grouped(Qa_, ya_, 1, ua, oa, scale=sa)["W"][0]
        orders[3][:, out] = comb(np.tile(pooled, (N, 1)))
        Wu = fit_per_unit(Qa_, ya_, ua, oa, scale=sa)
        orders[4][:, out] = comb(full(Wu, pooled))

        T1 = int(np.ceil(2 / 3 * Q.shape[1]))
        Wa1 = fit_per_unit(Qa_[:, :T1], ya_[:, :T1], ua, oa, scale=sa)
        pa1 = fit_grouped(Qa_[:, :T1], ya_[:, :T1], 1, ua, oa, scale=sa)["W"][0]
        ks = np.linspace(0, 1, 5)
        sc = [np.diag(cost_matrix(Qa_[:, T1:], ya_[:, T1:], (1 - k) * Wa1 + k * pa1, ua, oa)).sum() for k in ks]
        kap = ks[int(np.argmin(sc))]
        orders[5][:, out] = comb(full((1 - kap) * Wu + kap * pooled, pooled))

        sc = []
        for G in range(1, G_MAX + 1):
            f = fit_kmeans_two_step(Qa_[:, :T1], ya_[:, :T1], G, ua, oa, scale=sa, W_unit=Wa1)
            sc.append(cost_matrix(Qa_[:, T1:], ya_[:, T1:], f["W"], ua, oa)[np.arange(len(a)), f["labels"]].sum())
        Gk = int(np.argmin(sc)) + 1
        km = fit_kmeans_two_step(Qa_, ya_, Gk, ua, oa, scale=sa, W_unit=Wu)
        orders[6][:, out] = comb(full(km["W"][km["labels"]], pooled))

        Fq, fy = window_arrays(F, Y, win)
        Fq, fy = np.nan_to_num(Fq[a]), fy[a]
        Wf = fit_per_unit(Fq, fy, loss="squared", scale=sa)
        Ff = np.nan_to_num(F)
        with np.errstate(all="ignore"):
            orders[7][a[:, None], out[None, :]] = fto_orders(Ff[:, a], Y[a], Wf, np.arange(len(a)), tau[a], win, out)
            gf = select_grouped(Fq, fy, G_MAX, loss="squared", scale=sa, n_init=args.n_init)
            orders[8][a[:, None], out[None, :]] = fto_orders(Ff[:, a], Y[a], gf["W"], gf["labels"], tau[a], win, out)

        init = None if prev is None else prev[a]
        sg = select_soft_grouped(Qa_, ya_, G_MAX, ua, oa, scale=sa, n_init=args.n_init, init_labels=init)
        labels = np.full(N, -1)
        labels[a] = sg["labels"]
        prev = np.where(labels >= 0, labels, 0)
        Wg = sg["W"][sg["labels"]]
        orders[9][:, out] = comb(full(Wg, pooled))
        orders[10][:, out] = comb(full(sg["W_units"], pooled))

        hist["G"].append(sg["G"]); hist["labels"].append(labels); hist["W"].append(full(Wg, pooled))
        hist["kappa"].append(sg["kappa"]); hist["kappa_shrink"].append(kap)
        hist["G_km"].append(Gk); hist["G_fto"].append(gf["G"]); hist["active"].append(active)
        if n % 20 == 0:
            print(f"[{tag}] origin {n + 1}/{len(origins)} active={active.sum()} G={sg['G']} "
                  f"kappa={sg['kappa']:.2f} {time.time() - t0:.0f}s", flush=True)

    ev = np.arange(start, Dn)
    yy = Y[:, ev, :]
    # scored only where demand and every candidate commitment exist
    valid = np.isfinite(yy) & np.isfinite(P[:, :, ev, :]).all(axis=0)
    yz = np.where(valid, yy, 0.0)
    cost = np.where(valid[None], newsvendor_cost(yz[None], np.nan_to_num(orders[:, :, ev, :]),
                                                  u[None, :, None, None], o[None, :, None, None]), 0.0)
    short = np.where(valid[None], orders[:, :, ev, :] < yz[None], False)
    keep = [METHOD_NAMES.index(m) for m in SAVE_ORDERS]
    np.savez_compressed(os.path.join(ROOT, "results", f"backtest_{tag}.npz"),
                        daily_cost=cost.sum(axis=3), short_hours=short.sum(axis=3),
                        valid_hours=valid.sum(axis=2), methods=np.array(METHOD_NAMES),
                        eval_days=ev, tau=tau, u=u, o=o, scale=scale, code=d["code"],
                        continent=d["continent"], G=np.array(hist["G"]),
                        labels=np.array(hist["labels"]), W=np.array(hist["W"]),
                        kappa=np.array(hist["kappa"]), kappa_shrink=np.array(hist["kappa_shrink"]),
                        G_km=np.array(hist["G_km"]), G_fto=np.array(hist["G_fto"]),
                        active=np.array(hist["active"]), origins=origins,
                        candidates=np.array(cand))
    np.savez_compressed(os.path.join(ROOT, "results", f"orders_{tag}.npz"),
                        orders=orders[keep][:, :, ev, :], methods=np.array(SAVE_ORDERS), eval_days=ev)
    tot = cost.sum(axis=(1, 2, 3))
    for k, name in enumerate(METHOD_NAMES):
        print(f"[{tag}] {name:12s} total cost / Official = {tot[k] / tot[0]:.4f}")


if __name__ == "__main__":
    main()
