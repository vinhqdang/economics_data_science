"""Monte Carlo study of grouped decision-focused model averaging.

DGP.  Unit i (critical ratio tau_i ~ U[0.6, 0.95], u_i = tau_i/(1-tau_i), o_i=1)
has demand y_it = mu_i + x_it with x_it a stationary AR(1) (rho=0.5), and M = 5
candidate point forecasts f_itm = y_it - e_itm.  Forecast errors are serially
correlated (AR(1), rho=0.5) with innovation scales s_{i,m} and skewness signs
k_{i,m} that define the latent heterogeneity:
  'grouped'     : (s_i, k_i) take one of G0 = 3 group-specific profiles
  'homogeneous' : one common profile for all units
  'continuous'  : error scales mix the three profiles with Dirichlet(1,1,1)
                  weights (skewness from the dominant profile), so no two units
                  share the same optimal weights
Candidate decisions are q_itm = f_itm + Qhat_tau_i(e_im) (residual quantile from
a pre-sample of 200 periods).  Weights are estimated on T periods and evaluated
on 1000 fresh periods.
"""
import os
import sys
import time
import itertools
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from gdma.core import (newsvendor_cost, fit_per_unit, fit_grouped, select_grouped,  # noqa: E402
                       fit_kmeans_two_step, cost_matrix, fit_weights_exact,
                       select_soft_grouped)

M = 5
PROFILES_S = np.array([[1.0, 2.0, 2.0, 1.5, 3.0],
                       [2.5, 1.0, 2.0, 1.5, 3.0],
                       [2.5, 2.0, 1.0, 3.0, 1.2]])
PROFILES_K = np.array([[+1, -1, +1, 0, -1],
                       [-1, +1, 0, -1, +1],
                       [+1, +1, -1, +1, 0]])
T_TEST, T_PRE = 1000, 200
METHODS = ["EW", "Pooled", "PerUnit", "Shrink", "KMeans2S", "GDMA", "GDMA-soft", "Oracle"]


def ar1(shape, rho, rng):
    z = rng.standard_normal(shape)
    out = np.empty(shape)
    out[..., 0] = z[..., 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + z[..., t]
    return out * np.sqrt(1 - rho ** 2)


def skewed(shape, k, rng):
    """Standardised innovations: k=+1 right-skewed, -1 left-skewed, 0 symmetric."""
    g = (rng.gamma(2.0, 1.0, shape) - 2.0) / np.sqrt(2.0)
    n = rng.standard_normal(shape)
    return np.where(k == 0, n, k * g)


def simulate(N, T, design, rng):
    if design == "grouped":
        g0 = rng.integers(3, size=N)
        mix = np.eye(3)[g0]
    elif design == "homogeneous":
        g0 = np.zeros(N, int)
        mix = np.tile([1.0, 0, 0], (N, 1))
    else:
        mix = rng.dirichlet(np.ones(3), size=N)
        g0 = mix.argmax(1)
    s = mix @ PROFILES_S
    k = PROFILES_K[g0]
    tau = rng.uniform(0.6, 0.95, N)
    mu = rng.uniform(20, 100, N)
    TT = T_PRE + T + T_TEST
    y = mu[:, None] + 5 * ar1((N, TT), 0.5, rng)
    innov = skewed((N, M, TT), k[:, :, None], rng)
    e = np.empty_like(innov)
    e[..., 0] = innov[..., 0]
    for t in range(1, TT):
        e[..., t] = 0.5 * e[..., t - 1] + np.sqrt(1 - 0.25) * innov[..., t]
    e *= s[:, :, None]
    f = y[:, None, :] - e
    qadj = np.array([np.quantile(e[i, :, :T_PRE], tau[i], axis=1) for i in range(N)])
    q = f + qadj[:, :, None]
    Q = q.transpose(0, 2, 1)   # (N, TT, M)
    return Q, y, tau, g0


def one_rep(args):
    N, T, design, rep = args
    rng = np.random.default_rng(10_000 * rep + 97 * N + T + {"grouped": 1, "homogeneous": 2, "continuous": 3}[design])
    Q, y, tau, g0 = simulate(N, T, design, rng)
    u, o = tau / (1 - tau), np.ones(N)
    tr = slice(T_PRE, T_PRE + T)
    te = slice(T_PRE + T, None)
    Qa, ya, Qb, yb = Q[:, tr], y[:, tr], Q[:, te], y[:, te]
    scale = y[:, :T_PRE].mean(1)
    res = {}
    t0 = time.time()

    def test_cost(Wn):
        return np.diag(cost_matrix(Qb, yb, Wn, u, o)).sum() if Wn.ndim == 2 and len(Wn) == N else None

    res["EW"] = test_cost(np.full((N, M), 1 / M))
    pooled = fit_grouped(Qa, ya, 1, u, o, scale=scale)["W"][0]
    res["Pooled"] = test_cost(np.tile(pooled, (N, 1)))
    Wu = fit_per_unit(Qa, ya, u, o, scale=scale)
    res["PerUnit"] = test_cost(Wu)
    T1 = int(np.ceil(2 / 3 * T))
    Wa = fit_per_unit(Qa[:, :T1], ya[:, :T1], u, o, scale=scale)
    pa = fit_grouped(Qa[:, :T1], ya[:, :T1], 1, u, o, scale=scale)["W"][0]
    ks = np.linspace(0, 1, 5)
    sc = [np.diag(cost_matrix(Qa[:, T1:], ya[:, T1:], (1 - k) * Wa + k * pa, u, o)).sum() for k in ks]
    kap = ks[int(np.argmin(sc))]
    res["Shrink"] = test_cost((1 - kap) * Wu + kap * pooled)
    sc = []
    for G in range(1, 7):
        fk = fit_kmeans_two_step(Qa[:, :T1], ya[:, :T1], G, u, o, scale=scale, W_unit=Wa)
        sc.append(cost_matrix(Qa[:, T1:], ya[:, T1:], fk["W"], u, o)[np.arange(N), fk["labels"]].sum())
    km = fit_kmeans_two_step(Qa, ya, int(np.argmin(sc)) + 1, u, o, scale=scale, W_unit=Wu)
    res["KMeans2S"] = test_cost(km["W"][km["labels"]])
    g = select_grouped(Qa, ya, 6, u, o, scale=scale, n_init=6, seed=rep)
    res["GDMA"] = test_cost(g["W"][g["labels"]])
    sg = select_soft_grouped(Qa, ya, 6, u, o, scale=scale, n_init=6, seed=rep)
    res["GDMA-soft"] = test_cost(sg["W_units"])
    # infeasible benchmark: exact per-unit optimum on the test sample itself
    Wor = np.vstack([fit_weights_exact(Qb[i], yb[i], u[i], o[i]) for i in range(N)])
    res["Oracle"] = test_cost(Wor)
    out = {m: res[m] / res["Oracle"] for m in METHODS}
    out.update(N=N, T=T, design=design, rep=rep, G_hat=g["G"], G_soft=sg["G"], kappa=sg["kappa"],
               ARI=adjusted_rand_score(g0, g["labels"]) if design == "grouped" else np.nan,
               ARI_km=adjusted_rand_score(g0, km["labels"]) if design == "grouped" else np.nan,
               secs=time.time() - t0)
    return out


def main():
    import pandas as pd
    reps = int(os.environ.get("REPS", 50))
    grid = list(itertools.product([50, 200], [48, 168, 672], ["grouped", "homogeneous", "continuous"],
                                  range(reps)))
    with Pool(int(os.environ.get("POOL", 4))) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_rep, grid, chunksize=1)):
            rows.append(r)
            if i % 50 == 0:
                print(i, len(grid), r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "simulation_raw.csv"), index=False)
    summ = df.groupby(["design", "N", "T"])[METHODS + ["G_hat", "ARI", "ARI_km"]].mean()
    summ.to_csv(os.path.join(ROOT, "results", "simulation_summary.csv"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
