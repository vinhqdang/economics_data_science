"""Grouped decision-focused model averaging (GDMA).

Notation used throughout
------------------------
N units (items), T periods in the estimation window, M candidate policies.
Q : array (N, T, M)   candidate decisions (order quantities) q_{it}^{(m)}
Y : array (N, T)      realised demand y_{it}
u : array (N,)        unit underage cost (lost margin per unit short)
o : array (N,)        unit overage cost (loss per unsold unit)

The newsvendor cost of a combined decision q = Q_it' w is
    C_i(q, y) = u_i (y - q)^+ + o_i (q - y)^+
             = (u_i + o_i) * rho_{tau_i}(y - q),   tau_i = u_i / (u_i + o_i),
where rho_tau is the check (pinball) function.  GDMA solves
    min_{g in {1..G}^N, w_1..w_G in simplex}  sum_i sum_t C_i(Q_it' w_{g_i}, y_it).
"""
import numpy as np
from scipy.optimize import minimize

__all__ = [
    "newsvendor_cost", "squared_cost", "fit_weights", "cost_matrix",
    "fit_grouped", "select_grouped", "fit_per_unit", "fit_kmeans_two_step",
    "fit_weights_exact", "error_scale", "select_soft_grouped",
]


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------
def newsvendor_cost(y, q, u, o):
    """Elementwise newsvendor cost u (y-q)^+ + o (q-y)^+ (broadcasting)."""
    r = y - q
    return u * np.maximum(r, 0.0) + o * np.maximum(-r, 0.0)


def squared_cost(y, q, scale):
    """Scaled squared error ((y - q) / scale)^2, used by forecast-then-order."""
    return ((y - q) / scale) ** 2


# --------------------------------------------------------------------------
# weight estimation on the simplex for one pooled sample
# --------------------------------------------------------------------------
def _smooth_pinball(w, Q, y, a, tau, eps):
    """Softplus smoothing of a * rho_tau(y - Qw); exact as eps -> 0."""
    r = y - Q @ w
    z = -r / eps
    val = np.sum(a * (tau * r + eps * np.logaddexp(0.0, z)))
    sig = 0.5 * (1.0 + np.tanh(0.5 * z))
    grad = -(Q.T @ (a * (tau - sig)))
    return val, grad


def _squared(w, Q, y, a):
    r = y - Q @ w
    return np.sum(a * r * r), -2.0 * (Q.T @ (a * r))


def error_scale(Q, Y):
    """Typical absolute error of the equal-weight decision, per unit (N,)."""
    return np.abs(Y - Q.mean(axis=2)).mean(axis=1) + 1e-8


def fit_weights(Q, y, u=None, o=None, loss="newsvendor", scale=None, w0=None,
                eps_frac=0.02, center=None, lam=0.0):
    """Minimise the (pooled) empirical loss over the unit simplex.

    Q : (n, M), y : (n,); u, o, scale : (n,) row-level parameters.
    For the newsvendor loss `scale` is the per-row error scale that sets the
    smoothing bandwidth eps = eps_frac * scale; for the squared loss it is the
    per-row normalisation of the errors.
    With lam > 0 a ridge penalty lam * n * ||w - center||^2 is added, which is
    used by the shrinkage benchmark.
    """
    n, M = Q.shape
    if w0 is None:
        w0 = np.full(M, 1.0 / M)
    if loss == "newsvendor":
        a = u + o
        tau = u / a
        # smoothing bandwidth relative to the size of the decision errors
        s = scale if scale is not None else np.abs(y - Q.mean(axis=1)).mean() + 1e-8
        eps = eps_frac * np.broadcast_to(s, y.shape) + 1e-8
        norm = 1.0 / max(np.sum(a * eps), 1e-12)

        def fun(w):
            v, g = _smooth_pinball(w, Q, y, a, tau, eps)
            return v * norm, g * norm
    elif loss == "squared":
        a = 1.0 / scale ** 2
        norm = 1.0 / n

        def fun(w):
            v, g = _squared(w, Q, y, a)
            return v * norm, g * norm
    else:
        raise ValueError(loss)

    if lam > 0:
        base = fun

        def fun(w):  # noqa: F811
            v, g = base(w)
            d = w - center
            return v + lam * d @ d, g + 2.0 * lam * d

    res = minimize(fun, w0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * M,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones(M)}],
                   options={"maxiter": 200, "ftol": 1e-10})
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def fit_weights_exact(Q, y, u, o):
    """Exact newsvendor-optimal simplex weights by linear programming (HiGHS)."""
    from scipy.optimize import linprog
    from scipy import sparse
    n, M = Q.shape
    A = sparse.hstack([sparse.csr_matrix(Q), sparse.eye(n), -sparse.eye(n)])
    A = sparse.vstack([A, sparse.hstack([sparse.csr_matrix(np.ones((1, M))),
                                         sparse.csr_matrix((1, 2 * n))])]).tocsr()
    c = np.r_[np.zeros(M), np.broadcast_to(u, n), np.broadcast_to(o, n)]
    r = linprog(c, A_eq=A, b_eq=np.r_[y, 1.0], bounds=(0, None), method="highs")
    w = np.clip(r.x[:M], 0, None)
    return w / w.sum()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _flat(Q, Y, u, o, idx, scale):
    """Stack the observations of units idx into one pooled sample."""
    T = Y.shape[1]
    Qg = Q[idx].reshape(-1, Q.shape[2])
    yg = Y[idx].reshape(-1)
    ug = np.repeat(u[idx], T) if u is not None else None
    og = np.repeat(o[idx], T) if o is not None else None
    sg = np.repeat(scale[idx], T) if scale is not None else None
    return Qg, yg, ug, og, sg


def cost_matrix(Q, Y, W, u=None, o=None, loss="newsvendor", scale=None):
    """C[i, g] = total loss of unit i when it uses weight vector W[g]."""
    D = np.einsum("itm,gm->itg", Q, W)  # (N, T, G)
    if loss == "newsvendor":
        c = newsvendor_cost(Y[:, :, None], D, u[:, None, None], o[:, None, None])
    else:
        c = squared_cost(Y[:, :, None], D, scale[:, None, None])
    return c.sum(axis=1)


def _group_fit(Q, Y, u, o, labels, G, loss, scale, W0):
    W = np.empty((G, Q.shape[2]))
    for g in range(G):
        idx = np.flatnonzero(labels == g)
        Qg, yg, ug, og, sg = _flat(Q, Y, u, o, idx, scale)
        W[g] = fit_weights(Qg, yg, ug, og, loss=loss, scale=sg,
                           w0=None if W0 is None else W0[g])
    return W


def fit_per_unit(Q, Y, u=None, o=None, loss="newsvendor", scale=None):
    """Unit-specific weights (the G = N extreme)."""
    if loss == "newsvendor":
        scale = error_scale(Q, Y)
    N = Y.shape[0]
    return np.vstack([
        fit_weights(Q[i], Y[i], None if u is None else np.full(Y.shape[1], u[i]),
                    None if o is None else np.full(Y.shape[1], o[i]), loss=loss,
                    scale=None if scale is None else np.full(Y.shape[1], scale[i]))
        for i in range(N)])


# --------------------------------------------------------------------------
# grouped estimator
# --------------------------------------------------------------------------
def fit_grouped(Q, Y, G, u=None, o=None, loss="newsvendor", scale=None,
                n_init=8, max_iter=30, seed=0, W_unit=None, init_labels=None):
    """Alternating (Lloyd-type) minimisation of the grouped criterion.

    Returns dict(labels, W, objective).  Initial centres are drawn by a
    k-means++ rule in cost space from the unit-specific solutions W_unit,
    which makes the search considerably more reliable than random starts.
    """
    N, T, M = Q.shape
    rng = np.random.default_rng(seed)
    if loss == "newsvendor":
        scale = error_scale(Q, Y)
    if G == 1:
        W = _group_fit(Q, Y, u, o, np.zeros(N, int), 1, loss, scale, None)
        obj = cost_matrix(Q, Y, W, u, o, loss, scale).sum()
        return dict(labels=np.zeros(N, int), W=W, objective=obj)
    if W_unit is None:
        W_unit = fit_per_unit(Q, Y, u, o, loss, scale)
    # cost of every unit under every other unit's own weights (N x N)
    C_all = cost_matrix(Q, Y, W_unit, u, o, loss, scale)
    own = np.diag(C_all)
    best = None
    starts = n_init + (init_labels is not None)
    for r in range(starts):
        if r == n_init:  # warm start, e.g. from the previous estimation window
            lab0 = np.asarray(init_labels) % G
            for g in range(G):
                if not np.any(lab0 == g):
                    lab0[rng.integers(N)] = g
            W = _group_fit(Q, Y, u, o, lab0, G, loss, scale, None)
        else:
            # k-means++ seeding: next centre chosen with prob. prop. to regret
            centres = [rng.integers(N)]
            for _ in range(1, G):
                regret = C_all[:, centres].min(axis=1) - own
                p = np.maximum(regret, 0) + 1e-12
                centres.append(rng.choice(N, p=p / p.sum()))
            W = W_unit[centres].copy()
        labels = None
        for _ in range(max_iter):
            C = cost_matrix(Q, Y, W, u, o, loss, scale)
            new = C.argmin(axis=1)
            # repair empty groups with the worst-fitted units
            for g in range(G):
                if not np.any(new == g):
                    worst = np.argmax(C[np.arange(N), new] - own)
                    new[worst] = g
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            W = _group_fit(Q, Y, u, o, labels, G, loss, scale, W)
        obj = cost_matrix(Q, Y, W, u, o, loss, scale)[np.arange(N), labels].sum()
        if best is None or obj < best["objective"]:
            best = dict(labels=labels.copy(), W=W.copy(), objective=obj)
    return best


def select_grouped(Q, Y, G_max, u=None, o=None, loss="newsvendor", scale=None,
                   frac=2 / 3, n_init=8, seed=0, init_labels=None, rule="1se"):
    """Choose G by temporal hold-out inside the window, then refit on it all.

    The window is split into an early fitting block and a late validation
    block; each G in 1..G_max is fitted on the early block and scored on the
    late block with its fitted memberships and weights.  With rule='1se' the
    smallest G within one standard error of the best hold-out cost is chosen
    (standard error of the summed unit-level cost differences); rule='min'
    takes the minimiser.
    """
    T = Y.shape[1]
    T1 = max(int(np.ceil(frac * T)), 2)
    Qa, Ya, Qb, Yb = Q[:, :T1], Y[:, :T1], Q[:, T1:], Y[:, T1:]
    Wu = fit_per_unit(Qa, Ya, u, o, loss, scale)
    unit_scores = []
    for G in range(1, G_max + 1):
        fit = fit_grouped(Qa, Ya, G, u, o, loss, scale, n_init, seed=seed, W_unit=Wu)
        C = cost_matrix(Qb, Yb, fit["W"], u, o, loss, scale)
        unit_scores.append(C[np.arange(len(Y)), fit["labels"]])
    unit_scores = np.array(unit_scores)            # (G_max, N)
    scores = unit_scores.sum(axis=1)
    G_best = int(np.argmin(scores))
    G_star = G_best + 1
    if rule == "1se":
        # smallest G whose hold-out cost is within one standard error of the
        # minimum; the standard error is that of the paired unit-level difference
        for G in range(G_best):
            diff = unit_scores[G] - unit_scores[G_best]
            se = diff.std(ddof=1) * np.sqrt(len(diff))
            if diff.sum() <= se:
                G_star = G + 1
                break
    fit = fit_grouped(Q, Y, G_star, u, o, loss, scale, n_init, seed=seed,
                      init_labels=init_labels)
    fit["G"] = G_star
    fit["holdout_scores"] = np.array(scores)
    return fit


def fit_kmeans_two_step(Q, Y, G, u=None, o=None, loss="newsvendor", scale=None,
                        seed=0, W_unit=None):
    """Benchmark: k-means on unit-specific weight vectors, then pooled refit."""
    from sklearn.cluster import KMeans
    if W_unit is None:
        W_unit = fit_per_unit(Q, Y, u, o, loss, scale)
    if G == 1:
        labels = np.zeros(len(Y), int)
    else:
        labels = KMeans(G, n_init=10, random_state=seed).fit_predict(W_unit)
    if loss == "newsvendor":
        scale = error_scale(Q, Y)
    W = _group_fit(Q, Y, u, o, labels, G, loss, scale, None)
    return dict(labels=labels, W=W)


def select_soft_grouped(Q, Y, G_max, u=None, o=None, loss="newsvendor", scale=None,
                        frac=2 / 3, n_init=8, seed=0, init_labels=None,
                        kappas=np.linspace(0.0, 1.0, 5)):
    """Soft GDMA: unit weights shrunk towards their segment's weights,
        w_i = (1 - kappa) * w_i^unit + kappa * w_{g_i}.
    G is chosen by the one-standard-error hold-out rule of `select_grouped`;
    kappa is then chosen on the same hold-out block given G.  kappa = 1 is
    GDMA, G = 1 is shrinkage towards pooled weights, kappa = 0 is unit-specific
    weights.
    """
    N, T = Y.shape
    T1 = max(int(np.ceil(frac * T)), 2)
    Qa, Ya, Qb, Yb = Q[:, :T1], Y[:, :T1], Q[:, T1:], Y[:, T1:]
    Wu_a = fit_per_unit(Qa, Ya, u, o, loss, scale)
    fits, unit_scores = [], []
    for G in range(1, G_max + 1):
        fit = fit_grouped(Qa, Ya, G, u, o, loss, scale, n_init, seed=seed, W_unit=Wu_a)
        fits.append(fit)
        C = cost_matrix(Qb, Yb, fit["W"], u, o, loss, scale)
        unit_scores.append(C[np.arange(N), fit["labels"]])
    unit_scores = np.array(unit_scores)
    G_best = int(np.argmin(unit_scores.sum(axis=1)))
    G_star = G_best + 1
    for G in range(G_best):
        diff = unit_scores[G] - unit_scores[G_best]
        if diff.sum() <= diff.std(ddof=1) * np.sqrt(N):
            G_star = G + 1
            break
    fa = fits[G_star - 1]
    Wg_a = fa["W"][fa["labels"]]
    sc = [np.diag(cost_matrix(Qb, Yb, (1 - k) * Wu_a + k * Wg_a, u, o, loss, scale)).sum()
          for k in kappas]
    k_star = float(kappas[int(np.argmin(sc))])
    Wu = fit_per_unit(Q, Y, u, o, loss, scale)
    fit = fit_grouped(Q, Y, G_star, u, o, loss, scale, n_init, seed=seed,
                      W_unit=Wu, init_labels=init_labels)
    fit["W_units"] = (1 - k_star) * Wu + k_star * fit["W"][fit["labels"]]
    fit["G"], fit["kappa"] = G_star, k_star
    return fit
