"""Model confidence set (Hansen, Lunde and Nason, 2011) with the T_max
statistic and a moving-block bootstrap over days, and block-bootstrap
confidence intervals for cost ratios."""
import numpy as np


def _block_indices(T, block, rng):
    n = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=n)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]


def model_confidence_set(L, alpha=0.10, B=1000, block=7, seed=0):
    """L: (K, T) daily losses of K methods.  Returns (in_set, pvalues): the
    boolean membership of the (1 - alpha) MCS and the MCS p-value of every
    method (the method is in the set iff its p-value exceeds alpha)."""
    L = np.asarray(L, float)
    K, T = L.shape
    rng = np.random.default_rng(seed)
    idx = [_block_indices(T, block, rng) for _ in range(B)]
    alive = list(range(K))
    pval = np.zeros(K)
    running_max = 0.0
    while len(alive) > 1:
        La = L[alive]
        d = La - La.mean(axis=0, keepdims=True)          # loss relative to the set average
        dbar = d.mean(axis=1)
        boot = np.array([d[:, ix].mean(axis=1) for ix in idx])   # (B, k)
        se = boot.std(axis=0, ddof=1) + 1e-12
        t = dbar / se
        tmax = t.max()
        tb = ((boot - dbar[None, :]) / se[None, :]).max(axis=1)
        p = float((tb >= tmax).mean())
        worst = alive[int(np.argmax(t))]
        running_max = max(running_max, p)
        pval[worst] = running_max
        if p >= alpha:
            for k in alive:
                pval[k] = max(pval[k], running_max)
            break
        alive.remove(worst)
    else:
        pval[alive[0]] = 1.0
    if len(alive) == 1:
        pval[alive[0]] = 1.0
    return pval > alpha, pval


def ratio_ci(num, den, B=1000, block=7, level=0.90, seed=0):
    """Block-bootstrap interval for sum(num) / sum(den) over days."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    T = len(num)
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(B):
        ix = _block_indices(T, min(block, T), rng)
        r.append(num[ix].sum() / max(den[ix].sum(), 1e-12))
    a = (1 - level) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))
