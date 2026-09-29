"""Battery siting on top of a learned day-ahead commitment policy.

Stage 2 (operations) fixes a commitment rule, which leaves each unit with an
hourly shortfall s_t = (y_t - q_t)^+ and surplus x_t = (q_t - y_t)^+.
A battery of power P (MW) and energy E = duration * P (MWh) placed in the unit
discharges to cover shortfalls and recharges from committed-but-unused capacity
(which is already paid for), with round-trip efficiency eta applied on charging.
Every MWh of shortfall it covers saves the unit's underage cost u_i.

Stage 1 (planning) allocates a storage budget across units.  Because the value
V_i(P) of battery power is concave in P (a larger battery covers ever rarer
shortfalls), a greedy allocation over small increments of power in order of
marginal value is optimal for the budget-constrained problem
    max_{P_i >= 0} sum_i V_i(P_i)  s.t.  sum_i P_i <= B.
"""
import numpy as np

__all__ = ["battery_dispatch", "value_curves", "allocate_budget", "optimal_sizes",
           "allocate_with_floors"]


def battery_dispatch(short, surplus, P, duration=4.0, eta=0.85, soc0=0.5):
    """Greedy chronological dispatch for many units at once.

    short, surplus : arrays (N, T) of hourly shortfall and surplus (MWh)
    P              : array (N,) battery power (MW)
    Returns the energy discharged into shortfalls, array (N, T).
    """
    N, T = short.shape
    E = duration * P
    soc = soc0 * E
    out = np.zeros((N, T))
    for t in range(T):
        d = np.minimum(np.minimum(P, short[:, t]), soc)
        soc = soc - d
        c = np.minimum(np.minimum(P, surplus[:, t]), (E - soc) / eta)
        soc = soc + eta * c
        out[:, t] = d
    return out


def value_curves(short, surplus, u, grid, years, **kw):
    """V[i, k] = annual avoided shortfall cost of a battery of power grid[i, k].

    grid : array (N, K) of candidate power levels per unit, increasing in k,
           with grid[:, 0] = 0.
    years: length of the evaluation sample in years.
    """
    N, K = grid.shape
    V = np.zeros((N, K))
    for k in range(1, K):
        V[:, k] = u * battery_dispatch(short, surplus, grid[:, k], **kw).sum(axis=1) / years
    return V


def _concave_increments(grid, V):
    """Marginal values per MW of each grid step, made non-increasing (the
    least concave majorant of a sampled concave curve is taken for safety)."""
    dP = np.diff(grid, axis=1)
    dV = np.diff(V, axis=1)
    m = dV / np.where(dP > 0, dP, np.inf)
    return dP, np.minimum.accumulate(m, axis=1)


def allocate_budget(grid, V, budget, P0=None, eligible=None):
    """Greedy allocation of `budget` MW of battery power across units, starting
    from P0 (default zero) and only adding power to `eligible` units."""
    dP, m = _concave_increments(grid, V)
    N, K1 = m.shape
    if eligible is not None:
        m = np.where(eligible[:, None], m, -np.inf)
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    P = np.zeros(N) if P0 is None else P0.copy()
    start = P.copy()
    left = budget
    for i, k in order:
        if left <= 0 or m[i, k] <= 0:
            break
        if P[i] + 1e-9 < grid[i, k]:      # increments must be taken in order
            continue
        if P[i] >= grid[i, k + 1] - 1e-9:  # increment already taken (warm start)
            continue
        step = min(grid[i, k + 1] - P[i], left)
        P[i] += step
        left -= step
    return P


def optimal_sizes(grid, V, cost_per_mw):
    """Unconstrained optimum: add power while the marginal value exceeds the
    annualised cost of one MW of battery power."""
    dP, m = _concave_increments(grid, V)
    take = m > cost_per_mw
    # increments are taken in order, so stop at the first unprofitable step
    first_bad = np.where((~take).any(axis=1), (~take).argmax(axis=1), take.shape[1])
    P = np.array([dP[i, :first_bad[i]].sum() for i in range(len(first_bad))])
    return P


def allocate_with_floors(grid, V, budget, groups, floors):
    """Budget allocation in which every group g (e.g. a continent) receives at
    least floors[g] MW: each floor is first filled greedily within its group,
    then the remaining budget is allocated greedily across all units."""
    P = np.zeros(grid.shape[0])
    used = 0.0
    for g, f in floors.items():
        f = min(f, budget - used)
        if f <= 0:
            continue
        P = allocate_budget(grid, V, f, P0=P, eligible=(groups == g))
        used = P.sum()
    return allocate_budget(grid, V, budget - used, P0=P)


def allocate_maximin(grid, V, budget, groups, base_cost):
    """Rawlsian (maximin) siting across groups: the next increment of power goes
    to the group whose relative benefit (value captured / its commitment cost
    without storage) is currently the lowest, and within that group to the
    unit with the highest marginal value."""
    dP, m = _concave_increments(grid, V)
    N = grid.shape[0]
    P = np.zeros(N)
    step_idx = np.zeros(N, dtype=int)            # next increment of each unit
    gvals = {g: 0.0 for g in base_cost}
    left = budget
    while left > 1e-9:
        cand = {}
        for g in base_cost:
            idx = np.flatnonzero(groups == g)
            idx = idx[step_idx[idx] < m.shape[1]]
            if len(idx) == 0:
                continue
            mm = m[idx, step_idx[idx]]
            j = int(np.argmax(mm))
            if mm[j] > 0:
                cand[g] = idx[j]
        if not cand:
            break
        g = min(cand, key=lambda k: gvals[k] / max(base_cost[k], 1e-12))
        i = cand[g]
        k = step_idx[i]
        step = min(dP[i, k], left)
        gvals[g] += m[i, k] * step
        P[i] += step
        left -= step
        step_idx[i] += 1
    return P
