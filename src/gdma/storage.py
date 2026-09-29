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

__all__ = ["battery_dispatch", "value_curves", "allocate_budget", "optimal_sizes"]


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


def allocate_budget(grid, V, budget):
    """Greedy allocation of `budget` MW of battery power across units."""
    dP, m = _concave_increments(grid, V)
    N, K1 = m.shape
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    P = np.zeros(N)
    left = budget
    for i, k in order:
        if left <= 0 or m[i, k] <= 0:
            break
        if P[i] + 1e-9 < grid[i, k]:      # increments must be taken in order
            continue
        step = min(dP[i, k], left)
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
