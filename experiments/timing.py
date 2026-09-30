"""Information available at the time of the day-ahead decision.

The commitment for (UTC) operating day d is made on the morning of day d-1,
before the day-ahead gate closure.  At that time demand is observed only up to
the end of day d-2, and so are the realised errors of every candidate; the
operator's own day-ahead forecast for day d and numerical weather forecasts
initialised by then are available.  Every candidate, margin and estimation
window uses only this information.
"""
import numpy as np

INFO_LAG = 2          # latest complete day of demand known when committing for day d


def trailing_mean(Y, k=7):
    """Mean demand of each unit over days t-k..t-1, shape (N, Dn): a causal
    scale for day t."""
    N, Dn, H = Y.shape
    daily = np.nanmean(Y, axis=2)
    out = np.full((N, Dn), np.nan)
    c = np.nancumsum(np.nan_to_num(daily), axis=1)
    n = np.cumsum(np.isfinite(daily), axis=1)
    for t in range(k, Dn):
        s = c[:, t - 1] - (c[:, t - k - 1] if t - k - 1 >= 0 else 0)
        m = n[:, t - 1] - (n[:, t - k - 1] if t - k - 1 >= 0 else 0)
        out[:, t] = np.where(m > 0, s / np.maximum(m, 1), np.nan)
    return out


# The 00 UTC run of the weather models on day d-1 becomes available at about
# 04-05 UTC.  That is before a 10:00-local gate closure in the Americas, Europe
# and Algeria (09:00 UTC or later), but after it in East Asia and Australia, so
# numerical weather forecasts are used only in the first group.
NWP_CONTINENTS = {"North America", "Europe", "Africa"}


def nwp_mask(continent):
    """Boolean (N,) array: may unit i use the 00 UTC run of day d-1?"""
    return np.array([str(c) in NWP_CONTINENTS for c in continent])
