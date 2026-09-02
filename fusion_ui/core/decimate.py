"""Min/max envelope downsampling for 1D traces headed to Plotly.

Plotly dies above roughly 50k points, so every 1D trace goes through here on
its way to a chart -- the single most important performance detail in the
project. Naive striding (``y[::n]``) throws away spikes; bucketing the trace
and keeping both the min and the max of each bucket does not.
"""

import numpy as np


def envelope(x, y, max_points=4000):
    """``(x, y)`` reduced to at most ``max_points`` samples, spikes intact.

    The trace is split into ``max_points // 2`` buckets; each contributes the
    sample at its minimum and the sample at its maximum, in time order. A
    bucket that is all-NaN contributes nothing rather than a spurious point.
    Below ``max_points`` samples, the input is returned unchanged.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = y.shape[0]
    if n <= max_points or max_points < 2:
        return x, y

    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)

    out_idx = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        chunk = y[start:stop]
        if np.all(np.isnan(chunk)):
            continue
        lo = start + int(np.nanargmin(chunk))
        hi = start + int(np.nanargmax(chunk))
        out_idx.append(lo)
        out_idx.append(hi)

    if not out_idx:
        return x[:0], y[:0]

    order = np.unique(out_idx)  # sorted, and de-duplicates lo == hi
    return x[order], y[order]
