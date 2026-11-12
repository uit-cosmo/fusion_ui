"""The paper's cut: the pixels ``reliable()`` keeps, the same ones for every method.

``imaging_methods.decorrelation.reliable`` is how the paper decides which pixels of a velocity field
to believe, once, for all the methods together::

    ok = (nlags_com >= MIN_LAGS) & (nevents >= MIN_EVENTS)
    for v in (vr_com, vz_com, vr3_tde, vz3_tde, vr2_tde, vz2_tde):
        ok &= isfinite(v)
    return ok & interior        # everything but the array's border

The Fields page's default cut is per method (``views.methods``) and offers this one as a checkbox,
with the page's own thresholds for the two minimums. :func:`rule` is the same rule with the reason
for each pixel it leaves out; ``tests/test_views_reliable.py`` holds it to the library's.
"""

from dataclasses import dataclass

import numpy as np

#: What must be a number besides the centroid's lags and the events, in the order the paper lists
#: it, with the method each belongs to.
FINITE = (
    ("vr_com", "2DCA centroid"),
    ("vz_com", "2DCA centroid"),
    ("vr3_tde", "3TDE"),
    ("vz3_tde", "3TDE"),
    ("vr2_tde", "2TDE"),
    ("vz2_tde", "2TDE"),
)
PREFIX = "cut (paper's rule): "


@dataclass(frozen=True)
class Reliable:
    """What the paper's rule keeps of one ``method_fields``."""

    ok: np.ndarray  # (y, x) bool: the pixels ``reliable()`` keeps
    reason: np.ndarray  # (y, x) object: why a pixel is left out; "" where it is kept


def interior(shape):
    """``(y, x)`` bool: True away from the array's border."""
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


def plural(n, word):
    return f"{n:g} {word}" + ("" if n == 1 else "s")


def rule(fields, shape, min_lags, min_events):
    """The pixels the paper's ``reliable()`` keeps of ``fields``, and why it leaves the others out.

    ``fields`` is a ``method_fields`` dataset or ``None``; a variable that is not there is read as
    all-NaN, so nothing is kept. A NaN count of lags or events is below any minimum, and a dead
    pixel is left out by the velocities it does not have.
    """

    def read(name):
        if fields is None or name not in fields:
            return np.full(shape, np.nan)
        return np.asarray(fields[name].values, dtype=float)

    lags, events = read("nlags_com"), read("nevents")
    with np.errstate(invalid="ignore"):
        enough_lags, enough_events = lags >= min_lags, events >= min_events
    finite = {name: np.isfinite(read(name)) for name, _ in FINITE}
    inside = interior(shape)
    ok = enough_lags & enough_events & inside
    for is_number in finite.values():
        ok &= is_number

    reason = np.full(shape, "", dtype=object)
    for y, x in zip(*np.nonzero(~ok)):
        parts = []
        # A centroid with no velocity has no lags to count: the velocity says so, once.
        if np.isfinite(lags[y, x]):
            if not enough_lags[y, x]:
                parts.append(
                    f"2DCA centroid on {plural(lags[y, x], 'lag')} < {min_lags}"
                )
        elif finite["vr_com"][y, x] and finite["vz_com"][y, x]:
            parts.append("2DCA centroid lags unknown")
        if not enough_events[y, x]:
            parts.append(
                f"{plural(events[y, x], 'event')} < {min_events}"
                if np.isfinite(events[y, x])
                else "no event count"
            )
        for method in dict.fromkeys(label for _, label in FINITE):
            if not all(finite[name][y, x] for name, label in FINITE if label == method):
                parts.append(f"no {method} velocity")
        if not inside[y, x]:
            parts.append("edge pixel")
        reason[y, x] = PREFIX + ", ".join(parts)
    return Reliable(ok=ok, reason=reason)
