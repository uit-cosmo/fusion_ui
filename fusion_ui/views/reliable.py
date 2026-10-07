"""The paper's cut: the pixels ``reliable()`` keeps, the same ones for every method.

``fusion_scripts/decorrelation/apd_check/figures.py`` (``reliable``, line 127) is how the paper decides
which pixels of a velocity field to believe, once, for all the methods together::

    ok = (nlags_com >= MIN_LAGS) & (nevents >= min_events)
    for v in (vr_com, vz_com, vr3_tde, vz3_tde, vr2_tde, vz2_tde):
        ok &= isfinite(v)
    return ok & interior        # everything but the array's border

The Fields page's default cut is per method (``views.methods``: each cut applies to what each estimate
rests on). The user chose, at G2, to keep that and to offer this one beside it as a checkbox, so that a
figure can be held against the paper's: with it on, every panel shows only these pixels, whatever the
panel's own method did there. The page's own thresholds stand in for ``MIN_LAGS`` and ``MIN_EVENTS``; their
defaults, 8 and 200, are the paper's.

The rule is a pure function of a ``method_fields`` dataset, and ``tests/test_views_reliable.py`` holds it
equal to ``reliable()`` itself on a real result. ``reliable()``'s ``ca`` argument, which also asks for
``vr`` of the ``_ca`` group to be finite, is left out: that group is not part of phase 06.
"""

from dataclasses import dataclass

import numpy as np

#: The 2DCA track the paper's rule reads (``figures.TRACK``): the contour's centroid. The maximum sticks
#: to the reference pixel for a few lags where the pulse passes close to its centre.
TRACK = "com"
#: What must be a number besides the track's own lags and the events, in the order the paper lists it, with
#: the method each belongs to, for the reason a pixel is left out.
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
    """``(y, x)`` bool: True away from the array's border, as the paper's ``reliable()`` makes it."""
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


def _count(n, word):
    return f"{n:g} {word}" + ("" if n == 1 else "s")


def rule(fields, shape, min_lags, min_events):
    """The pixels the paper's ``reliable()`` keeps of ``fields``, and why it leaves the others out.

    ``fields`` is a ``method_fields`` dataset (or ``None``, or one that lacks a variable: a variable that
    is not there is read as all-NaN, so nothing is kept, where the paper's code would raise).
    ``shape`` is ``(ny, nx)``. A NaN count of lags or events is below any minimum, as in the paper's
    comparison, and the dead pixels are left out by the velocities they do not have, as there.
    """

    def read(name):
        if fields is None or name not in fields:
            return np.full(shape, np.nan)
        return np.asarray(fields[name].values, dtype=float)

    lags, events = read(f"nlags_{TRACK}"), read("nevents")
    with np.errstate(invalid="ignore"):
        enough_lags, enough_events = lags >= min_lags, events >= min_events
    finite = {name: np.isfinite(read(name)) for name, _ in FINITE}
    inside = interior(shape)
    ok = enough_lags & enough_events & inside
    for isnumber in finite.values():
        ok = ok & isnumber

    reason = np.full(shape, "", dtype=object)
    for y, x in zip(*np.nonzero(~ok)):
        parts = []
        # A centroid with no velocity has no lags to count: the velocity says so, once.
        if np.isfinite(lags[y, x]):
            if not enough_lags[y, x]:
                parts.append(
                    f"2DCA centroid on {_count(lags[y, x], 'lag')} < {min_lags}"
                )
        elif finite["vr_com"][y, x] and finite["vz_com"][y, x]:
            parts.append("2DCA centroid lags unknown")
        if not enough_events[y, x]:
            parts.append(
                f"{_count(events[y, x], 'event')} < {min_events}"
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
