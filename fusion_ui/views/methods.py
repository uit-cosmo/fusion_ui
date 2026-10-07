"""The seven ways a velocity is read off the data, and what each pixel of one of them is.

``method_fields`` stores a field per method under the names ``apd<shot>_velocities.nc`` has always
used, so the paper's code can read a blob as it is. This module is the page's table of them: which
variables make a panel, what colours its arrows, and which view cuts apply to it.

Every pixel of a panel is exactly one of four things, and the four look different on the screen
because they mean different things:

``OK``
    A number the view draws.
``FAILED``
    A live pixel whose estimator returned nothing: no events, a contour that never closed, a fit
    resting on one lag, a correlation below the TDE's minimum. A failed fit is not a dead pixel.
``CUT``
    A number the view cuts leave out (too few lags or events, or an edge pixel). It is still in the
    product; moving the cut brings it back.
``DEAD``
    A pixel the mask marks dead. It was never computed.

The view cuts apply per method, to what each estimate rests on. The minimum of lags applies to the
three 2DCA tracks, which each fit a slope through the lags they tracked. The minimum of events
applies to the methods read off the conditional average: the 2DCA maximum and centroid, and the
TDEs applied to it. The 2DCC is read off the cross-correlation of the whole record and the other
TDEs off the record, so no number of events cuts them. The interior-only cut applies to every panel.

The paper cuts the other way: one rule for every method, ``reliable()`` of ``apd_check/figures.py``
(``views.reliable``). With ``Cuts.paper`` set, every panel leaves out the pixels it leaves out and no
others: a pixel is CUT in each panel where its own method found a number, and is still FAILED in the
panel whose method found none.
"""

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fusion_ui.views import reliable
from fusion_ui.views.reliable import interior


class Status(enum.IntEnum):
    OK = 0
    FAILED = 1
    CUT = 2
    DEAD = 3


@dataclass(frozen=True)
class Track:
    """A 2DCA track: where a structure is at each lag, read off one field of the average."""

    key: str  # the suffix of its variables in method_fields: vr_<key>, pos_r_<key>, ...
    label: str
    field: str  # the field of pixel_averages it is read off
    method: (
        str  # how: "max" (sub-pixel maximum) or "contouring" (centroid of the contour)
    )


#: In the order the panels and the tracks figure list them.
TRACKS = (
    Track("max", "2DCA max", "cond_av", "max"),
    Track("com", "2DCA centroid", "cond_av", "contouring"),
    Track("2dcc", "2DCC", "cross_corr", "max"),
)
TRACK_BY_KEY = {track.key: track for track in TRACKS}


@dataclass(frozen=True)
class Method:
    """One panel of the shot level."""

    key: str
    label: str
    vr: str  # method_fields variable names
    vz: str
    nlags: Optional[str] = None  # the lags a track's slope rests on; None for a TDE
    track: Optional[str] = None  # a Track key, for the 2DCA methods
    colour: str = "events"  # what arrows are coloured by: "events" or "cc"
    #: Whether the estimate is read off the conditional average, and so rests on the events it was
    #: averaged over: what the minimum-events cut applies to.
    uses_events: bool = True


#: "3TDE and 2TDE by velocity_estimation's CA method wait for L9", so the CA here is the TDE
#: applied to the conditional average itself: the same field sliced the other way.
#:
#: The 2DCC track is read off ``cross_corr``, which ``imaging_methods`` computes from the whole
#: record ("Spatiotemporal cross-correlation on full dataset", ``cond_av.py``) and not from the
#: events, so it keeps its lags cut and has no events cut, like the TDEs off the record. Its arrows
#: are still coloured by the pixel's events, the one per-pixel count the product has for it.
METHODS = (
    Method("max", "2DCA max", "vr_max", "vz_max", "nlags_max", track="max"),
    Method("com", "2DCA centroid", "vr_com", "vz_com", "nlags_com", track="com"),
    Method(
        "2dcc",
        "2DCC",
        "vr_2dcc",
        "vz_2dcc",
        "nlags_2dcc",
        track="2dcc",
        uses_events=False,
    ),
    Method("tde3", "3TDE (CC)", "vr3_tde", "vz3_tde", colour="cc", uses_events=False),
    Method("tde2", "2TDE (CC)", "vr2_tde", "vz2_tde", colour="cc", uses_events=False),
    Method("catde3", "3TDE on the CA", "vr3_catde", "vz3_catde"),
    Method("catde2", "2TDE on the CA", "vr2_catde", "vz2_catde"),
)
METHOD_BY_KEY = {method.key: method for method in METHODS}


def _plural(n, word):
    return f"{n:g} {word}" + ("" if n == 1 else "s")


@dataclass(frozen=True)
class Panel:
    """One method over the whole array: the numbers, and what each pixel of it is."""

    method: Method
    status: np.ndarray  # (y, x) of Status
    reason: np.ndarray  # (y, x) object: why a pixel is not drawn; "" for OK
    vr: np.ndarray
    vz: np.ndarray
    nlags: Optional[np.ndarray]
    events: np.ndarray
    cc: Optional[np.ndarray]
    colour: np.ndarray  # what the arrows are coloured by

    @property
    def ok(self):
        return self.status == Status.OK

    @property
    def speed(self):
        return np.hypot(self.vr, self.vz)


def panel(bundle, method):
    """The :class:`Panel` of ``method`` on ``bundle``'s ``method_fields``, with the view cuts applied.

    A variable the product lacks is read as all-NaN, so every pixel of that panel is a failed fit
    rather than the page failing. The cuts are per method, or with ``cuts.paper`` the paper's rule,
    the same for every method (see the module docstring).
    """
    shape = bundle.shape
    nan = np.full(shape, np.nan)

    def read(name):
        values = bundle.field_values(name) if name else None
        return nan if values is None else values

    vr, vz = read(method.vr), read(method.vz)
    nlags = read(method.nlags) if method.nlags else None
    events = read("nevents")
    cc = read("cc_tde") if method.colour == "cc" else None
    dead = bundle.dead
    cuts = bundle.cuts

    finite = np.isfinite(vr) & np.isfinite(vz)
    live = ~dead
    too_few_lags = np.zeros(shape, dtype=bool)
    too_few_events = np.zeros(shape, dtype=bool)
    edge = np.zeros(shape, dtype=bool)
    if cuts.paper:
        kept = reliable.rule(bundle.fields, shape, cuts.min_lags, cuts.min_events)
        left_out = ~kept.ok
    else:
        if nlags is not None:
            with np.errstate(invalid="ignore"):
                too_few_lags = nlags < cuts.min_lags
        if method.uses_events:
            with np.errstate(invalid="ignore"):
                too_few_events = events < cuts.min_events
        if cuts.interior_only:
            edge = ~interior(shape)
        left_out = too_few_lags | too_few_events | edge

    cut = live & finite & left_out
    failed = live & ~finite

    status = np.full(shape, Status.OK, dtype=np.int8)
    status[cut] = Status.CUT
    status[failed] = Status.FAILED
    status[dead] = Status.DEAD

    reason = np.full(shape, "", dtype=object)
    for y, x in zip(*np.nonzero(dead)):
        reason[y, x] = "dead pixel: not computed (mask)"
    for y, x in zip(*np.nonzero(failed)):
        if not events[y, x] or not np.isfinite(events[y, x]):
            reason[y, x] = "no fit: no events at this reference"
        elif nlags is not None and np.isfinite(nlags[y, x]) and nlags[y, x] < 3:
            reason[y, x] = f"no fit: the track rests on {_plural(nlags[y, x], 'lag')}"
        else:
            reason[y, x] = "no fit: the method returned no estimate"
    for y, x in zip(*np.nonzero(cut)):
        if cuts.paper:
            reason[y, x] = kept.reason[y, x]
            continue
        parts = []
        if too_few_lags[y, x]:
            parts.append(f"{_plural(nlags[y, x], 'lag')} < {cuts.min_lags}")
        if too_few_events[y, x]:
            parts.append(f"{_plural(events[y, x], 'event')} < {cuts.min_events}")
        if edge[y, x]:
            parts.append("edge pixel")
        reason[y, x] = "cut: " + ", ".join(parts)

    return Panel(
        method=method,
        status=status,
        reason=reason,
        vr=vr,
        vz=vz,
        nlags=nlags,
        events=events,
        cc=cc,
        colour=cc if method.colour == "cc" else events,
    )


def describe_cut(cuts):
    """Which cut is on, in a sentence: the shot-level figure writes it across its top."""
    if cuts.paper:
        return (
            "Cut: the paper's reliable(), the same pixels in every panel: the 2DCA centroid on at least "
            f"{cuts.min_lags} lags and {cuts.min_events} events, a number from it and from both TDEs, "
            "away from the border"
        )
    text = (
        f"Cut per method: the 2DCA tracks on at least {cuts.min_lags} lags; the 2DCA max and centroid and "
        f"the TDEs on the CA on at least {cuts.min_events} events"
    )
    return text + ("; interior pixels only" if cuts.interior_only else "")
