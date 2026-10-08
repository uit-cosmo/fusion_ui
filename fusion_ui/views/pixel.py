"""One reference pixel's data, as the pixel level's figures read it off the products.

``pixel_averages`` holds the 2DCA at every reference as ``(ref_y, ref_x, y, x, time)``; one
reference's field is ``bank[name].isel(ref_y=y, ref_x=x)``, a ``(y, x, time)`` array on the shot's
R and Z. ``method_fields`` holds what was read off those averages: tracked positions and the lags
each track's slope rests on.

A pixel can have nothing to show, and that is not an error: it can be dead (the mask), or live with
no events (the average is empty). :func:`problem` says which, in a sentence the page puts where the
figure would have been.

A track is read two ways. :func:`track_positions` is where it put the structure at *one* lag, which
every panel of the lag strip marks. :func:`track_paths` is where it put it at *every* lag of the bank,
the trajectory in the R-Z plane that the strip's zero-lag panel draws.
"""

from dataclasses import dataclass

import numpy as np

from fusion_ui.views.methods import TRACK_BY_KEY, Track
from fusion_ui.views.overlays import neighbour_level

#: The fields of the average, and the tracks that are read off each.
FIELDS = {
    "cond_av": "conditional average",
    "cond_repr": "conditional representativeness",
    "cross_corr": "cross-correlation",
}
FIELD_TRACKS = {"cond_av": ("max", "com"), "cond_repr": (), "cross_corr": ("2dcc",)}


def problem(bundle, need_bank=True):
    """A sentence saying why there is nothing to draw at the pixel in view, or ``None``.

    A dead pixel and a live one with no events are both ordinary, not errors: the page puts this
    sentence where the figure would have been. ``need_bank`` says the figure draws the conditional
    average itself, which only ``pixel_averages`` holds; a figure that reads ``method_fields``
    alone passes ``False`` and the events are read off that.
    """
    if bundle.pixel is None:
        return "No pixel is chosen."
    x, y = bundle.pixel
    ny, nx = bundle.shape
    if not (0 <= y < ny and 0 <= x < nx):
        return f"Pixel (x={x}, y={y}) is outside this array ({nx} x {ny})."
    if bundle.dead[y, x]:
        return f"Pixel (x={x}, y={y}) is dead in this shot's mask: no average was computed there."
    if need_bank and bundle.bank is None:
        return (
            "The conditional averages of this shot (pixel_averages) are not available."
        )
    empty = f"No events at pixel (x={x}, y={y}): its conditional average is empty."
    if bundle.bank is not None:
        events = np.asarray(bundle.bank["nevents"].values)[y, x]
        if (
            events == 0
            or not np.isfinite(
                bundle.bank["cond_av"].isel(ref_y=y, ref_x=x).values
            ).any()
        ):
            return empty
    elif bundle.fields is not None and "nevents" in bundle.fields:
        events = float(bundle.fields["nevents"].values[y, x])
        if events == 0 or not np.isfinite(events):
            return empty
    return None


def lags(bundle):
    """The bank's lag axis, in seconds."""
    return np.asarray(bundle.bank["time"].values, dtype=float)


def reference_field(bundle, name):
    """One reference's ``(y, x, time)`` DataArray of ``name`` at the pixel in view."""
    x, y = bundle.pixel
    return bundle.bank[name].isel(ref_y=y, ref_x=x)


def contour_level(bundle, name, field):
    """The level of ``name``'s contour at the pixel in view, as a fraction of the field's maximum.

    ``method_fields`` stores the level the centroid track used for the conditional average
    (``level_com``), so that field is drawn at exactly the level its stored track was fitted at.
    Anything else gets the same rule applied to its own zero-lag frame.
    """
    if name == "cond_av" and bundle.fields is not None and "level_com" in bundle.fields:
        x, y = bundle.pixel
        stored = float(bundle.fields["level_com"].values[y, x])
        if np.isfinite(stored):
            return stored
    return neighbour_level(field.values, lags(bundle), bundle.neighbour_step)


def track_positions(bundle, keys, index):
    """``[(Track, (r, z)), …]``: where each track put the structure at lag ``index``, NaN if nowhere."""
    if bundle.fields is None:
        return []
    x, y = bundle.pixel
    out = []
    for key in keys:
        r, z = f"pos_r_{key}", f"pos_z_{key}"
        if r in bundle.fields and z in bundle.fields:
            out.append(
                (
                    TRACK_BY_KEY[key],
                    (
                        float(bundle.fields[r].values[y, x, index]),
                        float(bundle.fields[z].values[y, x, index]),
                    ),
                )
            )
    return out


def velocity_line(lag, position, fitted, slope):
    """A stored velocity as a straight line through the mean of the fitted points: ``(lag, line)`` or ``None``.

    ``lag`` and ``position`` are on one lag axis, ``fitted`` marks the lags the slope rests on that
    have a position, and ``slope`` is position per lag unit, in whatever units the caller draws. The
    line runs over the fitted lags and goes through the mean of the points there: the least-squares
    line itself when the estimator is ``lsq`` (the deck's), and the mean slope the velocity stands for
    otherwise. ``None`` when the fit gave no slope or nothing was fitted.

    One definition for both planes: the Tracks view draws R(tau) and Z(tau) with it, and the lag
    strip draws the same two lines as one path in the R-Z plane (:meth:`TrackPath.straight`), so the
    dashed path there is the same fit seen in the other plane.
    """
    if not (np.isfinite(slope) and np.any(fitted)):
        return None
    t = lag[fitted]
    return t, slope * (t - t.mean()) + position[fitted].mean()


@dataclass(frozen=True, eq=False)
class TrackPath:
    """One track's trajectory at the pixel in view: where it put the structure at every lag of the bank.

    ``method_fields`` stores a track's positions as R and Z at each lag, NaN where the track followed
    nothing (the tracker finds the two together, so they are NaN at the same lags), the lags its slope
    rests on, and the velocity that slope is. All in SI units, on the bank's lag axis.
    """

    track: Track
    lag: np.ndarray  # seconds
    r: np.ndarray  # metres; NaN at a lag the track followed nothing
    z: np.ndarray
    fit: np.ndarray  # bool: the lags the slope rests on
    vr: float  # m/s, the stored velocity; NaN when the fit failed
    vz: float

    @property
    def tracked(self):
        """The lags that have a position."""
        return np.isfinite(self.r) & np.isfinite(self.z)

    @property
    def fitted(self):
        """The lags the slope rests on that have a position: the ones highlighted.

        The tracker fills a lag it lost by interpolation before it fits, so a lag in ``fit`` can have
        no position; there are then fewer of these than lags counted in ``nlags_*``.
        """
        return self.tracked & self.fit

    def straight(self):
        """``(lag, r, z)``: the stored velocity as a straight path in the R-Z plane, or ``None``.

        The line of :func:`velocity_line` in R and the line of it in Z, at the same lags: a path along
        ``(v_R, v_Z)`` through the mean of the fitted points, over the span of the fitted lags.
        """
        in_r = velocity_line(self.lag, self.r, self.fitted, self.vr)
        in_z = velocity_line(self.lag, self.z, self.fitted, self.vz)
        if in_r is None or in_z is None:
            return None
        return in_r[0], in_r[1], in_z[1]


def track_paths(bundle, keys):
    """``[TrackPath, …]``: each track's trajectory at the pixel in view, in the order of ``keys``.

    Empty without ``method_fields``. A track the product holds no positions of is left out, as
    :func:`track_positions` leaves it out, and so is one that followed nothing at this pixel (a
    trajectory with no point has nothing to draw) or whose lags are not the bank's. A missing
    ``fit_*`` is no lag fitted, and a missing velocity is NaN.
    """
    if bundle.fields is None:
        return []
    fields = bundle.fields
    x, y = bundle.pixel
    lag = lags(bundle)

    def stored(name):
        return float(fields[name].values[y, x]) if name in fields else float("nan")

    out = []
    for key in keys:
        names = (f"pos_r_{key}", f"pos_z_{key}")
        if not all(name in fields for name in names):
            continue
        r, z = (np.asarray(fields[name].values, dtype=float)[y, x] for name in names)
        if r.shape != lag.shape or z.shape != lag.shape:
            continue
        if not (np.isfinite(r) & np.isfinite(z)).any():
            continue
        fit = (
            np.asarray(fields[f"fit_{key}"].values)[y, x].astype(bool)
            if f"fit_{key}" in fields
            else np.zeros(lag.shape, dtype=bool)
        )
        out.append(
            TrackPath(
                track=TRACK_BY_KEY[key],
                lag=lag,
                r=r,
                z=z,
                fit=fit,
                vr=stored(f"vr_{key}"),
                vz=stored(f"vz_{key}"),
            )
        )
    return out


def colour_range(field):
    """``(low, high)`` of a field over every lag: one scale for all the frames drawn of it.

    A per-frame scale makes an average that decays look as if it never does.
    """
    values = np.asarray(field.values, dtype=float)
    high = float(np.nanmax(values))
    low = float(np.nanmin(values))
    return (0.0 if low >= 0 else low), high
