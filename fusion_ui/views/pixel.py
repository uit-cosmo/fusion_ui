"""One reference pixel's data, as the pixel level's figures read it off the products.

``pixel_averages`` holds the 2DCA at every reference as ``(ref_y, ref_x, y, x, time)``; one
reference's field is ``bank[name].isel(ref_y=y, ref_x=x)``, a ``(y, x, time)`` array on the shot's
R and Z. ``method_fields`` holds what was read off those averages: tracked positions and the lags
each track's slope rests on.

A pixel can have nothing to show, and that is not an error: it can be dead (the mask), or live with
no events (the average is empty). :func:`problem` says which, in a sentence the page puts where the
figure would have been.
"""

import numpy as np

from fusion_ui.views.methods import TRACK_BY_KEY
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


def colour_range(field):
    """``(low, high)`` of a field over every lag: one scale for all the frames drawn of it.

    A per-frame scale makes an average that decays look as if it never does.
    """
    values = np.asarray(field.values, dtype=float)
    high = float(np.nanmax(values))
    low = float(np.nanmin(values))
    return (0.0 if low >= 0 else low), high
