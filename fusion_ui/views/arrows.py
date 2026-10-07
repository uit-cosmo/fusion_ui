"""Arrows: one scale for a whole figure, and the key that makes it quantitative.

Plotly has no quiver. An arrow is a line segment from the pixel to ``pixel + v / scale`` and a head
at its tip, rotated to point along the segment (``marker.angle``). ``scale`` is a velocity per unit
length, taken from **all the panels together** so that two methods compare by eye: a per-panel scale
makes every field look equally fast.

Ported from ``plotting_scripts.twodca_plots`` (``auto_quiver_scale``, ``nice_speed``), which are
drawn per figure.
"""

import numpy as np

from fusion_ui.views.geometry import ARROW_FRACTION


def _positive(speeds):
    speeds = np.asarray(speeds, dtype=float).ravel()
    return speeds[np.isfinite(speeds) & (speeds > 0)]


def arrow_scale(speeds, width, fraction=ARROW_FRACTION):
    """Velocity per unit length that draws the 90th-percentile speed ``fraction`` of ``width`` long.

    ``None`` when there is nothing to scale off: no positive speed, or no width.
    """
    speeds = _positive(speeds)
    if speeds.size == 0 or not np.isfinite(width) or width <= 0:
        return None
    return float(np.percentile(speeds, 90)) / (fraction * width)


def nice_speed(speeds):
    """A round number near the upper end of ``speeds``, for the key: 1, 2 or 5 times a power of ten."""
    speeds = _positive(speeds)
    if speeds.size == 0:
        return None
    v = float(np.percentile(speeds, 90))
    exponent = np.floor(np.log10(v))
    step = min((1, 2, 5, 10), key=lambda s: abs(s - v / 10**exponent))
    return float(step * 10**exponent)


def shafts(x0, y0, x1, y1):
    """``(xs, ys)`` of one line trace drawing every segment ``(x0, y0) -> (x1, y1)``, gaps between them."""
    n = len(x0)
    xs = np.full(3 * n, np.nan)
    ys = np.full(3 * n, np.nan)
    xs[0::3], xs[1::3] = x0, x1
    ys[0::3], ys[1::3] = y0, y1
    return xs, ys


def head_angles(dx, dy):
    """Marker angles, in degrees clockwise from up, for arrows pointing along ``(dx, dy)``.

    Right for axes drawn at equal aspect with y up, which is how every panel here is drawn.
    """
    return np.degrees(np.arctan2(dx, dy))


def quiver(x, y, vr, vz, scale):
    """``dict`` of the arrows' parts for pixels at ``(x, y)`` moving at ``(vr, vz)``, drawn at ``scale``.

    ``tail_x/tail_y`` and ``tip_x/tip_y`` are the ends of each arrow, ``shaft_x/shaft_y`` the single
    NaN-separated line through all of them, ``angle`` the head's rotation.
    """
    x, y, vr, vz = (np.asarray(a, dtype=float) for a in (x, y, vr, vz))
    tip_x = x + vr / scale
    tip_y = y + vz / scale
    shaft_x, shaft_y = shafts(x, y, tip_x, tip_y)
    return dict(
        tail_x=x,
        tail_y=y,
        tip_x=tip_x,
        tip_y=tip_y,
        shaft_x=shaft_x,
        shaft_y=shaft_y,
        angle=head_angles(tip_x - x, tip_y - y),
    )
