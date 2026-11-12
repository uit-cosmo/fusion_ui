"""Arrows: one scale for a whole figure, and the key that makes it quantitative.

Plotly has no quiver. An arrow is a line segment from the pixel to ``pixel + v / scale`` and a head
at its tip, rotated to point along the segment (``marker.angle``). ``scale`` is a velocity per unit
length, one for the whole figure: a per-panel scale makes every field look equally fast.
"""

import numpy as np

#: The 90th-percentile speed is drawn this long, as a fraction of the array's R extent.
ARROW_FRACTION = 0.18


def _positive(speeds):
    speeds = np.asarray(speeds, dtype=float).ravel()
    return speeds[np.isfinite(speeds) & (speeds > 0)]


def arrow_scale(speeds, width):
    """Velocity per unit length that draws the 90th-percentile speed ``ARROW_FRACTION`` of ``width`` long.

    ``None`` when there is nothing to scale off: no positive speed, or no width.
    """
    speeds = _positive(speeds)
    if speeds.size == 0 or not np.isfinite(width) or width <= 0:
        return None
    return float(np.percentile(speeds, 90)) / (ARROW_FRACTION * width)


def nice_speed(speeds):
    """A round number near the upper end of ``speeds``, for the key: 1, 2 or 5 times a power of ten."""
    speeds = _positive(speeds)
    if speeds.size == 0:
        return None
    v = float(np.percentile(speeds, 90))
    exponent = np.floor(np.log10(v))
    step = min((1, 2, 5, 10), key=lambda s: abs(s - v / 10**exponent))
    return float(step * 10**exponent)


def quiver(x, y, vr, vz, scale):
    """``dict`` of the arrows' parts for pixels at ``(x, y)`` moving at ``(vr, vz)``, drawn at ``scale``.

    ``shaft_x/shaft_y`` is one NaN-separated line through every arrow, ``tip_x/tip_y`` where the
    heads go, and ``angle`` each head's rotation in degrees clockwise from up, which is right for
    axes at equal aspect with y up.
    """
    x, y, vr, vz = (np.asarray(a, dtype=float) for a in (x, y, vr, vz))
    tip_x = x + vr / scale
    tip_y = y + vz / scale
    shaft_x = np.full(3 * len(x), np.nan)
    shaft_y = np.full(3 * len(x), np.nan)
    shaft_x[0::3], shaft_x[1::3] = x, tip_x
    shaft_y[0::3], shaft_y[1::3] = y, tip_y
    return dict(
        tip_x=tip_x,
        tip_y=tip_y,
        shaft_x=shaft_x,
        shaft_y=shaft_y,
        angle=np.degrees(np.arctan2(tip_x - x, tip_y - y)),
    )
