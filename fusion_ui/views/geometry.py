"""The array as the figures draw it: axes for a heatmap, reading order, arrow scale.

The pixel grid is almost rectilinear: on C-Mod's APD the columns of R differ by a tenth of a
millimetre along a pixel pitch of nearly four. A Plotly heatmap takes one axis per direction, so a
frame is drawn on the mean R of each column and the mean Z of each row, an error of a few percent
of a pixel. Anything placed at a pixel's true position (a marker, an arrow tail, a contour) uses the
2-D ``R`` and ``Z``.
"""

import numpy as np

#: The 90th-percentile speed is drawn this long, as a fraction of the array's R extent
#: (``plotting_scripts.twodca_plots.ARROW_FRACTION``).
ARROW_FRACTION = 0.18


def grid_axes(R, Z):
    """``(x_axis, y_axis)``: one coordinate per column and per row, for a heatmap of this grid.

    The means of ``R`` down each column and of ``Z`` along each row. Falls back to the pixel indices
    when the coordinates are not finite or not monotonic, which a heatmap cannot place.
    """
    x_axis = (
        np.nanmean(R, axis=0)
        if np.isfinite(R).any()
        else np.arange(R.shape[1], dtype=float)
    )
    y_axis = (
        np.nanmean(Z, axis=1)
        if np.isfinite(Z).any()
        else np.arange(Z.shape[0], dtype=float)
    )
    if not _monotonic(x_axis):
        x_axis = np.arange(R.shape[1], dtype=float)
    if not _monotonic(y_axis):
        y_axis = np.arange(Z.shape[0], dtype=float)
    return x_axis, y_axis


def _monotonic(values):
    steps = np.diff(values)
    return bool(
        np.all(np.isfinite(values)) and (np.all(steps > 0) or np.all(steps < 0))
    )


def pitch(R, Z):
    """The mean distance between neighbouring pixels, in the grid's units."""
    steps = []
    if R.shape[1] > 1:
        steps.append(np.nanmean(np.hypot(np.diff(R, axis=1), np.diff(Z, axis=1))))
    if R.shape[0] > 1:
        steps.append(np.nanmean(np.hypot(np.diff(R, axis=0), np.diff(Z, axis=0))))
    steps = [s for s in steps if np.isfinite(s) and s > 0]
    return float(np.mean(steps)) if steps else 1.0


def reading_order(R, Z):
    """Every pixel as ``(x, y)`` in the order it is read off the drawn array.

    Top row first, left to right: Z up and R to the right, as the figures draw it. A row is a row
    of the array (``y``); it is the top one when its Z is the largest, whichever way ``y`` runs.
    """
    ny, nx = R.shape
    z_by_row = (
        np.nanmean(Z, axis=1) if np.isfinite(Z).any() else np.arange(ny, dtype=float)
    )
    r_by_column = (
        np.nanmean(R, axis=0) if np.isfinite(R).any() else np.arange(nx, dtype=float)
    )
    rows = sorted(range(ny), key=lambda y: -z_by_row[y])
    columns = sorted(range(nx), key=lambda x: r_by_column[x])
    return [(x, y) for y in rows for x in columns]


def step_pixel(order, dead, pixel, direction):
    """The next live pixel after ``pixel`` in ``order`` (``direction`` +1) or the one before (-1).

    Dead pixels are skipped and the walk wraps round. ``pixel`` itself may be dead, or not in
    ``order`` at all, in which case the walk starts from the nearest end. ``pixel`` comes back
    unchanged when no pixel is live.
    """
    live = [(x, y) for x, y in order if not dead[y, x]]
    if not live:
        return pixel
    if pixel in order:
        here = order.index(pixel)
        count = len(order)
        for offset in range(1, count + 1):
            candidate = order[(here + direction * offset) % count]
            if not dead[candidate[1], candidate[0]]:
                return candidate
        return pixel
    return live[0] if direction > 0 else live[-1]


def _customdata(point):
    for read in (
        lambda: point["customdata"],
        lambda: point["customData"],
        lambda: point.customdata,
    ):
        try:
            value = read()
        except (TypeError, KeyError, IndexError, AttributeError):
            continue
        if value is not None:
            return value
    return None


def _pair(custom):
    """``(x, y)`` from one point's ``customdata``, or ``None`` when it does not hold a pixel.

    Usually a list. A typed array that went through JSON arrives as an object keyed "0", "1", …
    and is read as the list it was, so a figure that forgot to send plain lists still works.
    """
    if isinstance(custom, dict):
        custom = [custom.get(str(i), custom.get(i)) for i in range(2)]
    try:
        return int(custom[0]), int(custom[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def pixel_from_event(event):
    """The ``(x, y)`` a ``st.plotly_chart`` selection event names, or ``None``.

    Each marker carries its ``[x, y]`` as ``customdata``, as the pixel map of ``core.multipixel``
    does. With several points selected the first, in reading of ``(x, y)``, is taken. Anything that
    is not a pixel is skipped, never an error in the page.
    """
    from fusion_ui.core import decimate

    pairs = {_pair(_customdata(point)) for point in decimate.selection_points(event)}
    pairs.discard(None)
    return min(pairs) if pairs else None
