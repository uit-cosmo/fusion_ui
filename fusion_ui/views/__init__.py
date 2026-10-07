"""Pure builders for the Fields page, and the two lists the page is made from.

``fusion_ui/pages/5_fields.py`` draws nothing itself. It loads the three products of a shot into a
:class:`~fusion_ui.views.bundle.Bundle`, then walks :data:`SHOT_VIEWS` (what is drawn for the whole
array) and :data:`PIXEL_VIEWS` (what is drawn once a pixel is chosen). Each entry is a
:class:`~fusion_ui.views.base.View`: the products it reads, the controls it asks for and a pure
builder.

**To add a figure**, write a builder ``build(bundle, **controls) -> Figure | DataFrame | str |
list`` in this package, with a test, and add one entry to the list it belongs in. The page does not
change. A figure that reads a new per-pixel product (one chained to ``pixel_averages``) adds its key
to ``products.KEYS`` and names it in ``reads``; the page then offers the command that fills it until
it is computed.

No builder calls Streamlit, the store, the database or the filesystem, so every one is a function of
loaded datasets and can be tested on a synthetic blob. ``products`` is the one module that reads, and
only the ledger: it never writes.
"""

import numpy as np

from fusion_ui.views import frame, lag_strip, numbers, panels, pixel, tracks
from fusion_ui.views.base import Control, View
from fusion_ui.views.bundle import Bundle, Cuts

__all__ = ["Bundle", "Control", "Cuts", "PIXEL_VIEWS", "SHOT_VIEWS", "View"]


def _panel_controls(bundle):
    return (
        Control(
            "mode",
            "Show",
            "radio",
            default="arrows",
            options=tuple(panels.MODES),
            labels=panels.MODES,
            help=(
                "Arrows, or the v_R or v_Z of every method as a map on one diverging colour scale. "
                "Either way nothing is rescaled between the panels."
            ),
        ),
        Control(
            "arrow_gain",
            "Arrow length ×",
            "slider",
            default=1.0,
            bounds=(0.25, 4.0, 0.25),
            help=(
                "Arrows only. Every panel shares one scale, set by the speeds of the three 2DCA tracks, "
                "so a method that reads faster (a TDE) draws longer arrows, and those that leave the "
                "view are cut off at its edge. The other panels set the scale only if no 2DCA track "
                "drew anything. This makes every arrow in every panel that many times longer or "
                "shorter; the key moves with it."
            ),
        ),
    )


def _lag_limits(bundle):
    """``(step, window)`` of the bank's lags, in microseconds."""
    time = pixel.lags(bundle)
    step = float(np.median(np.diff(time))) * 1e6 if time.size > 1 else 1.0
    # Rounded: the axis is seconds times a million, and a slider's bound should read -15, not -14.999999999999998.
    return round(step, 6), round(float(np.max(np.abs(time))) * 1e6, 6)


def _strip_controls(bundle):
    step, window = _lag_limits(bundle)
    span, source = lag_strip.default_span_us(bundle)
    origin = (
        "the lags the centroid track was fitted on, widened"
        if source == "fit"
        else "the deck's lags"
    )
    return (
        Control(
            "n",
            "Panels",
            "slider",
            default=lag_strip.DEFAULT_PANELS,
            bounds=(2, lag_strip.MAX_PANELS, 1),
        ),
        Control(
            "span_us",
            "Span ± [µs]",
            "number",
            default=None,
            bounds=(step, window, step),
            placeholder=f"auto: ±{span:g} µs",
            help=(
                f"Empty: {origin}. The largest lag shown, either side of zero; it stops at the bank's "
                f"window of ±{window:g} µs. A shot that needs a longer one has its own settings."
            ),
        ),
        Control(
            "lags",
            "Lags [µs]",
            "text",
            default="",
            placeholder="or type them: -6, -3, 0, 3, 6",
            help="Lags typed in override the span and the number of panels.",
        ),
    )


def _frame_controls(bundle):
    step, window = _lag_limits(bundle)
    return (
        Control(
            "name",
            "Field",
            "radio",
            default="cond_av",
            options=tuple(pixel.FIELDS),
            labels=pixel.FIELDS,
        ),
        Control(
            "lag", "Lag [µs]", "slider", default=0.0, bounds=(-window, window, step)
        ),
    )


def _frame_build(bundle, name="cond_av", lag=0.0):
    return [
        frame.frame_figure(bundle, name, lag),
        frame.trace_figure(bundle, name, lag),
    ]


SHOT_VIEWS = [
    View(
        key="velocity_panels",
        title="Velocity fields",
        reads=("method_fields",),
        build=panels.velocity_panels,
        controls=_panel_controls,
        selectable=True,
        caption=(
            "One panel per method, on one scale. Click a pixel to open it below; hover for its numbers. "
            "Dead pixels (grey squares) were never computed; a red cross is a live pixel whose method "
            "found no fit; a hollow circle is cut by the settings in the sidebar."
        ),
    ),
]

PIXEL_VIEWS = [
    View(
        key="lag_strip",
        title="The average at a few lags",
        reads=("pixel_averages",),
        optional=("method_fields",),
        build=lag_strip.lag_strip,
        controls=_strip_controls,
        caption=(
            "Each row keeps one colour scale across its lags. The dashed contour is drawn at the level "
            "the average implies, one level for every lag, so it shrinks and vanishes as the average decays."
        ),
    ),
    View(
        key="frame",
        title="One frame",
        reads=("pixel_averages",),
        optional=("method_fields",),
        build=_frame_build,
        controls=_frame_controls,
    ),
    View(
        key="tracks",
        title="Tracks",
        reads=("method_fields",),
        build=tracks.tracks_figure,
        caption=(
            "Where each track put the structure at each lag, as a displacement from the reference pixel. "
            "Highlighted: the lags the slope rests on that have a position (the tracker fills a lag it lost "
            "by interpolating, and that lag counts in the fit but has no position to show, so fewer points "
            "can be highlighted than the lags counted); dashed: the stored velocity as a line; "
            "dash-dot: the straight lines the two TDE velocities imply."
        ),
    ),
    View(
        key="numbers",
        title="Numbers",
        reads=("method_fields",),
        optional=("blob_parameters",),
        build=numbers.numbers,
    ),
]
