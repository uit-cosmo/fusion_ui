"""Two-dimensional conditional averaging: the shape of a blob at a pixel.

Ported from ``imaging_methods.find_events_and_2dca`` as ``density_scan`` drives
it. Events are the times the reference pixel exceeds a threshold; the result is
the average frame sequence around those peaks, plus the conditional
representativeness and the two-dimensional cross-correlation over the same
lags.

This is the **base of the phase-03 chain**. On a real preprocessed APD shot it
costs about half a minute, and almost every blob quantity the group reports --
velocities, sizes, areas, tilt -- is derived from the conditional average it
produces rather than from the raw frames. So it is a spec of its own and the
derived plots declare ``requires="two_dca"``: the average is computed once per
(shot, pixel, parameters) and read from its blob by everything downstream.
"""

from dataclasses import dataclass, field

import imaging_methods as im
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from imaging_methods.method_parameters import TwoDcaParams

from fusion_ui.core import loader, registry

#: Which stored field the viewer can show, and what it means.
FIELDS = {
    "cond_av": "conditional average",
    "cond_repr": "conditional representativeness",
    "cross_corr": "cross-correlation",
}


def _apd_two_dca_defaults():
    # threshold = 2 rather than TwoDcaParams' own 2.5: this is what the APD
    # analyses run, and matching it is what makes a fresh number and a seeded
    # one comparable.
    return im.get_default_apd_method_params().two_dca


@dataclass
class TwoDcaSpecParams:
    two_dca: TwoDcaParams = field(default_factory=_apd_two_dca_defaults)


def choices(ds, path, chosen):
    """Pixel indices, bounded by the array that was actually opened."""
    if path == "two_dca.refx":
        return tuple(range(ds.sizes["x"]))
    if path == "two_dca.refy":
        return tuple(range(ds.sizes["y"]))
    return None


def compute(ds, params):
    """The conditional average around every event at the reference pixel."""
    events, average = im.find_events_and_2dca(ds, params.two_dca, verbose=False)
    if not events or not average.data_vars:
        # An empty Dataset is what upstream returns when nothing survived the
        # cuts. Raising here turns it into a recorded failure carrying the
        # reason, rather than a blob that every downstream spec has to re-check.
        raise ValueError(
            "no events survived at pixel "
            f"(x={params.two_dca.refx}, y={params.two_dca.refy}): "
            f"nothing exceeded {params.two_dca.threshold} standard deviations, "
            "or every peak failed the check_max / window cuts. Lower the "
            "threshold or pick another pixel."
        )
    # The per-event windows are deliberately dropped: they are the whole record
    # again, and nothing downstream reads them.
    return average


# ---------------------------------------------------------------------------
# The view. A conditional average is a short movie, so this one draws rather
# than returning a figure -- the same latitude the frame viewer uses, here for
# a cached spec.
# ---------------------------------------------------------------------------


def _axes(average, values):
    r_grid, z_grid = loader.pixel_grid(average)
    if r_grid is None:
        return np.arange(values.shape[1]), np.arange(values.shape[0]), "x", "y"
    return r_grid[0, :], z_grid[:, 0], "R [cm]", "Z [cm]"


def render(result, params, target):
    refx, refy = int(result["refx"]), int(result["refy"])
    times = result["time"].values

    st.caption(
        f"{int(result['number_events'])} events at pixel (x={refx}, y={refy}) "
        f"· window {times.size} frames, {times[0] * 1e6:.1f} to "
        f"{times[-1] * 1e6:.1f} µs around the peak"
    )

    field_key = f"two_dca.field.{target.key}"
    frame_key = f"two_dca.frame.{target.key}"
    st.session_state.setdefault(frame_key, times.size // 2)

    picker, slider = st.columns([1, 3])
    name = picker.selectbox(
        "Field", list(FIELDS), format_func=lambda n: FIELDS[n], key=field_key
    )
    index = slider.slider(
        "Lag",
        0,
        times.size - 1,
        key=frame_key,
        format="%d",
        help="Frames from the peak",
    )

    values = result[name].isel(time=index).values
    x_axis, y_axis, x_label, y_label = _axes(result, values)

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=x_axis,
            y=y_axis,
            colorscale="Plasma",
            # Fixed across the sweep: a per-frame rescale makes a decaying
            # average look like it never decays.
            zmin=float(result[name].min()),
            zmax=float(result[name].max()),
            colorbar=dict(title=name),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[x_axis[refx]],
            y=[y_axis[refy]],
            mode="markers",
            marker=dict(color="white", size=12, symbol="x", line=dict(width=2)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"{FIELDS[name]} at lag {times[index] * 1e6:+.2f} µs",
    )
    st.plotly_chart(figure, use_container_width=True)

    trace = result[name].isel(x=refx, y=refy).values
    line = go.Figure(go.Scatter(x=times * 1e6, y=trace, mode="lines"))
    line.add_vline(x=times[index] * 1e6, line_dash="dot", line_color="grey")
    line.update_layout(
        xaxis_title="lag [µs]",
        yaxis_title=f"{name} at the reference pixel",
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(line, use_container_width=True)
    return None


def scalars(result):
    x, y = int(result["refx"]), int(result["refy"])
    return {(x, y, "number_events"): float(result["number_events"])}


SPEC = registry.register(
    registry.PlotSpec(
        key="two_dca",
        label="Conditional average (2DCA)",
        diagnostics=("apd", "phantom"),
        params=TwoDcaSpecParams,
        render=render,
        compute=compute,
        scalars=scalars,
        choices=choices,
        description=(
            "Average frame sequence around every threshold crossing at the "
            "reference pixel. Emits number_events, and is the input every "
            "derived blob quantity is built on."
        ),
    )
)
