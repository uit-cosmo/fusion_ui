"""The raw imaging view: frames, click-a-pixel, movie export.

A *live* spec -- there is nothing derived to cache. The time-sliced dataset is
the result, and everything expensive about it (a frame, a pixel's series) is
read one slice at a time from the open file. It is registered anyway so that
the single-shot page needs no special case for it: the page picks a spec and
renders it, and this happens to be the one that draws its own widgets.

Frame index and selected pixel are **view state**, not parameters: they live in
``st.session_state`` keyed off ``Target.key`` and never reach a hash. Minting a
new ``param_sets`` row on every slider drag would fill the table with noise and
tell you nothing.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fusion_ui.core import decimate, loader, params_ui, registry, store


@dataclass
class RawFramesParams:
    """
    colorscale: Plotly colour scale for the frame.
    max_movie_frames: The window is strided down to stay under this frame count.
    movie_fps: Playback rate of the exported mp4.
    """

    colorscale: str = "Plasma"
    max_movie_frames: int = 300
    movie_fps: int = 20


params_ui.CHOICES[("RawFramesParams", "colorscale")] = (
    "Plasma",
    "Viridis",
    "Inferno",
    "Turbo",
    "RdBu",
    "Greys",
)


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def _frame_index(times, target):
    """The slider, plus a jump-to-time box that agrees with it."""
    state_key = f"frame.{target.key}"
    if times.size == 1:
        return 0

    jump_key = f"{state_key}.jump"

    def jump_to_nearest():
        # Fires only on an actual edit (Streamlit's on_change semantics), so it
        # cannot fight the slider on an ordinary drag rerun.
        st.session_state[state_key] = loader.nearest_index(
            times, st.session_state[jump_key]
        )

    st.session_state.setdefault(state_key, times.size // 2)
    slider_col, jump_col = st.columns([4, 1])
    index = slider_col.slider("Frame", 0, times.size - 1, key=state_key)
    jump_col.number_input(
        "Jump to t [s]",
        min_value=float(times[0]),
        max_value=float(times[-1]),
        value=float(times[index]),
        format="%.6f",
        key=jump_key,
        on_change=jump_to_nearest,
    )
    return index


def _axes(ds, values):
    """``(x axis, y axis, x label, y label)`` -- physical R/Z where we have it."""
    r_grid, z_grid = loader.pixel_grid(ds)
    if r_grid is None:
        return (
            np.arange(values.shape[1]),
            np.arange(values.shape[0]),
            "x",
            "y",
        )
    return r_grid[0, :], z_grid[:, 0], "R [cm]", "Z [cm]"


def _selected_pixel(target, shape):
    key = f"pixel.{target.key}"
    iy, ix = st.session_state.get(key, (shape[0] // 2, shape[1] // 2))
    return min(iy, shape[0] - 1), min(ix, shape[1] - 1)


def _frame_figure(values, x_axis, y_axis, labels, pixel, colorscale):
    iy, ix = pixel
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=x_axis,
            y=y_axis,
            colorscale=colorscale,
            colorbar=dict(title="signal"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[x_axis[ix]],
            y=[y_axis[iy]],
            mode="markers",
            marker=dict(color="white", size=12, symbol="x", line=dict(width=2)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis_title=labels[0],
        yaxis_title=labels[1],
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return figure


# ---------------------------------------------------------------------------
# Movie export
# ---------------------------------------------------------------------------


#: Plotly's colour-scale names to matplotlib's. Mostly the lowercase form, but
#: not for the ColorBrewer scales -- "RdBu".lower() is not a matplotlib cmap.
_MPL_CMAP = {
    "Plasma": "plasma",
    "Viridis": "viridis",
    "Inferno": "inferno",
    "Turbo": "turbo",
    "RdBu": "RdBu",
    "Greys": "Greys",
}


def _movie_extent(r_grid, z_grid):
    """``(r0, r1, z0, z1)`` for ``imshow(..., origin='lower')``, or ``None``."""
    if r_grid is None:
        return None
    return (
        float(r_grid[0, 0]),
        float(r_grid[0, -1]),
        float(z_grid[0, 0]),
        float(z_grid[-1, 0]),
    )


def _render_movie(ds, stride, params, out_path):
    """Matplotlib, not Plotly: animation is the one thing the interactive stack
    is worse at, and ``plot_movies.py`` already produces what we want."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    indices = list(range(0, ds.sizes[loader.TIME_DIM], stride))
    times = ds[loader.TIME_DIM].values
    r_grid, z_grid = loader.pixel_grid(ds)

    first = loader.frame(ds, indices[0]).values
    figure, axes = plt.subplots(figsize=(4, 4))
    image = axes.imshow(
        first,
        origin="lower",
        cmap=_MPL_CMAP.get(params.colorscale, "plasma"),
        vmin=float(np.nanmin(first)),
        vmax=float(np.nanmax(first)),
        extent=_movie_extent(r_grid, z_grid),
    )
    title = axes.set_title(f"t = {times[indices[0]]:.5f} s")
    axes.set_xticks([])
    axes.set_yticks([])
    figure.colorbar(image, ax=axes)

    def update(i):
        image.set_data(loader.frame(ds, indices[i]).values)
        title.set_text(f"t = {times[indices[i]]:.5f} s")
        return image, title

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    animation.FuncAnimation(figure, update, frames=len(indices)).save(
        out_path, writer="ffmpeg", fps=params.movie_fps
    )
    plt.close(figure)
    return out_path


def _movie_export(ds, times, params, target):
    with st.expander("Movie export"):
        stride = max(1, math.ceil(times.size / params.max_movie_frames))
        n_frames = math.ceil(times.size / stride)
        st.caption(
            f"{times.size} frames in the window → stride {stride} → {n_frames} "
            f"frames, {n_frames / params.movie_fps:.1f} s of video at "
            f"{params.movie_fps} fps. Max frames and fps are in the sidebar."
        )
        # The movie's own parameters are in its path, so changing the frame rate
        # produces a second file rather than silently overwriting the first.
        digest, _ = params_ui.hash_params("movie", params)
        out_path = store.blob_path("movie", digest, target, suffix=".mp4")
        if os.path.exists(out_path):
            st.video(out_path)
            st.caption("Already rendered for these settings.")
            return
        if st.button("Render movie", key=f"movie.{target.key}"):
            with st.spinner("Rendering…"):
                try:
                    _render_movie(ds, stride, params, out_path)
                except Exception as error:  # noqa: BLE001 - surfaced to the user
                    st.error(f"Movie export failed: {error}", icon="⚠️")
                else:
                    st.video(out_path)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def render(ds, params, target):
    times = loader.cached_frame_times(target.path, target.t_start, target.t_end)
    if times.size == 0:
        st.warning("No samples in this window.", icon="⚠️")
        return None

    index = _frame_index(times, target)
    st.caption(f"t = {times[index]:.6f} s · frame {index + 1} / {times.size}")

    values = loader.frame(ds, index).values
    x_axis, y_axis, x_label, y_label = _axes(ds, values)
    iy, ix = _selected_pixel(target, values.shape)

    event = st.plotly_chart(
        _frame_figure(
            values, x_axis, y_axis, (x_label, y_label), (iy, ix), params.colorscale
        ),
        on_select="rerun",
        selection_mode="points",
        key=f"click.{target.key}",
    )
    clicked = [
        p
        for p in (event.selection["points"] if event else [])
        if p.get("curve_number", 0) == 0
    ]
    if clicked:
        new = (
            int(np.argmin(np.abs(y_axis - clicked[0]["y"]))),
            int(np.argmin(np.abs(x_axis - clicked[0]["x"]))),
        )
        if new != (iy, ix):
            st.session_state[f"pixel.{target.key}"] = new
            st.rerun()

    location = (
        f" (R={x_axis[ix]:.2f}, Z={y_axis[iy]:.2f})" if x_label.startswith("R") else ""
    )
    st.caption(
        f"Selected pixel: y={iy}, x={ix}{location} — click the frame to move it."
    )

    pixel_time, pixel_values = loader.pixel_series(ds, iy, ix)
    x_decimated, y_decimated = decimate.envelope(pixel_time, pixel_values)
    trace = go.Figure(go.Scatter(x=x_decimated, y=y_decimated, mode="lines"))
    trace.update_layout(
        xaxis_title="time [s]",
        yaxis_title="signal",
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(trace, use_container_width=True)
    st.caption(
        f"{pixel_time.size} samples in window, {x_decimated.size} plotted after "
        "min/max-envelope decimation."
    )

    _movie_export(ds, times, params, target)
    return None


SPEC = registry.register(
    registry.PlotSpec(
        key="raw_frames",
        label="Frames and pixel trace",
        diagnostics=("apd", "phantom"),
        params=RawFramesParams,
        render=render,
        description="The frame at a time, the trace at a pixel, and an mp4 of the window.",
    )
)
