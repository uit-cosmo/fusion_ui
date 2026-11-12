"""The raw imaging view: frames, click-a-pixel, movie export.

A live spec: the time-sliced dataset is the result, and a frame or a pixel's
series is read off the open file one slice at a time.

The frame index and the selected pixel are view state, not parameters: they
live in ``st.session_state`` keyed off ``Target.key`` and never reach a hash.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fusion_ui.core import decimate, loader, params_ui, registry, shared, store


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


def _frame_index(times, target):
    """The slider, plus a jump-to-time box that agrees with it."""
    state_key = f"frame.{target.key}"
    if times.size == 1:
        return 0

    jump_key = f"{state_key}.jump"

    def jump_to_nearest():
        # on_change fires only on an edit, so it cannot fight the slider.
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
    """``(x axis, y axis, x label, y label)``: R and Z where the file has them."""
    r_grid, z_grid = loader.pixel_grid(ds)
    if r_grid is None:
        return np.arange(values.shape[1]), np.arange(values.shape[0]), "x", "y"
    return r_grid[0, :], z_grid[:, 0], "R [cm]", "Z [cm]"


def _point_to_pixel(point, x_axis, y_axis, shape):
    """``(iy, ix)`` for one selected point, or ``None`` for one that cannot be
    mapped.

    A marker of the click grid (see :func:`_frame_figure`) carries its own
    ``[iy, ix, value]`` as customdata. Any other point maps to the pixel
    nearest its ``x`` and ``y``.
    """
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple)) and len(custom) >= 2:
        try:
            iy, ix = int(custom[0]), int(custom[1])
        except (TypeError, ValueError):
            pass
        else:
            if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
                return iy, ix
    try:
        ix = int(np.argmin(np.abs(x_axis - point["x"])))
        iy = int(np.argmin(np.abs(y_axis - point["y"])))
    except (KeyError, TypeError, ValueError):
        return None
    return iy, ix


def _frame_figure(values, x_axis, y_axis, labels, pixel, colorscale):
    iy, ix = pixel
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=x_axis,
            y=y_axis,
            colorscale=colorscale,
            colorbar=dict(title="signal"),
            # A heatmap cannot be selected, so a click on it never reaches the
            # server. Skipped from hover so that the click grid below is
            # always the closest hit.
            hoverinfo="skip",
        )
    )
    # The click target: one invisible marker per cell, carrying its own
    # ``[iy, ix, value]``. A click reaches the server only as a selection, and
    # Plotly's select-on-click starts from the hover data, so this trace must
    # not have ``hoverinfo="skip"``.
    ny, nx = values.shape
    grid_x, grid_y, grid_custom = [], [], []
    for row in range(ny):
        for col in range(nx):
            grid_x.append(x_axis[col])
            grid_y.append(y_axis[row])
            grid_custom.append([row, col, values[row, col]])
    figure.add_trace(
        go.Scatter(
            x=grid_x,
            y=grid_y,
            customdata=grid_custom,
            mode="markers",
            marker=dict(opacity=0, size=30),
            showlegend=False,
            hovertemplate=(
                f"{labels[0]}=%{{x:.2f}}<br>{labels[1]}=%{{y:.2f}}"
                "<br>y=%{customdata[0]}, x=%{customdata[1]}"
                "<br>signal=%{customdata[2]:.4g}<extra></extra>"
            ),
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
        # Clicks must select, and must never miss. ``hoverdistance=-1``
        # removes the 20-pixel cutoff, so a click anywhere in the axes finds
        # the nearest grid marker. ``dragmode="pan"``, because Streamlit turns
        # select-on-click off whenever the dragmode is select or lasso.
        clickmode="event+select",
        dragmode="pan",
        hovermode="closest",
        hoverdistance=-1,
    )
    return figure


#: Plotly's colour-scale names to matplotlib's.
_MPL_CMAP = {
    "Plasma": "plasma",
    "Viridis": "viridis",
    "Inferno": "inferno",
    "Turbo": "turbo",
    "RdBu": "RdBu",
    "Greys": "Greys",
}


def _render_movie(ds, stride, params, out_path):
    """The window as an mp4, every ``stride``-th frame, with matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    indices = list(range(0, ds.sizes[loader.TIME_DIM], stride))
    times = ds[loader.TIME_DIM].values
    r_grid, z_grid = loader.pixel_grid(ds)
    extent = None
    if r_grid is not None:
        extent = tuple(map(float, (r_grid[0, 0], r_grid[0, -1], z_grid[0, 0], z_grid[-1, 0])))

    first = loader.frame(ds, indices[0]).values
    figure, axes = plt.subplots(figsize=(4, 4))
    image = axes.imshow(
        first,
        origin="lower",
        cmap=_MPL_CMAP.get(params.colorscale, "plasma"),
        vmin=float(np.nanmin(first)),
        vmax=float(np.nanmax(first)),
        extent=extent,
    )
    title = axes.set_title(f"t = {times[indices[0]]:.5f} s")
    axes.set_xticks([])
    axes.set_yticks([])
    figure.colorbar(image, ax=axes)

    def update(i):
        image.set_data(loader.frame(ds, indices[i]).values)
        title.set_text(f"t = {times[indices[i]]:.5f} s")
        return image, title

    shared.makedirs(os.path.dirname(out_path))
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
        # The parameters are in the path: another frame rate is another file.
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


def render(ds, params, target):
    times = loader.cached_frame_times(target.path, target.t_start, target.t_end)
    if times.size == 0:
        st.warning("No samples in this window.", icon="⚠️")
        return None

    index = _frame_index(times, target)
    st.caption(f"t = {times[index]:.6f} s · frame {index + 1} / {times.size}")

    values = loader.frame(ds, index).values
    ny, nx = values.shape
    x_axis, y_axis, x_label, y_label = _axes(ds, values)
    pixel_key = f"pixel.{target.key}"
    iy, ix = st.session_state.get(pixel_key, (ny // 2, nx // 2))
    iy, ix = min(iy, ny - 1), min(ix, nx - 1)

    event = st.plotly_chart(
        _frame_figure(
            values, x_axis, y_axis, (x_label, y_label), (iy, ix), params.colorscale
        ),
        on_select="rerun",
        selection_mode="points",
        key=f"click.{target.key}",
    )
    for point in decimate.selection_points(event):
        mapped = _point_to_pixel(point, x_axis, y_axis, values.shape)
        if mapped is not None and mapped != (iy, ix):
            st.session_state[pixel_key] = mapped
            st.rerun()

    location = (
        f" (R={x_axis[ix]:.2f}, Z={y_axis[iy]:.2f})" if x_label.startswith("R") else ""
    )
    st.caption(
        f"Selected pixel: y={iy}, x={ix}{location} — click the frame to move it, "
        "or type the indices below."
    )
    # Keyless on purpose: the pixel in session state is the one source of
    # truth, and keyed widgets would keep stale values that fight it.
    x_col, y_col = st.columns(2)
    new_ix = x_col.number_input("Pixel x", min_value=0, max_value=nx - 1, value=int(ix))
    new_iy = y_col.number_input("Pixel y", min_value=0, max_value=ny - 1, value=int(iy))
    if (int(new_iy), int(new_ix)) != (iy, ix):
        st.session_state[pixel_key] = (int(new_iy), int(new_ix))
        st.rerun()

    pixel_time, pixel_values = loader.pixel_series(ds, iy, ix)
    decimate.zoomable_trace(
        pixel_time,
        pixel_values,
        key=f"trace.{target.key}.{iy}.{ix}",
        x_label="time [s]",
        y_label="signal",
        height=280,
    )

    _movie_export(ds, times, params, target)
    return None


def overlay(items, params, target):
    """Every selected pixel's time trace on one axis, each as its min/max
    envelope.

    ``items`` carry the open time-sliced dataset ``render`` gets, which the
    caller may have sliced to a zoom window.
    """
    from plotly.colors import qualitative

    cycle = qualitative.Plotly
    figure = go.Figure()
    for index, ((x, y), ds) in enumerate(items):
        times, values = loader.pixel_series(ds, int(y), int(x))
        env_t, env_v = decimate.envelope(times, values)
        figure.add_trace(
            go.Scatter(
                x=env_t,
                y=env_v,
                mode="lines",
                line=dict(color=cycle[index % len(cycle)]),
                name=f"(x={x}, y={y})",
            )
        )
    figure.update_layout(
        xaxis_title="time [s]",
        yaxis_title="signal",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        title=f"Pixel traces at {len(items)} pixels",
    )
    return figure


SPEC = registry.register(
    registry.PlotSpec(
        key="raw_frames",
        label="Frames and pixel trace",
        diagnostics=("apd", "phantom"),
        params=RawFramesParams,
        render=render,
        overlay=overlay,
        description="The frame at a time, the trace at a pixel, and an mp4 of the window.",
    )
)
