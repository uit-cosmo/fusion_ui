"""Single shot -- frames, click-a-pixel time series, movie export, probes.

Written directly against Streamlit and Plotly, deliberately before the
``PlotSpec`` registry exists (phase 02): this page and the shot browser are
the two real, different consumers the registry gets designed against, rather
than one imagined one.
"""

import math
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fusion_ui import config, ui
from fusion_ui.core import catalog, decimate, loader, probes

st.set_page_config(page_title="Single shot · Shot Explorer", layout="wide")

IMAGING = {"apd", "phantom"}
PROBES = {"asp", "fsp"}


# ---------------------------------------------------------------------------
# Shot / diagnostic picker -- seeded from the browser's selection, always
# overridable here so this page works standalone too.
# ---------------------------------------------------------------------------


def available_targets(row):
    """``{diagnostic: (has_raw, has_preprocessed)}`` for one shot_table row."""
    targets = {}
    for diagnostic in catalog.DIAGNOSTICS:
        availability = row[diagnostic]
        if availability:
            targets[diagnostic] = ("R" in availability, "P" in availability)
    return targets


def pick_shot_and_target(table):
    selection = st.session_state.get("selection")
    shots = sorted(table["shot"].astype(int).unique())
    if not shots:
        return None

    default_shot = (
        selection["shot"] if selection and selection["shot"] in shots else shots[0]
    )
    shot = st.sidebar.selectbox("Shot", shots, index=shots.index(default_shot))

    row = table.set_index("shot").loc[shot]
    targets = available_targets(row)
    if not targets:
        st.sidebar.warning("No diagnostic files indexed for this shot.", icon="⚠️")
        return None
    diagnostic_names = list(targets)

    default_diagnostic = (
        selection["diagnostic"]
        if selection
        and selection["shot"] == shot
        and selection["diagnostic"] in targets
        else diagnostic_names[0]
    )
    diagnostic = st.sidebar.selectbox(
        "Diagnostic", diagnostic_names, index=diagnostic_names.index(default_diagnostic)
    )

    has_raw, has_preprocessed = targets[diagnostic]
    if has_raw and has_preprocessed:
        preprocessed = (
            st.sidebar.radio("Version", ["Preprocessed", "Raw"], horizontal=True)
            == "Preprocessed"
        )
    else:
        preprocessed = has_preprocessed

    return {
        "machine": str(row["machine"]),
        "shot": int(shot),
        "diagnostic": diagnostic,
        "preprocessed": preprocessed,
    }


def discharge_for_shot(shot):
    path, error = ui.resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(path):
        return None
    return catalog.load_discharges(path).get(shot)


# ---------------------------------------------------------------------------
# APD / phantom: frame view, click-a-pixel, movie export
# ---------------------------------------------------------------------------


def frame_view(target, discharge):
    path = loader.dataset_path(
        target["machine"], target["shot"], target["diagnostic"], target["preprocessed"]
    )
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`", icon="⚠️")
        return
    ds = loader.open_dataset(path)

    t_start, t_end, source = loader.time_window(ds, discharge)
    st.caption(
        f"Window {t_start:.4f}–{t_end:.4f} s — "
        + (
            "from the discharge DB."
            if source == "metadata"
            else "no discharge-DB entry yet, showing a centred 0.2 s default."
        )
    )

    times = loader.cached_frame_times(path, t_start, t_end)
    if times.size == 0:
        st.warning("No samples in this window.", icon="⚠️")
        return
    windowed = loader.sliced(ds, t_start, t_end)

    state_key = (
        f"frame_index_{target['machine']}_{target['shot']}_"
        f"{target['diagnostic']}_{int(target['preprocessed'])}"
    )
    if times.size == 1:
        index = 0
    else:
        jump_key = f"{state_key}_jump"

        def _jump_to_nearest_frame():
            # Only fires on an actual edit (Streamlit's on_change semantics),
            # so it cannot fight the slider on an ordinary drag rerun.
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
            on_change=_jump_to_nearest_frame,
        )
    st.caption(f"t = {times[index]:.6f} s · frame {index + 1} / {times.size}")

    frame_da = loader.frame(windowed, index)
    values = frame_da.values
    r_grid, z_grid = loader.pixel_grid(windowed)
    if r_grid is not None:
        r_axis, z_axis = r_grid[0, :], z_grid[:, 0]
        x_label, y_label = "R [cm]", "Z [cm]"
    else:
        r_axis, z_axis = np.arange(values.shape[1]), np.arange(values.shape[0])
        x_label, y_label = "x", "y"

    pixel_key = f"pixel_{target['machine']}_{target['shot']}_{target['diagnostic']}"
    default_pixel = (values.shape[0] // 2, values.shape[1] // 2)
    iy, ix = st.session_state.get(pixel_key, default_pixel)
    iy = min(iy, values.shape[0] - 1)
    ix = min(ix, values.shape[1] - 1)

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=r_axis,
            y=z_axis,
            colorscale="Plasma",
            colorbar=dict(title="signal"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[r_axis[ix]],
            y=[z_axis[iy]],
            mode="markers",
            marker=dict(color="white", size=12, symbol="x", line=dict(width=2)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    event = st.plotly_chart(
        fig, on_select="rerun", selection_mode="points", key=f"{pixel_key}_click"
    )
    points = event.selection["points"] if event else []
    clicked = [p for p in points if p.get("curve_number", 0) == 0]
    if clicked:
        new_ix = int(np.argmin(np.abs(r_axis - clicked[0]["x"])))
        new_iy = int(np.argmin(np.abs(z_axis - clicked[0]["y"])))
        if (new_iy, new_ix) != (iy, ix):
            st.session_state[pixel_key] = (new_iy, new_ix)
            st.rerun()

    location = (
        f" (R={r_axis[ix]:.2f}, Z={z_axis[iy]:.2f})" if r_grid is not None else ""
    )
    st.caption(
        f"Selected pixel: y={iy}, x={ix}{location} — click the frame to move it."
    )

    pixel_time, pixel_values = loader.pixel_series(windowed, iy, ix)
    x_dec, y_dec = decimate.envelope(pixel_time, pixel_values)
    trace_fig = go.Figure(go.Scatter(x=x_dec, y=y_dec, mode="lines"))
    trace_fig.update_layout(
        xaxis_title="time [s]",
        yaxis_title="signal",
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(trace_fig, use_container_width=True)
    st.caption(
        f"{pixel_time.size} samples in window, "
        f"{x_dec.size} plotted after min/max-envelope decimation."
    )

    movie_export(windowed, times, target)


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


def render_movie(ds, stride, fps, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    n_total = ds.sizes[loader.TIME_DIM]
    indices = list(range(0, n_total, stride))
    times = ds[loader.TIME_DIM].values
    r_grid, z_grid = loader.pixel_grid(ds)
    extent = _movie_extent(r_grid, z_grid)

    first = loader.frame(ds, indices[0]).values
    vmin, vmax = float(np.nanmin(first)), float(np.nanmax(first))

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(
        first, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax, extent=extent
    )
    title = ax.set_title(f"t = {times[indices[0]]:.5f} s")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax)

    def update(i):
        idx = indices[i]
        im.set_data(loader.frame(ds, idx).values)
        title.set_text(f"t = {times[idx]:.5f} s")
        return im, title

    animation_obj = animation.FuncAnimation(fig, update, frames=len(indices))

    out_dir = os.path.join(config.CACHE_DIR, "movies")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_preprocessed" if target["preprocessed"] else ""
    out_path = os.path.join(
        out_dir,
        f"{target['machine']}_{target['shot']}_{target['diagnostic']}{suffix}.mp4",
    )
    animation_obj.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return out_path


def movie_export(windowed, times, target):
    with st.expander("Movie export"):
        max_frames = st.number_input(
            "Max frames",
            min_value=20,
            max_value=2000,
            value=300,
            step=20,
            help="The window is strided down to stay under this frame count.",
        )
        fps = st.number_input("Frames per second", min_value=1, max_value=60, value=20)
        stride = max(1, math.ceil(times.size / max_frames))
        n_frames = math.ceil(times.size / stride)
        st.caption(
            f"{times.size} frames in the window → stride {stride} → "
            f"{n_frames} frames, {n_frames / fps:.1f} s of video at {fps} fps."
        )
        if st.button(
            "Render movie", key=f"render_{target['shot']}_{target['diagnostic']}"
        ):
            with st.spinner("Rendering…"):
                try:
                    out_path = render_movie(windowed, stride, fps, target)
                except Exception as error:  # noqa: BLE001 - surfaced to the user
                    st.error(f"Movie export failed: {error}", icon="⚠️")
                else:
                    st.video(out_path)


# ---------------------------------------------------------------------------
# ASP / FSP: the ragged probe adapter
# ---------------------------------------------------------------------------


def probe_view(target):
    path = loader.dataset_path(
        target["machine"], target["shot"], target["diagnostic"], target["preprocessed"]
    )
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`", icon="⚠️")
        return
    ds = loader.open_dataset(path)

    available = probes.quantities_and_positions(ds)
    if not available:
        st.warning("No probe quantities found in this file.", icon="⚠️")
        return

    left, right = st.columns(2)
    quantity = left.selectbox("Quantity", list(available))
    position = right.selectbox("Probe position", available[quantity])

    trace = probes.load_trace(ds, quantity, position)
    x_dec, y_dec = decimate.envelope(trace.time, trace.value)

    fig = go.Figure(go.Scatter(x=x_dec, y=y_dec, mode="lines"))
    fig.update_layout(
        xaxis_title="time [s]",
        yaxis_title=f"{quantity}_{position}",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{trace.time.size} samples on this position's own time base, "
        f"{x_dec.size} plotted after min/max-envelope decimation."
    )

    if trace.rho is not None:
        with st.expander("Flux coordinate ρ for this position"):
            st.caption(
                "ρ is computed on its own, coarser time base -- not a resampling "
                "of the trace above."
            )
            rho_x, rho_y = decimate.envelope(trace.rho_time, trace.rho)
            rho_fig = go.Figure(go.Scatter(x=rho_x, y=rho_y, mode="lines"))
            rho_fig.update_layout(
                xaxis_title="time [s]",
                yaxis_title="ρ",
                height=280,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(rho_fig, use_container_width=True)

    geometry = probes.probe_geometry(ds)
    if geometry["probe_type"]:
        origin = geometry["probe_origin"]
        st.caption(
            f"{geometry['probe_type']}"
            + (f" · origin {origin}" if origin is not None else "")
        )


# ---------------------------------------------------------------------------


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])

    st.title("Single shot")

    table = ui.cached_shot_table()
    if table.empty:
        st.warning(
            "The shot index is empty — run a rescan on the Shot browser page.",
            icon="⚠️",
        )
        return

    target = pick_shot_and_target(table)
    if target is None:
        return

    if target["diagnostic"] in IMAGING:
        frame_view(target, discharge_for_shot(target["shot"]))
    elif target["diagnostic"] in PROBES:
        probe_view(target)
    else:
        st.error(f"No viewer for diagnostic {target['diagnostic']!r}.", icon="⚠️")


main()
