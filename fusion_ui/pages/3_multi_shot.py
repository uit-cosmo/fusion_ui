"""Multi shot: one scalar against an axis, across a selection of shots.

Reads the scalar store and the shot table's metadata, collapses per-pixel
scalars to one number per shot, and draws the scatter coloured by confinement
mode. The scalars and their sources are whatever the store holds; a scalar is
named by its label where :mod:`fusion_ui.core.scalar_labels` has one, and by
its stored name otherwise.

A click on a point opens what it came from (:mod:`fusion_ui.jump`): the Fields
page for ``method_fields`` and ``blob_parameters`` when it has settings to open
on, the single-shot page otherwise.
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec
from fusion_ui import config, jump, ui
from fusion_ui.core import (
    decimate,
    multishot,
    params_ui,
    registry,
    scalar_labels,
    store,
)
from fusion_ui.views import products as prod

st.set_page_config(page_title="Multi shot · Shot Explorer", layout="wide")


def source_label(source):
    plot, params_hash, diagnostic, preprocessed = source
    kind = "preprocessed" if preprocessed else "raw"
    return f"{plot} · {diagnostic} ({kind}) · {params_hash[:8]}"


def pick(label, options, key, **kwargs):
    """A sidebar selectbox. A stored choice that is no longer offered falls
    back to the first option."""
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = options[0]
    return st.sidebar.selectbox(label, options, key=key, **kwargs)


def jump_to_single_shot(conn, machine, shot, source):
    """Open the single-shot page on the plot and parameters that made a point.

    The page reads ``selection`` and ``spec.{diagnostic}``. Its parameter
    widgets are seeded from the stored ``params_json`` and the run is marked
    ready, so the plot is drawn from the cache. A source whose plot is not
    registered has no spec to restore: the jump is the shot alone.
    """
    plot, params_hash, diagnostic, preprocessed = source
    st.session_state["selection"] = jump.selection(machine, shot, source)

    if plot in registry.REGISTRY:
        spec = registry.get(plot)
        st.session_state[f"spec.{diagnostic}"] = spec
        text = store.params_json(conn, params_hash)
        if text:
            body = json.loads(text)
            params = params_ui.from_dict(spec.params, body["params"]["values"])
            params_ui.seed_session_state(st.session_state, f"params.{spec.key}", params)
        target_key = f"{machine}_{shot}_{diagnostic}_{'p' if preprocessed else 'r'}"
        st.session_state[f"ready.{spec.key}.{target_key}"] = True

    st.switch_page("pages/2_single_shot.py")


def fields_settings(conn, machine, shot, source):
    """The ``method_fields`` settings the Fields page opens on for a point, or ``None``.

    A ``method_fields`` point names its own, and a ``blob_parameters`` point is
    matched to the settings that go with it
    (:func:`fusion_ui.jump.settings_for`). ``None`` for any other source, and
    for a ``blob_parameters`` run that no settings go with.
    """
    plot, params_hash, _, _ = source
    if not jump.opens_fields(plot):
        return None
    found, _ = prod.specs(registry)
    options = prod.settings(conn, found["method_fields"], machine)
    good = {o.hash for o in options if shot in prod.good_shots(conn, machine, o.hash)}
    return jump.settings_for(plot, params_hash, found, options, good)


def jump_to_point(conn, machine, shot, source, how, pixel):
    """Open what a clicked point came from.

    The Fields page shows the blob parameters of the settings it is on. So a
    ``blob_parameters`` run that no settings go with opens the single-shot
    page, like every other source: only there are the numbers the ones clicked.
    """
    settings = fields_settings(conn, machine, shot, source)
    if settings is None:
        jump_to_single_shot(conn, machine, shot, source)
        return
    # The Fields page reads `fields.open` once.
    st.session_state["selection"] = jump.selection(machine, shot, source)
    st.session_state["fields.open"] = jump.fields_request(shot, settings, how, pixel)
    st.switch_page("pages/5_fields.py")


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])

    st.title("Multi shot")

    conn = ui.get_connection()
    # One machine at a time: shot numbers are not unique across machines, and
    # neither the browser's selection nor the metadata join names the machine.
    machine = config.MACHINE
    frame = store.scalar_frame(conn, machine=machine)
    if frame.empty:
        st.warning(
            f"No results are stored for machine `{machine}` yet. Compute "
            "something on the single-shot page, or run `fusion-ui precompute` to warm "
            "the cache overnight.",
            icon="⚠️",
        )
        return

    # ---- sidebar: what to plot -------------------------------------------
    # Shown as "name — label" where there is a label; the value is the stored name.
    name = pick(
        "Scalar",
        multishot.distinct_names(frame),
        "ms.scalar",
        format_func=scalar_labels.option,
    )
    source = pick(
        "Source",
        multishot.distinct_sources(frame, name),
        "ms.source",
        format_func=source_label,
    )

    shot_level = multishot.is_shot_level(frame, source, name)
    how = "mean"
    if not shot_level:
        how = st.sidebar.selectbox(
            "Aggregate",
            list(multishot.AGGREGATES),
            format_func=lambda key: multishot.AGGREGATES[key],
            key="ms.aggregate",
        )

    pixel = None
    if how == "pixel":
        pairs = multishot.pixel_choices(frame, source, name)
        if not pairs:
            st.sidebar.warning("No pixel-indexed values for this source.")
            return
        x = pick("x", sorted({x for x, _ in pairs}), "ms.pixel.x")
        y = pick("y", sorted({y for px_, y in pairs if px_ == x}), "ms.pixel.y")
        pixel = (x, y)

    x_axis = st.sidebar.selectbox(
        "x-axis",
        list(multishot.X_AXES),
        format_func=lambda key: multishot.X_AXES[key][1],
        key="ms.xaxis",
    )

    # ---- shot selection, carried over from the browser -------------------
    shot_selection = st.session_state.get("shot_selection") or []
    if shot_selection:
        st.sidebar.caption(
            f"{len(shot_selection)} shots carried over from the browser."
        )
        if st.sidebar.button("Clear selection", use_container_width=True):
            st.session_state["shot_selection"] = []
            st.rerun()
    else:
        st.sidebar.caption("No selection carried over — using every stored shot.")

    # ---- the number itself ----------------------------------------------
    aggregated = multishot.aggregate(frame, source, name, how, pixel)
    if shot_selection:
        aggregated = aggregated[aggregated["shot"].isin(set(shot_selection))]

    if aggregated.empty:
        st.info("No stored values fall in this selection.", icon="ℹ️")
        return

    merged = multishot.with_metadata(aggregated, ui.cached_shot_table())
    column, x_label = multishot.X_AXES[x_axis]
    x = (
        merged["shot"].astype(int)
        if x_axis == "shot"
        else pd.to_numeric(merged[column], errors="coerce")
    )
    plot_frame = merged.assign(
        _x=x,
        _mode=merged["mode"].fillna("").replace("", "no metadata"),
    )
    plot_frame = plot_frame[pd.notna(plot_frame["_x"]) & pd.notna(plot_frame["value"])]

    if plot_frame.empty:
        st.info(
            f"No shot in this selection has a value for {name!r} on this axis.",
            icon="ℹ️",
        )
        return

    # A shot-level scalar is one number per shot: there is no collapse to name.
    collapse = None if shot_level else multishot.AGGREGATES[how]
    st.caption(
        f"{name} · {collapse or 'one value per shot'} · "
        f"{len(plot_frame)} shots on `{machine}` · click a point to open that shot"
    )

    figure = px.scatter(
        plot_frame,
        x="_x",
        y="value",
        color="_mode",
        custom_data=["machine", "shot"],
        hover_name="shot",
        hover_data={
            "machine": False,
            "shot": False,
            "f_GW": True,
            "n_e_bar": True,
            "I_p": True,
        },
        labels={
            "_x": x_label,
            "value": scalar_labels.wrapped(scalar_labels.axis_title(name, collapse)),
            "_mode": "confinement mode",
        },
    )
    figure.update_layout(legend=dict(orientation="h", y=-0.3), height=520)

    event = st.plotly_chart(
        figure,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="ms.scatter",
    )
    for point in decimate.selection_points(event):
        clicked = multishot.clicked_shot(point)
        if clicked is not None:
            jump_to_point(conn, *clicked, source, how, pixel)
            break


main()
