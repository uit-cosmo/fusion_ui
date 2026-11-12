"""Statistics over a basket of traces: time trace, PDF, PSD, ACF and CCF.

Pixels of any imaging shot and channels of any probe shot, mixed freely, are
drawn together under one statistic and labelled with magnetic coordinates.
Statistics are live: widgets take effect on the next rerun, there is no Compute
button, and nothing is written to the run ledger.
"""

import dataclasses
import os

import pandas as pd
import streamlit as st

import fusion_ui.stats  # noqa: F401 - importing the package registers every spec
from fusion_ui import ui
from fusion_ui.core import loader, multipixel, params_ui, registry, statistics, traces

st.set_page_config(page_title="Statistics · Shot Explorer", layout="wide")

#: Warn past this many samples across the basket: about 160 MB, roughly 32
#: full APD windows.
SAMPLE_BUDGET = 20_000_000

#: The label styles of :func:`fusion_ui.core.traces.label`, as the sidebar names them.
LABEL_STYLES = {
    "magnetic": "magnetic (R-R_sep / ρ)",
    "channel": "channel",
    "rz": "R, Z",
}


@st.cache_data(show_spinner=False)
def _cached_trace(path, mtime, ref, t_start, t_end):
    """One extracted trace. ``mtime`` is a cache key only, so that a file
    copied again is read again."""
    return traces.extract(loader.open_dataset(path), ref, t_start, t_end)


def with_traces(table):
    """``table`` cut to the shots that have a file a trace can come from.

    A shot with only a Thomson scattering file has nothing to add here.
    """
    if table.empty:
        return table
    return table[(table[list(traces.SOURCES)] != "").any(axis=1)]


def shots_of(table, machine):
    return sorted(table[table["machine"].astype(str) == machine]["shot"].astype(int))


def targets_of(table, machine, shot):
    """``{diagnostic: (has_raw, has_preprocessed)}`` over the files of one shot
    that a trace can come from."""
    rows = table[table["machine"].astype(str) == machine]
    row = rows[rows["shot"].astype(int) == int(shot)].iloc[0]
    return ui.available_targets(row, traces.SOURCES)


def default_picker(table):
    """``(machine, shot)``: the browser's selection when it is on the table,
    the first of each otherwise."""
    selection = st.session_state.get("selection") or {}
    machines = sorted(table["machine"].astype(str).unique())
    machine = selection.get("machine")
    if machine not in machines:
        machine, selection = machines[0], {}
    shots = shots_of(table, machine)
    shot = selection.get("shot")
    return machine, int(shot if shot in shots else shots[0])


def ensure_window(table):
    """Seed the time window once, from the discharge window of the shot the
    picker opens on."""
    if "stats.t_start" in st.session_state and "stats.t_end" in st.session_state:
        return
    machine, shot = default_picker(table)
    targets = targets_of(table, machine, shot)
    diagnostic = next(iter(targets))
    try:
        path = loader.dataset_path(machine, shot, diagnostic, targets[diagnostic][1])
        t_start, t_end, _ = loader.time_window(
            loader.open_dataset(path), ui.discharge(shot)
        )
    except (OSError, KeyError, ValueError):
        return
    st.session_state.setdefault("stats.t_start", float(t_start))
    st.session_state.setdefault("stats.t_end", float(t_end))


def add_picker(table):
    """The "Add traces" expander: shot, diagnostic, pixels or channels."""
    basket = st.session_state["stats.basket"]
    with st.expander("Add traces", expanded=not basket):
        if table.empty:
            st.warning(
                "No shot with traces (imaging pixels or probe channels) is indexed. A Thomson scattering file holds "
                "profiles, not traces: the Single shot page draws it. If files were added, run a rescan on the Shot "
                "browser page.",
                icon="⚠️",
            )
            return
        machine, shot = default_picker(table)
        machines = sorted(table["machine"].astype(str).unique())
        if len(machines) > 1:
            machine = st.selectbox("Machine", machines, index=machines.index(machine))
        shots = shots_of(table, machine)
        shot = st.selectbox(
            "Shot", shots, index=shots.index(shot) if shot in shots else 0
        )
        targets = targets_of(table, machine, shot)
        diagnostic = st.radio("Diagnostic", list(targets), horizontal=True)
        has_raw, preprocessed = targets[diagnostic]
        if has_raw and preprocessed:
            preprocessed = (
                st.radio("Version", ["Preprocessed", "Raw"], horizontal=True)
                == "Preprocessed"
            )

        path = loader.dataset_path(machine, int(shot), diagnostic, preprocessed)
        if not os.path.exists(path):
            st.error(f"File not found: `{path}`", icon="⚠️")
            return
        ds = loader.open_dataset(path)

        if diagnostic in traces.IMAGING:
            # The selector reads only the target's key.
            target = registry.Target(
                machine, int(shot), diagnostic, preprocessed, path,
                float("nan"), float("nan"), "none",
            )
            picked = [("pixel", x, y) for x, y in multipixel.selector(ds, target)]
            noun = "pixels"
        else:
            names = {traces.channel_text(c): c for c in traces.channels(ds, diagnostic)}
            chosen = st.multiselect("Channels", list(names), key="stats.channels")
            picked = [names[name] for name in chosen]
            noun = "channels"
        if st.button(
            f"Add {len(picked)} selected {noun}",
            key=f"stats.add_{noun}",
            disabled=not picked,
        ):
            for channel in picked:
                ref = traces.TraceRef(machine, int(shot), diagnostic, preprocessed, channel)
                if ref not in basket:
                    basket.append(ref)
            st.session_state["stats.basket"] = basket
            st.rerun()


def basket_frame(basket, t_start, t_end, style):
    """``(table, traces, refs left out)``: each trace of the basket extracted
    through the cache and labelled. A trace that cannot be read, or has no
    sample in the window, is left out."""
    rows, extracted, missed = [], [], []
    for ref in basket:
        try:
            path = loader.dataset_path(
                ref.machine, ref.shot, ref.diagnostic, ref.preprocessed
            )
            trace = _cached_trace(path, os.path.getmtime(path), ref, t_start, t_end)
        except Exception:  # noqa: BLE001 - one bad trace must not fail the page
            trace = None
        if trace is None:
            missed.append(ref)
            continue
        trace = dataclasses.replace(trace, label=traces.label(trace, style))
        extracted.append(trace)
        rows.append(
            {
                "shot": ref.shot,
                "diagnostic": ref.diagnostic,
                "channel": traces.channel_text(ref.channel),
                "label": trace.label,
                "magnetic": traces.magnetic_text(trace),
                "samples": len(trace.time),
            }
        )
    return pd.DataFrame(rows), extracted, missed


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])
    st.session_state.setdefault("stats.basket", [])
    st.session_state.setdefault("stats.reference", 0)

    st.title("Statistics")

    table = with_traces(ui.cached_shot_table())

    # ---- sidebar ---------------------------------------------------------
    spec = statistics.get(
        st.sidebar.selectbox(
            "Statistic",
            list(statistics.REGISTRY),
            format_func=lambda key: statistics.get(key).label,
            key="stats.spec",
        )
    )
    if spec.description:
        st.sidebar.caption(spec.description)

    params = params_ui.form(
        spec.params, f"stats.params.{spec.key}", container=st.sidebar
    )

    if not table.empty:
        ensure_window(table)
    t_start = float(
        st.sidebar.number_input("Start time [s]", format="%g", key="stats.t_start")
    )
    t_end = float(
        st.sidebar.number_input("End time [s]", format="%g", key="stats.t_end")
    )
    if st.sidebar.button("Reset to the discharge window", key="stats.reset_window"):
        st.session_state.pop("stats.t_start", None)
        st.session_state.pop("stats.t_end", None)
        st.rerun()
    style = st.sidebar.radio(
        "Labels",
        list(LABEL_STYLES),
        format_func=LABEL_STYLES.get,
        key="stats.labels",
    )

    # ---- main ------------------------------------------------------------
    add_picker(table)

    basket = st.session_state["stats.basket"]
    if not basket:
        st.info(
            "The basket is empty — add pixel traces or probe channels above, "
            "then every trace is drawn together under one statistic.",
            icon="ℹ️",
        )
        return

    frame, extracted, missed = basket_frame(basket, t_start, t_end, style)
    if missed:
        st.warning(
            f"Left out of the window ({t_start:.4f}–{t_end:.4f} s): "
            + ", ".join(r.key for r in missed),
            icon="⚠️",
        )
    if frame.empty:
        st.info("No trace in the basket intersects this time window.", icon="ℹ️")
        return

    total = int(frame["samples"].sum())
    if total > SAMPLE_BUDGET:
        st.warning(
            f"The basket holds {total / 1e6:.0f} M samples (about "
            f"{total * 8 / 1e6:.0f} MB) — narrow the window for a snappier redraw.",
            icon="⚠️",
        )

    event = st.dataframe(
        frame, hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="multi-row",
        key="stats.basket_table",
    )
    left, right = st.columns(2)
    if left.button("Remove selected", key="stats.remove"):
        for index in sorted(event.selection.rows, reverse=True):
            if 0 <= index < len(basket):
                basket.pop(index)
        st.session_state["stats.basket"] = basket
        st.rerun()
    if right.button("Clear all", key="stats.clear"):
        st.session_state["stats.basket"] = []
        st.session_state["stats.reference"] = 0
        st.rerun()

    # A pairwise statistic gets every trace on the reference's time base.
    reference = ()
    if spec.pairwise:
        ref_index = int(st.session_state.get("stats.reference", 0))
        labels = [statistics.name(trace) for trace in extracted]
        ref_index = int(
            st.selectbox(
                "Reference",
                range(len(extracted)),
                format_func=lambda i: labels[i],
                index=ref_index if ref_index < len(extracted) else 0,
                key="stats.reference_box",
            )
        )
        st.session_state["stats.reference"] = ref_index
        gridded = traces.common_grid(extracted, extracted[ref_index])
        if any(new is not old for new, old in zip(gridded, extracted)):
            st.caption("Some traces were resampled onto the reference's time base.")
        extracted, reference = gridded, (gridded[ref_index],)

    items, failures = [], []
    for trace in extracted:
        try:
            items.append((trace, spec.compute(trace, *reference, params)))
        except Exception as error:  # noqa: BLE001 - one bad trace must not fail the page
            failures.append(f"{trace.ref.key}: {error}")
    if failures:
        st.warning(
            f"{len(failures)} of {len(extracted)} traces failed and are left out"
            " of the figure:\n\n" + "\n\n".join(failures),
            icon="⚠️",
        )
    if not items:
        st.info("Nothing to draw.", icon="ℹ️")
        return

    try:
        figure = spec.render(items, params)
    except Exception as error:  # noqa: BLE001 - a stat bug must not kill the page
        st.error(f"{type(error).__name__}: {error}", icon="⚠️")
        return
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    st.caption(
        f"Window {t_start:.4f}–{t_end:.4f} s · {len(items)} traces · "
        "R-R_sep is the horizontal distance to the separatrix at the pixel's "
        "own height, not a flux coordinate · probe ρ is the window mean, and "
        "the probe moves during its plunge."
    )


main()
