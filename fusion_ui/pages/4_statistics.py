"""Statistics over arbitrary traces: time trace, PDF, PSD, ACF and CCF on one axis.

A basket of traces -- pixels off any imaging shot, channels off any probe
shot, mixed freely -- drawn together under one statistic, labelled with
magnetic coordinates. Statistics are live: a Welch PSD or a histogram is tens
of milliseconds, so widgets take effect on the next rerun with no Compute
button, and the page writes nothing to the run ledger. The cached
``taud_psd`` plot on the single-shot page stays the per-pixel,
scalar-producing spectrum; the PSD here is its exploratory,
cross-diagnostic cousin.
"""

import dataclasses
import os

import pandas as pd
import streamlit as st

import fusion_ui.stats  # noqa: F401 - importing the package registers every spec
from fusion_ui import ui
from fusion_ui.core import catalog, loader, multipixel, params_ui, probes, statistics, traces

st.set_page_config(page_title="Statistics · Shot Explorer", layout="wide")

#: Warn once past this many samples across the basket: about 160 MB, roughly
#: 32 full APD windows.
SAMPLE_BUDGET = 20_000_000

LABEL_STYLES = ("magnetic", "channel", "rz")


@st.cache_data(show_spinner=False)
def _cached_trace(path, mtime, machine, shot, diagnostic, preprocessed,
                   channel, t_start, t_end):
    """One extracted trace, memoised on ``(path, mtime, ref, window)``.

    ``mtime`` is not used in the body -- it is in the signature so a re-copied
    file busts the cache, the same trick ``loader._cached_times`` uses. Dragging
    a widget then does not re-read the trace off disk.
    """
    ds = loader.open_dataset(path)
    ref = traces.TraceRef(
        machine=machine,
        shot=int(shot),
        diagnostic=diagnostic,
        preprocessed=bool(preprocessed),
        channel=tuple(channel),
    )
    return traces.extract(ds, ref, float(t_start), float(t_end))


def discharge_for_shot(shot):
    path, error = ui.resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(path or ""):
        return None
    return catalog.load_discharges(path).get(shot)


def available_targets(row):
    """``{diagnostic: (has_raw, has_preprocessed)}`` for one shot_table row."""
    targets = {}
    for diagnostic in catalog.DIAGNOSTICS:
        availability = row[diagnostic] if diagnostic in row else ""
        if availability:
            targets[diagnostic] = ("R" in availability, "P" in availability)
    return targets


def default_picker(table):
    """``(machine, shot)`` -- the browser's selection when it is on the table."""
    selection = st.session_state.get("selection")
    machines = sorted(table["machine"].astype(str).unique())
    machine = machines[0]
    if selection and selection.get("machine") in machines:
        machine = selection["machine"]
    shots = sorted(table[table["machine"].astype(str) == machine]["shot"].astype(int))
    shot = shots[0]
    if (
        selection
        and selection.get("machine") == machine
        and selection.get("shot") in shots
    ):
        shot = selection["shot"]
    return machine, int(shot)


def ensure_window(machine, shot, diagnostic, preprocessed):
    """Seed the window from the discharge window once, for the first shot."""
    if "stats.t_start" in st.session_state and "stats.t_end" in st.session_state:
        return
    try:
        path = loader.dataset_path(machine, shot, diagnostic, preprocessed)
        ds = loader.open_dataset(path)
    except (OSError, KeyError, ValueError):
        return
    t_start, t_end, _ = loader.time_window(ds, discharge_for_shot(shot))
    st.session_state.setdefault("stats.t_start", float(t_start))
    st.session_state.setdefault("stats.t_end", float(t_end))


def channel_text(channel):
    kind = channel[0] if channel else ""
    if kind == "pixel":
        return f"pixel ({channel[1]}, {channel[2]})"
    if kind == "probe":
        return f"{channel[1]}_{channel[2]}"
    return str(channel)


def magnetic_text(trace):
    coords = trace.coords or {}
    kind = trace.ref.channel[0] if trace.ref.channel else ""
    if kind == "pixel" and coords.get("dr_sep") is not None:
        return f"R-R_sep={coords['dr_sep']:+.2f} cm"
    if kind == "probe" and coords.get("rho") is not None:
        return f"rho={coords['rho']:.2f}"
    return ""


def add_picker(table):
    """The "Add traces" expander: shot, diagnostic, pixels or channels."""
    basket = st.session_state.get("stats.basket", [])
    with st.expander("Add traces", expanded=not basket):
        if table.empty:
            st.warning(
                "The shot index is empty — run a rescan on the Shot browser page.",
                icon="⚠️",
            )
            return
        machine, default_shot = default_picker(table)
        machines = sorted(table["machine"].astype(str).unique())
        if len(machines) > 1:
            machine = st.selectbox(
                "Machine", machines, index=machines.index(machine)
            )
        shots = sorted(
            table[table["machine"].astype(str) == machine]["shot"].astype(int)
        )
        shot = st.selectbox(
            "Shot", shots, index=shots.index(default_shot)
        )
        row = table[table["machine"].astype(str) == machine]
        row = row[row["shot"].astype(int) == int(shot)].iloc[0]
        targets = available_targets(row)
        if not targets:
            st.warning("No diagnostic files indexed for this shot.", icon="⚠️")
            return
        names = list(targets)
        diagnostic = st.radio("Diagnostic", names, horizontal=True)
        has_raw, has_preprocessed = targets[diagnostic]
        if has_raw and has_preprocessed:
            preprocessed = (
                st.radio("Version", ["Preprocessed", "Raw"], horizontal=True)
                == "Preprocessed"
            )
        else:
            preprocessed = has_preprocessed

        path = loader.dataset_path(machine, int(shot), diagnostic, preprocessed)
        if not os.path.exists(path):
            st.error(f"File not found: `{path}`", icon="⚠️")
            return
        ds = loader.open_dataset(path)
        target = loader_target(machine, int(shot), diagnostic, preprocessed, path)

        if diagnostic in traces.IMAGING:
            pixels = multipixel.selector(ds, target)
            add_label = f"Add {len(pixels)} selected pixels"
            if st.button(add_label, key="stats.add_pixels", disabled=not pixels):
                for x, y in pixels:
                    ref = traces.TraceRef(
                        machine, int(shot), diagnostic, preprocessed,
                        ("pixel", int(x), int(y)),
                    )
                    if ref not in basket:
                        basket.append(ref)
                st.session_state["stats.basket"] = basket
                st.rerun()
        else:
            found = probes.quantities_and_positions(ds)
            options = [
                f"{quantity}_{position}"
                for quantity, positions in sorted(found.items())
                for position in positions
            ]
            picked = st.multiselect("Channels", options, key="stats.channels")
            if st.button(
                f"Add {len(picked)} selected channels",
                key="stats.add_channels",
                disabled=not picked,
            ):
                lookup = {
                    f"{quantity}_{position}": ("probe", quantity, position)
                    for quantity, positions in found.items()
                    for position in positions
                }
                for name in picked:
                    ref = traces.TraceRef(
                        machine, int(shot), diagnostic, preprocessed,
                        lookup[name],
                    )
                    if ref not in basket:
                        basket.append(ref)
                st.session_state["stats.basket"] = basket
                st.rerun()


def loader_target(machine, shot, diagnostic, preprocessed, path):
    """A ``Target`` for the pixel selector, which only reads ``target.key``."""
    from fusion_ui.core import registry

    return registry.Target(
        machine=machine,
        shot=int(shot),
        diagnostic=diagnostic,
        preprocessed=bool(preprocessed),
        path=path,
        t_start=float("nan"),
        t_end=float("nan"),
        window_source="none",
    )


def basket_frame(basket, t_start, t_end, style):
    """The basket as a dataframe, extracting each trace through the cache."""
    rows, extracted, missed = [], [], []
    for ref in basket:
        try:
            path = loader.dataset_path(
                ref.machine, ref.shot, ref.diagnostic, ref.preprocessed
            )
            mtime = os.path.getmtime(path)
        except OSError:
            missed.append(ref)
            continue
        try:
            trace = _cached_trace(
                path, mtime, ref.machine, ref.shot, ref.diagnostic,
                ref.preprocessed, tuple(ref.channel), t_start, t_end,
            )
        except Exception:  # noqa: BLE001 - one bad trace must not fail the page
            missed.append(ref)
            continue
        if trace is None:
            missed.append(ref)
            continue
        try:
            labelled = dataclasses.replace(trace, label=traces.label(trace, style))
        except Exception:  # noqa: BLE001 - a label must never fail a trace
            labelled = trace
        extracted.append(labelled)
        rows.append(
            {
                "shot": ref.shot,
                "diagnostic": ref.diagnostic,
                "channel": channel_text(ref.channel),
                "label": labelled.label,
                "magnetic": magnetic_text(labelled),
                "samples": len(labelled.time),
            }
        )
    return pd.DataFrame(rows), extracted, missed


def main():
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("shot_selection", [])
    st.session_state.setdefault("stats.basket", [])
    st.session_state.setdefault("stats.reference", 0)

    st.title("Statistics")

    table = ui.cached_shot_table()

    # ---- sidebar ---------------------------------------------------------
    specs = statistics.all_specs()
    keys = [s.key for s in specs]
    if "stats.spec" not in st.session_state or st.session_state["stats.spec"] not in keys:
        st.session_state["stats.spec"] = keys[0]
    st.sidebar.selectbox(
        "Statistic", keys, format_func=lambda k: statistics.get(k).label,
        key="stats.spec",
    )
    spec = statistics.get(st.session_state["stats.spec"])
    if spec.description:
        st.sidebar.caption(spec.description)
    if spec.key == "psd":
        st.sidebar.caption(
            "The exploratory cousin of the cached Duration time (PSD fit) on "
            "the single-shot page: same estimator, but cross-diagnostic and "
            "uncached. Trust the cached one for numbers on a multi-shot axis."
        )

    params = params_ui.form(spec.params, f"stats.params.{spec.key}",
                            container=st.sidebar)

    if not table.empty:
        machine, shot = default_picker(table)
        row = table[table["machine"].astype(str) == machine]
        row = row[row["shot"].astype(int) == int(shot)].iloc[0]
        targets = available_targets(row)
        first_diag = next(iter(targets)) if targets else "apd"
        _, first_prep = targets.get(first_diag, (True, False))
        ensure_window(machine, int(shot), first_diag, bool(first_prep))
    t_start = st.sidebar.number_input("Start time [s]", format="%g",
                                      key="stats.t_start")
    t_end = st.sidebar.number_input("End time [s]", format="%g", key="stats.t_end")
    if st.sidebar.button("Reset to the discharge window",
                         key="stats.reset_window"):
        st.session_state.pop("stats.t_start", None)
        st.session_state.pop("stats.t_end", None)
        st.rerun()
    st.sidebar.radio(
        "Labels", LABEL_STYLES,
        format_func=lambda s: {"magnetic": "magnetic (R-R_sep / ρ)",
                               "channel": "channel",
                               "rz": "R, Z"}[s],
        key="stats.labels",
    )
    style = st.session_state.get("stats.labels", "magnetic")

    # ---- main ------------------------------------------------------------
    add_picker(table)

    basket = st.session_state.get("stats.basket", [])
    if not basket:
        st.info(
            "The basket is empty — add pixel traces or probe channels above, "
            "then every trace is drawn together under one statistic.",
            icon="ℹ️",
        )
        return

    frame, extracted, missed = basket_frame(basket, float(t_start), float(t_end),
                                            style)
    if missed:
        st.warning(
            "Left out of the window "
            f"({float(t_start):.4f}–{float(t_end):.4f} s): "
            + ", ".join(r.key for r in missed),
            icon="⚠️",
        )
    if frame.empty:
        st.info("No trace in the basket intersects this time window.", icon="ℹ️")
        return

    total = int(frame["samples"].sum())
    if total > SAMPLE_BUDGET:
        st.warning(
            f"The basket holds {total / 1e6:.0f} M samples "
            "(about {0:.0f} MB) — narrow the window for a snappier redraw.".format(
                total * 8 / 1e6
            ),
            icon="⚠️",
        )

    event = st.dataframe(
        frame, hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="multi-row",
        key="stats.basket_table",
    )
    left, right = st.columns(2)
    if left.button("Remove selected", key="stats.remove"):
        rows = sorted(event.selection.rows, reverse=True)
        for index in rows:
            if 0 <= index < len(basket):
                basket.pop(index)
        st.session_state["stats.basket"] = basket
        st.rerun()
    if right.button("Clear all", key="stats.clear"):
        st.session_state["stats.basket"] = []
        st.session_state["stats.reference"] = 0
        st.rerun()

    reference = None
    ref_index = 0
    if spec.pairwise:
        ref_index = int(st.session_state.get("stats.reference", 0))
        if ref_index >= len(extracted):
            ref_index = 0
        labels = [t.label or t.ref.key for t in extracted]
        ref_index = int(st.selectbox(
            "Reference", range(len(extracted)),
            format_func=lambda i: labels[i], index=ref_index,
            key="stats.reference_box",
        ))
        st.session_state["stats.reference"] = ref_index

    if spec.pairwise:
        try:
            gridded = traces.common_grid(extracted, extracted[ref_index])
        except Exception as error:  # noqa: BLE001 - resampling must not kill page
            st.error(f"{type(error).__name__}: {error}", icon="⚠️")
            return
        if any(new is not old for new, old in zip(gridded, extracted)):
            st.caption("Some traces were resampled onto the reference's time base.")
        reference = gridded[ref_index]
        items, failures = [], []
        for trace in gridded:
            try:
                items.append((trace, spec.compute(trace, reference, params)))
            except Exception as error:  # noqa: BLE001 - one bad trace must not fail the page
                failures.append(f"{trace.ref.key}: {error}")
    else:
        items, failures = [], []
        for trace in extracted:
            try:
                items.append((trace, spec.compute(trace, params)))
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
        "Window "
        f"{float(t_start):.4f}–{float(t_end):.4f} s · {len(items)} traces · "
        "R-R_sep is the horizontal distance to the separatrix at the pixel's "
        "own height, not a flux coordinate · probe ρ is the window mean, and "
        "the probe moves during its plunge."
    )


main()
