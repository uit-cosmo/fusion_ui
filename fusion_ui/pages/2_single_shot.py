"""Single shot: pick a shot, pick a plot, get the plot.

This page knows nothing about any individual analysis. It resolves a
:class:`~fusion_ui.core.registry.Target`, asks the registry what can be drawn
for that diagnostic, renders the chosen spec's parameter panel, hands the whole
lot to the store, and draws whatever comes back. Adding a plot is a new module
in :mod:`fusion_ui.plots`; nothing here changes.

What the page does own is the chrome every plot should have and none should
have to write: the time-window caption, the figure container, the error surface
for a failed run, and the provenance line -- because a result computed by last
month's ``imaging_methods`` is the trap that makes people distrust the tool.
"""

import os

import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec
from fusion_ui import ui
from fusion_ui.core import catalog, loader, params_ui, registry, store

st.set_page_config(page_title="Single shot · Shot Explorer", layout="wide")


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

    return str(row["machine"]), int(shot), diagnostic, preprocessed


def discharge_for_shot(shot):
    path, error = ui.resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(path):
        return None
    return catalog.load_discharges(path).get(shot)


def open_target(machine, shot, diagnostic, preprocessed):
    """``(Target, dataset)`` -- the file, already restricted to its window.

    Imaging files are sliced to the discharge DB's ``t_start..t_end`` before
    anything downstream sees them: they are ~500 MB over 583k samples, and no
    view has a reason to touch the whole record. Probe files have no shared time
    axis to slice on -- every quantity x position carries its own -- so they are
    handed over whole and the probe adapter does the indexing.
    """
    path = loader.dataset_path(machine, shot, diagnostic, preprocessed)
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`", icon="⚠️")
        return None, None
    ds = loader.open_dataset(path)

    if loader.TIME_DIM in ds.dims:
        t_start, t_end, source = loader.time_window(ds, discharge_for_shot(shot))
        windowed = loader.sliced(ds, t_start, t_end)
    else:
        t_start = t_end = float("nan")
        source = "none"
        windowed = ds

    target = registry.Target(
        machine=machine,
        shot=shot,
        diagnostic=diagnostic,
        preprocessed=preprocessed,
        path=path,
        t_start=t_start,
        t_end=t_end,
        window_source=source,
    )
    return target, windowed


def window_caption(target):
    if target.window_source == "none":
        return
    st.caption(
        f"Window {target.t_start:.4f}–{target.t_end:.4f} s — "
        + (
            "from the discharge DB."
            if target.window_source == "metadata"
            else "no discharge-DB entry yet, showing a centred 0.2 s default."
        )
    )


# ---------------------------------------------------------------------------
# Plot selection and the chrome around a result
# ---------------------------------------------------------------------------


def pick_spec(diagnostic):
    specs = registry.for_diagnostic(diagnostic)
    if not specs:
        st.error(f"No plot is registered for diagnostic {diagnostic!r}.", icon="⚠️")
        return None
    spec = st.sidebar.selectbox(
        "Plot", specs, format_func=lambda s: s.label, key=f"spec.{diagnostic}"
    )
    if spec.description:
        st.sidebar.caption(spec.description)
    return spec


def provenance(run, conn):
    """What produced this figure, and the button to do it again.

    ``code_version`` is stored but deliberately not part of the cache key --
    hashing it would invalidate every result on every commit. Showing it, and
    offering Recompute next to it, leaves the judgement with the person looking
    at the plot.
    """
    if run is None:
        return
    left, right = st.columns([5, 1])
    elapsed = f" in {run['seconds']:.1f} s" if run["seconds"] is not None else ""
    left.caption(
        f"Computed {run['created_at']}{elapsed} · "
        f"{run['code_version'] or 'version unknown'} · "
        f"params `{run['params_hash'][:12]}`"
    )
    if right.button("Recompute", key=f"recompute.{run['id']}"):
        store.delete_run(conn, run)
        st.rerun()


def show_failure(run, conn):
    st.error(run["error"], icon="⚠️")
    st.caption(
        f"Failed {run['created_at']} · {run['code_version'] or 'version unknown'}. "
        "The failure is recorded, so this will not retry on its own."
    )
    if st.button("Retry", key=f"retry.{run['id']}"):
        store.delete_run(conn, run)
        st.rerun()


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

    picked = pick_shot_and_target(table)
    if picked is None:
        return

    target, ds = open_target(*picked)
    if target is None:
        return

    spec = pick_spec(target.diagnostic)
    if spec is None:
        return

    window_caption(target)

    params, ready = params_ui.panel(spec, target, ds=ds)
    if not ready:
        st.info(
            f"Press **Compute** in the sidebar to run {spec.label.lower()} on "
            f"{target.label}.",
            icon="▶️",
        )
        return

    conn = ui.get_connection()
    with st.spinner(f"Computing {spec.label.lower()}…" if spec.cached else ""):
        result, run = store.result(conn, spec, target, params, ds)

    if run is not None and run["status"] == "failed":
        show_failure(run, conn)
        return

    figure = spec.render(result, params, target)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    provenance(run, conn)


main()
