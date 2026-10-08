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

One rule it enforces for every spec: nothing batch only is ever computed here.
A batch-only spec is looked up and shown, or the command that fills it is; a
spec built on one is computed only once every batch-only link beneath it is
cached. Everything this page computes runs inside the process that serves the
whole group.
"""

import os
from datetime import datetime

import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec
from fusion_ui import ui
from fusion_ui.core import (
    catalog,
    loader,
    multipixel,
    params_ui,
    precompute,
    registry,
    store,
)

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
    if table.empty:
        return None

    # The index is keyed by (machine, shot), not by shot: `rescan` runs for one
    # machine and only ever deletes that machine's rows, so pointing
    # FUSION_MACHINE somewhere else leaves both machines indexed for good.
    # Picking on the shot number alone would merge the two lists and then hand
    # `available_targets` a two-row frame, whose columns are Series -- an
    # ambiguous-truth ValueError rather than a wrong answer.
    machines = sorted(table["machine"].astype(str).unique())
    if len(machines) > 1:
        default_machine = (
            selection["machine"]
            if selection and selection["machine"] in machines
            else machines[0]
        )
        machine = st.sidebar.selectbox(
            "Machine", machines, index=machines.index(default_machine)
        )
    else:
        # One machine is the normal case; a selectbox with a single option is
        # just noise in the sidebar.
        machine = machines[0]
    indexed = table[table["machine"].astype(str) == machine]

    shots = sorted(indexed["shot"].astype(int).unique())
    if not shots:
        return None

    on_this_machine = bool(selection) and selection["machine"] == machine
    default_shot = (
        selection["shot"]
        if on_this_machine and selection["shot"] in shots
        else shots[0]
    )
    shot = st.sidebar.selectbox("Shot", shots, index=shots.index(default_shot))

    row = indexed.set_index("shot").loc[shot]
    targets = available_targets(row)
    if not targets:
        st.sidebar.warning("No diagnostic files indexed for this shot.", icon="⚠️")
        return None
    diagnostic_names = list(targets)

    default_diagnostic = (
        selection["diagnostic"]
        if on_this_machine
        and selection["shot"] == shot
        and selection["diagnostic"] in targets
        else diagnostic_names[0]
    )
    diagnostic = st.sidebar.selectbox(
        "Diagnostic", diagnostic_names, index=diagnostic_names.index(default_diagnostic)
    )

    has_raw, has_preprocessed = targets[diagnostic]
    if has_raw and has_preprocessed:
        # The browser lands on the preprocessed file whenever there is one. A selection that names
        # the raw file one -- the Fields page's link to the dead-pixel view, a multi-shot point
        # computed on a raw file -- has to open on it, or the plot it came for is not offered.
        wants_raw = (
            on_this_machine
            and selection["shot"] == shot
            and selection["diagnostic"] == diagnostic
            and selection.get("preprocessed") is False
        )
        preprocessed = (
            st.sidebar.radio(
                "Version",
                ["Preprocessed", "Raw"],
                index=1 if wants_raw else 0,
                horizontal=True,
            )
            == "Preprocessed"
        )
    else:
        preprocessed = has_preprocessed

    return machine, int(shot), diagnostic, preprocessed


def discharge_for_shot(shot):
    path, error = ui.resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(path):
        return None
    return catalog.load_discharges(path).get(shot)


def open_target(machine, shot, diagnostic, preprocessed):
    """``(Target, dataset)`` -- the file as opened (lazy, nothing read), and the
    window it is to be cut to.

    The window is the discharge DB's ``t_start..t_end``, or a centred 0.2 s one
    when the shot has no entry. :func:`input_for` cuts the file to it once the
    plot is picked: imaging files are ~500 MB over 583k samples, and no view has
    a reason to touch the whole record -- but for one spec that finds something
    against the start of the record, which declares it (``whole_record``). Probe
    files have no shared time axis to cut -- every quantity x position carries
    its own -- so they are handed over whole and the probe adapter does the
    indexing.
    """
    path = loader.dataset_path(machine, shot, diagnostic, preprocessed)
    if not os.path.exists(path):
        st.error(f"File not found: `{path}`", icon="⚠️")
        return None, None
    try:
        ds = loader.open_dataset(path)
    except Exception as error:  # noqa: BLE001 - file vanished or is unreadable
        st.error(f"Could not open `{path}`: {type(error).__name__}: {error}", icon="⚠️")
        return None, None

    if loader.TIME_DIM in ds.dims:
        t_start, t_end, source = loader.time_window(ds, discharge_for_shot(shot))
    else:
        t_start = t_end = float("nan")
        source = "none"

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
    return target, ds


def input_for(spec, target, ds):
    """What ``spec`` draws from and computes on: the opened file cut to the
    target's window, or all of it with that window attached when the spec
    declares ``whole_record``. The same call ``fusion-ui precompute`` makes, so
    the two cannot disagree on what a result was computed from."""
    return loader.input_for(ds, target.t_start, target.t_end, spec.whole_record)


def window_caption(target, ds):
    """The window the plot is drawn over.

    A preprocessed file is cropped to its analysis window (the discharge window
    cut to the gas puff) and stores it as ``analysis_window``: that is the
    window of its data, and the caption gives it, with the discharge DB's beside
    it. Every other file is cut to the discharge window.
    """
    if target.window_source == "none":
        return
    stored = loader.stored_window(ds)
    if stored is not None:
        beside = (
            f" (the discharge DB's window is {target.t_start:.4f}–{target.t_end:.4f} s)"
            if target.window_source == "metadata"
            else ""
        )
        st.caption(
            f"Window {stored[0]:.4f}–{stored[1]:.4f} s — the analysis window this "
            f"file is cropped to{beside}."
        )
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


def pick_spec(diagnostic, preprocessed):
    specs = registry.for_diagnostic(diagnostic, preprocessed)
    if not specs:
        kind = "preprocessed" if preprocessed else "raw"
        st.error(
            f"No plot is registered for {kind} {diagnostic!r} files.", icon="⚠️"
        )
        return None
    # The key is shared with the multi-shot page's click-to-jump, so it stays
    # per diagnostic. The raw and preprocessed files offer different plots: a
    # choice the other version does not offer falls back to the first one.
    key = f"spec.{diagnostic}"
    if key in st.session_state and st.session_state[key] not in specs:
        del st.session_state[key]
    spec = st.sidebar.selectbox(
        "Plot", specs, format_func=lambda s: s.label, key=key
    )
    if spec.description:
        st.sidebar.caption(spec.description)
    return spec


def when(created_at):
    """A ledger timestamp to the second: microseconds are kept for ordering runs
    against each other, not for reading."""
    try:
        return datetime.fromisoformat(created_at).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return created_at


def show_params_to_save(spec, params, container=st):
    """The parameter set a command's ``--params-json`` file holds, when it is not the defaults.

    :func:`fusion_ui.core.precompute.command` names :data:`~fusion_ui.core.precompute.PARAMS_FILE` for
    any other parameters, and the page cannot write a file for the person: it shows what to save.
    Nothing is shown for the defaults, which the command needs no file for.
    """
    if precompute.is_default(spec, params):
        return
    container.caption(f"with these parameters saved as `{precompute.PARAMS_FILE}`:")
    container.code(params_ui.hash_params(spec.key, params)[1], language="json")


def provenance(run, conn, spec, target, params):
    """What produced this figure, and the button to do it again.

    ``code_version`` is stored but deliberately not part of the cache key --
    hashing it would invalidate every result on every commit. Showing it, and
    offering Recompute next to it, leaves the judgement with the person looking
    at the plot. A batch-only result gets the command instead of the button:
    Recompute would delete a result this page cannot compute again.
    """
    if run is None:
        return
    left, right = st.columns([5, 1])
    elapsed = f" in {run['seconds']:.1f} s" if run["seconds"] is not None else ""
    left.caption(
        f"Computed {when(run['created_at'])}{elapsed} · "
        f"{run['code_version'] or 'version unknown'} · "
        f"params `{run['params_hash'][:12]}`"
    )
    if spec.batch_only:
        # The command names a --params-json file when the parameters are not the
        # defaults, and the page cannot write it: show what goes in it, as
        # `show_batch_missing` does, and let the sentence run on into it.
        saved = not precompute.is_default(spec, params)
        left.caption(
            "Computed in batch only. To compute it again, run "
            f"`{precompute.command(spec, target, params, '--force')}`"
            + ("" if saved else ".")
        )
        show_params_to_save(spec, params, left)
        return
    if right.button("Recompute", key=f"recompute.{run['id']}"):
        store.delete_run(conn, run)
        st.rerun()


def show_failure(run, conn):
    st.error(run["error"], icon="⚠️")
    st.caption(
        f"Failed {when(run['created_at'])} · "
        f"{run['code_version'] or 'version unknown'}. "
        "The failure is recorded, so this will not retry on its own."
    )
    if st.button("Retry", key=f"retry.{run['id']}"):
        store.delete_run(conn, run)
        st.rerun()


def show_batch_missing(missing, target):
    """The command that fills each batch-only result this view lacks.

    ``missing`` is ``[(spec, params, run), …]`` as
    :func:`fusion_ui.core.store.missing_batch_upstreams` returns it. Nothing is
    computed here: a batch-only analysis takes most of an hour on one core, and
    this page runs inside the process that serves everyone.
    """
    for link, link_params, run in missing:
        if run is not None and run["status"] == "failed":
            st.error(f"{link.label} failed in batch: {run['error']}", icon="⚠️")
            flags = ("--retry-failed",)
        else:
            st.warning(
                f"{link.label} is computed in batch only, and is not cached for "
                f"{target.label} with these parameters. Nothing is computed "
                "here; fill it from the command line on the server:",
                icon="⏳",
            )
            # An `ok` row whose blob cannot be read is skipped as cached by a
            # plain precompute; only --force replaces it.
            flags = ("--force",) if run is not None else ()
        st.code(precompute.command(link, target, link_params, *flags), language="bash")
        show_params_to_save(link, link_params)


def draw(spec, result, params, target):
    """Render ``result``; ``False`` when the render itself failed."""
    try:
        figure = spec.render(result, params, target)
    except Exception as error:  # noqa: BLE001 - a plot bug must not kill the page
        st.error(f"{type(error).__name__}: {error}", icon="⚠️")
        st.caption(f"Rendering {spec.label.lower()} failed.")
        return False
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    return True


def show_batch_only(conn, spec, target, params):
    """A batch-only spec: its stored result, or the command that fills it.

    Read with :func:`fusion_ui.core.store.lookup`, which never computes, so
    there is no Compute to wait for: the form's Show button only commits new
    parameters.
    """
    result, run = store.lookup(conn, spec, target, params)
    if result is None:
        show_batch_missing([(spec, params, run)], target)
        return
    if draw(spec, result, params, target):
        provenance(run, conn, spec, target, params)


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

    target, opened = open_target(*picked)
    if target is None:
        return

    spec = pick_spec(target.diagnostic, target.preprocessed)
    if spec is None:
        return
    ds = input_for(spec, target, opened)

    window_caption(target, opened)

    if multipixel.supported(spec) and st.sidebar.radio(
        "Pixels", ["One", "Many"], horizontal=True, key=f"mode.{spec.key}"
    ) == "Many":
        params = params_ui.form(
            spec.params, f"params.{spec.key}", container=st.sidebar, spec=spec, ds=ds
        )
        if spec.cached:
            st.sidebar.caption(
                "The pixel selection below overrides refx/refy in this mode."
            )
        multipixel.view(ui.get_connection(), spec, target, params, ds)
        return

    params, ready = params_ui.panel(spec, target, ds=ds)
    conn = ui.get_connection()

    if spec.batch_only:
        show_batch_only(conn, spec, target, params)
        return

    # Built on a batch-only result that is not cached: say what to run rather
    # than offer a Compute that would have to start it in this process.
    missing = store.missing_batch_upstreams(conn, spec, target, params)
    if missing:
        show_batch_missing(missing, target)
        return

    if not ready:
        st.info(
            f"Press **Compute** in the sidebar to run {spec.label.lower()} on "
            f"{target.label}.",
            icon="▶️",
        )
        return

    try:
        with st.spinner(f"Computing {spec.label.lower()}…" if spec.cached else ""):
            result, run = store.result(conn, spec, target, params, ds)
    except store.BatchOnlyError as error:
        # The check above passed, but a batch-only link went missing since (a
        # `--force` deleting it) or its blob would not load: the store refused.
        digest, _ = params_ui.hash_params(error.spec.key, error.params)
        row = store.find_run(conn, target, error.spec.key, digest)
        show_batch_missing([(error.spec, error.params, row)], target)
        return
    except Exception as error:  # noqa: BLE001 - never show a traceback for a plot
        st.error(f"{type(error).__name__}: {error}", icon="⚠️")
        st.caption(
            "An unexpected error escaped the run ledger. Re-trying sometimes "
            "helps after a full disk or a permissions fix."
        )
        return

    if run is not None and run["status"] == "failed":
        show_failure(run, conn)
        return

    if draw(spec, result, params, target):
        provenance(run, conn, spec, target, params)


main()
