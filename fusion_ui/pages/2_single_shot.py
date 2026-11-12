"""Single shot: pick a shot, pick a plot, get the plot.

The page knows nothing about any one analysis. It resolves a
:class:`~fusion_ui.core.registry.Target`, asks the registry what can be drawn
for that diagnostic, renders the chosen spec's parameter panel, hands it all to
the store and draws what comes back, with the chrome every plot shares: the
window caption, the error for a failed run, and the line that says what
computed the result.

Nothing batch only is computed here, since the page runs inside the process
that serves the whole group. A batch-only spec is looked up and shown, or the
command that fills it is; a spec built on one is computed only once every
batch-only link beneath it is cached.
"""

import math
import os
from datetime import datetime

import streamlit as st

import fusion_ui.plots  # noqa: F401 - importing the package registers every spec
from fusion_ui import ui
from fusion_ui.core import loader, multipixel, params_ui, precompute, registry, store

st.set_page_config(page_title="Single shot · Shot Explorer", layout="wide")


# ---------------------------------------------------------------------------
# The shot and the file, seeded from the browser's selection
# ---------------------------------------------------------------------------


def pick_shot_and_target(table):
    """``(machine, shot, diagnostic, preprocessed)`` from the sidebar, or ``None``.

    Each picker opens on the browser's selection as far as the pickers above it
    still agree with it.
    """
    selection = st.session_state.get("selection") or {}

    def index(options, name):
        return options.index(selection[name]) if selection.get(name) in options else 0

    # The index is keyed by (machine, shot): a rescan deletes only its own
    # machine's rows, so two machines can hold the same shot number.
    machines = sorted(table["machine"].astype(str).unique())
    machine = machines[0]
    if len(machines) > 1:
        machine = st.sidebar.selectbox(
            "Machine", machines, index=index(machines, "machine")
        )
    if selection.get("machine") != machine:
        selection = {}
    indexed = table[table["machine"].astype(str) == machine]

    shots = sorted(indexed["shot"].astype(int).unique())
    shot = st.sidebar.selectbox("Shot", shots, index=index(shots, "shot"))
    if selection.get("shot") != shot:
        selection = {}

    targets = ui.available_targets(indexed.set_index("shot").loc[shot])
    if not targets:
        st.sidebar.warning("No diagnostic files indexed for this shot.", icon="⚠️")
        return None
    names = list(targets)
    diagnostic = st.sidebar.selectbox(
        "Diagnostic", names, index=index(names, "diagnostic")
    )

    has_raw, preprocessed = targets[diagnostic]
    if has_raw and preprocessed:
        # Preprocessed unless the selection names the raw file of this
        # diagnostic: a plot that runs on the raw file only is offered there.
        wants_raw = (
            selection.get("diagnostic") == diagnostic
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
    return machine, int(shot), diagnostic, preprocessed


def open_target(machine, shot, diagnostic, preprocessed):
    """``(Target, dataset)``: the file as opened (lazy, nothing read) and the
    window it is to be cut to.

    The window is the discharge DB's ``t_start..t_end``, or a centred 0.2 s one
    when the shot has no entry. A probe file has no shared time axis: it gets
    the discharge DB's window when there is one, and none otherwise.
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

    discharge = ui.discharge(shot)
    if loader.TIME_DIM in ds.dims:
        t_start, t_end, source = loader.time_window(ds, discharge)
    else:
        t_start, t_end, source = loader.metadata_window(discharge)

    target = registry.Target(
        machine=machine,
        shot=shot,
        diagnostic=diagnostic,
        preprocessed=preprocessed,
        path=path,
        t_start=t_start,
        t_end=t_end,
        window_source=source,
        discharge=discharge,
    )
    return target, ds


def open_companions(spec, target, table):
    """``{diagnostic: dataset or None}`` for each of ``spec.companions``.

    Each is the raw file of that diagnostic for the same shot, opened lazily
    and cut to the target's window as the spec's own file is. ``None`` where
    the index holds no such file for the shot, or it cannot be opened, which
    the sidebar says.
    """
    rows = table[
        (table["machine"].astype(str) == target.machine)
        & (table["shot"].astype(int) == target.shot)
    ]
    found = {}
    for diagnostic in spec.companions:
        found[diagnostic] = None
        if "R" not in str(rows.iloc[0][diagnostic] or ""):
            continue
        path = loader.dataset_path(target.machine, target.shot, diagnostic, False)
        try:
            companion = loader.open_lazily(path)
        # Gone since the rescan, or unreadable: the spec's own view is still drawn.
        except Exception as error:  # noqa: BLE001
            st.sidebar.warning(
                f"Could not open the {diagnostic} file of this shot, `{path}`: "
                f"{type(error).__name__}: {error}",
                icon="⚠️",
            )
            continue
        if math.isfinite(target.t_start) and math.isfinite(target.t_end):
            companion = loader.input_for(companion, target.t_start, target.t_end)
        found[diagnostic] = companion
    return found


def window_caption(target, ds):
    """The window the plot is drawn over.

    A preprocessed file is cropped to its analysis window (the discharge window
    cut to the gas puff) and stores it: the caption gives that, with the
    discharge DB's beside it. Every other file is cut to the discharge window,
    but for a probe file, which is handed to the plot whole.
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
        + (
            ""
            if loader.TIME_DIM in ds.dims
            else " A probe file has no shared time axis, so it is not cut to it."
        )
    )


# ---------------------------------------------------------------------------
# The plot, and the chrome around a result
# ---------------------------------------------------------------------------


def pick_spec(diagnostic, preprocessed):
    specs = registry.for_diagnostic(diagnostic, preprocessed)
    if not specs:
        kind = "preprocessed" if preprocessed else "raw"
        st.error(
            f"No plot is registered for {kind} {diagnostic!r} files.", icon="⚠️"
        )
        return None
    # The multi-shot page's jump sets this key, so it stays per diagnostic. A
    # choice this version of the file does not offer falls back to the first.
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
    """A ledger timestamp, to the second."""
    try:
        return datetime.fromisoformat(created_at).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return created_at


def show_params_to_save(spec, params, container=st):
    """What to save as the ``--params-json`` file a command names.

    :func:`fusion_ui.core.precompute.command` names that file whenever the
    parameters are not the defaults, and the page cannot write it for the
    person. Nothing is shown for the defaults.
    """
    if precompute.is_default(spec, params):
        return
    container.caption(f"with these parameters saved as `{precompute.PARAMS_FILE}`:")
    container.code(params_ui.hash_params(spec.key, params)[1], language="json")


def provenance(run, conn, spec, target, params):
    """What produced this figure, and the button to do it again.

    ``code_version`` is not part of the cache key, so it is shown, with
    Recompute beside it, for the person to judge. A batch-only result gets the
    command instead of the button, which would delete a result this page
    cannot compute again.
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
        # The sentence runs on into the parameters to save, when there are any.
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
    :func:`fusion_ui.core.store.missing_batch_upstreams` returns it.
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


def draw(spec, result, params, target, companions=None):
    """Render ``result``; ``False`` when the render itself failed.

    A spec that names companions is called as ``render(result, params, target,
    companions)``. A render that returns no figure drew into Streamlit itself.
    """
    extra = (companions,) if spec.companions else ()
    try:
        figure = spec.render(result, params, target, *extra)
    except Exception as error:  # noqa: BLE001 - a plot bug must not kill the page
        st.error(f"{type(error).__name__}: {error}", icon="⚠️")
        st.caption(f"Rendering {spec.label.lower()} failed.")
        return False
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    return True


def show_batch_only(conn, spec, target, params):
    """A batch-only spec: its stored result, or the command that fills it.

    :func:`fusion_ui.core.store.lookup` never computes, so the form's Show
    button only commits new parameters.
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
    # The call `fusion-ui precompute` makes, so the two compute on the same cut.
    ds = loader.input_for(opened, target.t_start, target.t_end, spec.whole_record)

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
        # The check above passed, but the batch-only link was deleted since, or
        # its blob does not load: the store refused.
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

    companions = open_companions(spec, target, table) if spec.companions else None
    if draw(spec, result, params, target, companions):
        provenance(run, conn, spec, target, params)


main()
