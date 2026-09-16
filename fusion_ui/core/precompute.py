"""The ``fusion-ui precompute`` engine: fill the cache for a plot overnight.

The single-shot view computes on demand and caches forever, so the first person
to want a quantity pays its full cost (half a minute for 2DCA, half an hour for
the velocity field). ``precompute`` walks the shot index instead and runs a
plot's ``compute`` on every matching shot ahead of time, so the multi-shot view
-- and everyone else -- reads warm caches. It is the same ``store.result`` call
the page makes, just driven from a list of targets instead of a widget.

Cache hits are skipped without even opening the data file: a single APD record
is ~500 MB, so an overnight fill must not re-read what is already stored.
"""

import errno
import os
import time
from dataclasses import dataclass

import xarray as xr

from fusion_ui import config
from fusion_ui.core import catalog, loader, multipixel, registry, shared, store


@dataclass
class PrecomputeStats:
    plot: str
    considered: int = 0  # targets offered
    cached: int = 0  # already ok, skipped without opening the file
    computed: int = 0  # newly computed ok (cache miss or --force)
    failed: int = 0  # recorded as a failed run
    seconds: float = 0.0

    def summary(self):
        parts = [
            f"{self.plot}: {self.considered} shots",
            f"{self.computed} computed",
            f"{self.cached} cached",
        ]
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts) + f" in {self.seconds:.1f}s"


def load_discharges():
    """``{shot: PlasmaDischarge}`` from the read-only descriptor, or ``{}``."""
    try:
        path = config.DISCHARGE_DB_PATH
    except RuntimeError:
        return {}
    if not os.path.exists(path):
        return {}
    return catalog.load_discharges(path)


def targets_for(conn, spec, machine=None, shots=None):
    """One :class:`~fusion_ui.core.registry.Target` per indexed shot this spec accepts.

    ``shots`` is an optional set of shot numbers to restrict to; ``machine``
    defaults to ``config.MACHINE``. The time window is left ``NaN`` here -- it is
    derived from the descriptor (or the record itself) when the file is opened
    in :func:`run`, exactly as the single-shot page does.

    The order is fixed by the query, not left to the query plan: an overnight
    fill prints one line per shot and is watched (and resumed) by shot number,
    so two runs over the same index must walk it the same way.
    """
    machine = machine or config.MACHINE
    rows = conn.execute(
        "SELECT shot, diagnostic, preprocessed, path FROM shots WHERE machine = ?"
        " ORDER BY shot, diagnostic, preprocessed",
        (machine,),
    ).fetchall()
    targets = []
    for row in rows:
        if row["diagnostic"] not in spec.diagnostics:
            continue
        if shots is not None and row["shot"] not in shots:
            continue
        targets.append(
            registry.Target(
                machine=machine,
                shot=row["shot"],
                diagnostic=row["diagnostic"],
                preprocessed=bool(row["preprocessed"]),
                path=row["path"],
                t_start=float("nan"),
                t_end=float("nan"),
                window_source="none",
            )
        )
    return targets


def default_params(spec, pixel=None):
    """The spec's default parameter set, optionally with a reference pixel.

    ``pixel`` is an ``(x, y)`` tuple applied to whichever fields are named
    ``refx`` / ``refy`` -- spectra and velocity_tde carry them at the top, and
    the 2DCA-derived plots nest them under ``two_dca``.
    """
    params = spec.params()
    if pixel is not None:
        return multipixel.with_pixel(params, pixel[0], pixel[1])
    return params


def _now_label():
    # Wall-clock prefix for overnight-log reading: tailing the log shows not
    # just which shot is in flight but since when.
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _one_line(message):
    return str(message).replace("\n", " | ")


#: Errnos that mean "the setup is wrong", not "the analysis failed": a cache
#: directory the other writer owns, a read-only mount, a missing group.
_INFRA_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


def _is_infra_error(error):
    """Whether ``error`` is the environment, not the analysis.

    An infrastructure failure must not become a ``failed`` ledger row: it would
    be skipped without ``--force`` ever after, although nothing about the shot
    or the parameters is wrong.
    """
    return isinstance(error, PermissionError) or (
        isinstance(error, OSError) and error.errno in _INFRA_ERRNOS
    )


def _permission_hint():
    try:
        cache = config.CACHE_DIR
    except RuntimeError:
        cache = "$FUSION_UI_CACHE"
    return (
        "cannot write the result cache, which is shared by two accounts (the"
        " service and whoever runs precompute by hand): check `id` shows the"
        " service group, then as root `chmod -R g+w"
        f" {cache}`, `find {cache} -type d -exec chmod g+s {{}} +` and"
        " `usermod -aG <service-user> <you>` (log out and back in). Then just"
        " rerun -- infrastructure failures leave no `failed` row behind, so no"
        " `--force` is needed for them"
    )


def _output_writable(spec, target, params_hash):
    """Whether this process can save the blob for ``target``.

    Creates the parent directory first (group-writable, per
    ``fusion_ui.core.shared``). ``(False, parent)`` means running the analysis
    now would only waste it -- the ~500 MB open and minutes of compute would
    end in a ``PermissionError`` on the save. Only the directory's owner (or
    root) can repair that, so no repair is attempted here; the caller reports
    it instead.
    """
    try:
        parent = os.path.dirname(store.blob_path(spec.key, params_hash, target))
    except RuntimeError:
        return True, ""  # cache unconfigured; let the compute surface that
    try:
        shared.makedirs(parent)
    except OSError:
        return False, parent
    if not os.access(parent, os.W_OK | os.X_OK):
        return False, parent
    return True, parent


def run(conn, spec, targets, params, force=False, log=None):
    """Compute ``spec`` with ``params`` on every target, skipping cache hits.

    A cache hit needs an ``ok`` run *and* its blob on disk: the ledger alone is
    not enough, because a cleared cache directory would otherwise make a fill
    skip a result it was asked to warm. Detecting it still costs no file open.

    A target whose compute raises still gets a ``failed`` row (via the store),
    so a broken shot does not stop the rest of the fill -- and a ``failed`` run
    is skipped without reopening its file, because ``store.result`` would only
    hand the same failure back. ``--force`` is the explicit retry.

    ``log``, when given, is called with one line per event -- a header, then
    each target's skip/computing/done line -- so an overnight fill watched
    through ``tail -f`` shows which shot is in flight. Commits stay per-target
    atomic: a line saying ``ok`` means the blob and the ledger row are on disk.
    """
    params_hash, _ = store.record_params(conn, spec.key, params)
    discharges = load_discharges()

    def emit(message):
        if log is not None:
            log(message)

    total = len(targets)
    emit(
        f"{_now_label()} {spec.key}: {total} targets,"
        f" params {params_hash[:12]}, force={force}"
    )

    stats = PrecomputeStats(plot=spec.key)
    started = time.perf_counter()
    for index, target in enumerate(targets, start=1):
        prefix = f"{_now_label()} [{index}/{total}] {target.label}"
        stats.considered += 1
        existing = store.find_run(conn, target, spec.key, params_hash)
        if (
            existing is not None
            and existing["status"] == "ok"
            and not force
            and existing["blob_path"]
            and os.path.exists(existing["blob_path"])
        ):
            stats.cached += 1
            emit(f"{prefix}: cached, skipping")
            continue
        if existing is not None and existing["status"] == "failed" and not force:
            stats.failed += 1
            emit(
                f"{prefix}: previous failure recorded, skipping (--force to retry):"
                f" {_one_line(existing['error'] or '')[:200]}"
            )
            continue
        if force and existing is not None:
            try:
                store.delete_run(conn, existing)
            except OSError as error:
                if not _is_infra_error(error):
                    raise
                stats.failed += 1
                emit(
                    f"{prefix}: cannot clear the previous result:"
                    f" {_one_line(f'{type(error).__name__}: {error}')}"
                    f" ({_permission_hint()})"
                )
                continue

        writable, parent = _output_writable(spec, target, params_hash)
        if not writable:
            stats.failed += 1
            emit(
                f"{prefix}: not computing -- {parent} is not writable by this"
                f" user ({_permission_hint()})"
            )
            continue

        emit(f"{prefix}: computing…")
        target_started = time.perf_counter()
        try:
            with xr.open_dataset(target.path) as ds:
                t_start, t_end, _ = loader.time_window(ds, discharges.get(target.shot))
                windowed = (
                    loader.sliced(ds, t_start, t_end)
                    if loader.TIME_DIM in ds.dims
                    else ds
                )
                _, run_row = store.result(conn, spec, target, params, windowed)
        except Exception as error:  # noqa: BLE001 - recorded through the ledger
            # The file was removed since rescan, is unreadable, or is not a
            # dataset at all (a corrupt file raises ValueError out of
            # xr.open_dataset, not OSError). Record it and keep going, so one
            # bad file cannot abort the overnight fill -- unless it is the
            # environment that is broken, which is reported and left
            # unrecorded (see _is_infra_error).
            if _is_infra_error(error):
                stats.failed += 1
                emit(
                    f"{prefix}: failed without recording a run:"
                    f" {_one_line(f'{type(error).__name__}: {error}')}"
                    f" ({_permission_hint()})"
                )
                continue
            store.record_run(
                conn,
                target,
                spec.key,
                params_hash,
                blob_path=None,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
            stats.failed += 1
            emit(
                f"{prefix}: failed after"
                f" {time.perf_counter() - target_started:.1f}s:"
                f" {_one_line(f'{type(error).__name__}: {error}')}"
            )
            continue

        elapsed = time.perf_counter() - target_started
        if run_row is not None and run_row["status"] == "failed":
            stats.failed += 1
            emit(f"{prefix}: failed after {elapsed:.1f}s: {_one_line(run_row['error'])}")
        else:
            stats.computed += 1
            emit(f"{prefix}: ok in {elapsed:.1f}s")

    stats.seconds = time.perf_counter() - started
    return stats
