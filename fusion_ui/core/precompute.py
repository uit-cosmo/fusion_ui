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

import dataclasses
import os
import time
from dataclasses import dataclass, field

import xarray as xr

from fusion_ui import config
from fusion_ui.core import catalog, loader, registry, store


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
    """
    machine = machine or config.MACHINE
    rows = conn.execute(
        "SELECT shot, diagnostic, preprocessed, path FROM shots WHERE machine = ?",
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
        _set_pixel(params, pixel[0], pixel[1])
    return params


def _set_pixel(params, x, y):
    """Set ``refx``/``refy`` wherever they appear in the params tree."""
    for f in dataclasses.fields(params):
        value = getattr(params, f.name)
        if dataclasses.is_dataclass(value):
            _set_pixel(value, x, y)
        elif f.name in ("refx", "refy"):
            setattr(params, f.name, x if f.name == "refx" else y)


def run(conn, spec, targets, params, force=False):
    """Compute ``spec`` with ``params`` on every target, skipping cache hits.

    A cache hit needs an ``ok`` run *and* its blob on disk: the ledger alone is
    not enough, because a cleared cache directory would otherwise make a fill
    skip a result it was asked to warm. Detecting it still costs no file open.
    A target whose compute raises still gets a ``failed`` row (via the store),
    so a broken shot does not stop the rest of the fill.
    """
    params_hash, _ = store.record_params(conn, spec.key, params)
    discharges = load_discharges()

    stats = PrecomputeStats(plot=spec.key)
    started = time.perf_counter()
    for target in targets:
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
            continue
        if force and existing is not None:
            store.delete_run(conn, existing)

        with xr.open_dataset(target.path) as ds:
            t_start, t_end, _ = loader.time_window(ds, discharges.get(target.shot))
            windowed = (
                loader.sliced(ds, t_start, t_end) if loader.TIME_DIM in ds.dims else ds
            )
            _, run_row = store.result(conn, spec, target, params, windowed)

        if run_row is not None and run_row["status"] == "failed":
            stats.failed += 1
        else:
            stats.computed += 1

    stats.seconds = time.perf_counter() - started
    return stats
