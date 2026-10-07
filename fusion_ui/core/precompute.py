"""The ``fusion-ui precompute`` engine: fill the cache in batch, in parallel.

The single-shot view computes on demand and caches forever, so the first person
to want a quantity pays its full cost (half a minute for 2DCA, most of an hour
for a shot's bank of averages). ``precompute`` walks the shot index instead and
runs the plots' ``compute`` on every matching shot ahead of time, so the views
-- and everyone else -- read warm caches. It is the same ``store.result`` call
the page makes, driven from a list of targets instead of a widget.

How a fill is organised:

- **Several plots run per target, in dependency order** (:func:`plan_fill`),
  so one process handles a shot end to end: an upstream named alongside the
  plots built on it is computed first, and once.
- **Cache hits are skipped from the ledger, without opening the data file**:
  a single APD record is ~500 MB, so a fill must not re-read what is stored.
- **With ``workers > 1`` a pool of spawned processes** takes the targets left
  to compute, largest file first, so the last shots do not start alone. Each
  worker opens its own database connection, loads the windowed record into
  memory once per target, and prints its own timestamped lines. One worker
  (the default) runs in this process, exactly as a fill always has.
- **``--stale`` recomputes stale results in place** (:func:`plan_stale`), each
  with the parameters it was computed with, upstream before downstream. A
  result whose upstream is stale too, and not named, is left alone with the
  command that fixes it: recomputed on a stale upstream it would still be
  stale, and a quiet "ok" would hide that.
- **Ctrl-C stops cleanly**: targets not started are cancelled, and a target
  in flight is abandoned only while it reads or computes, never while the
  store writes, so it leaves no ``failed`` row, no half-written blob and no
  temporary file (see :class:`_Interrupts`).
"""

import concurrent.futures
import contextlib
import dataclasses
import errno
import functools
import glob
import importlib
import json
import multiprocessing
import os
import signal
import sqlite3
import sys
import threading
import time
import types
from dataclasses import dataclass, field

import xarray as xr

from fusion_ui import config
from fusion_ui.core import (
    catalog,
    db,
    loader,
    multipixel,
    params_ui,
    registry,
    shared,
    store,
)

#: Outcomes of one plot on one target.
COMPUTED, CACHED, FAILED, SKIPPED, CANCELLED = (
    "computed",
    "cached",
    "failed",
    "skipped",
    "cancelled",
)

#: Niceness of the workers when there are several and ``nice`` is not given:
#: the batch shares the server with the service.
DEFAULT_NICE = 10

#: How long a worker waits for another writer's lock before failing a write.
#: Several workers write the ledger at once; each write is a short transaction.
WORKER_BUSY_TIMEOUT_MS = 60_000

#: What a worker imports to register the specs it runs. A spawned worker is a
#: fresh interpreter: it knows a spec only once the module that registers it
#: has been imported, and ``fusion_ui.plots`` registers every spec of the app.
WORKER_MODULES = ("fusion_ui.plots",)

#: The configuration a worker must see as its parent does. ``config`` reads
#: ``.env`` files on import, which would otherwise fill in what the parent's
#: environment left unset.
_CONFIG_VARIABLES = (
    "FUSION_DISCHARGE_DB",
    "FUSION_DATA_FOLDER",
    "FUSION_UI_DB",
    "FUSION_UI_CACHE",
    "FUSION_MACHINE",
)


# ---------------------------------------------------------------------------
# What a fill reports
# ---------------------------------------------------------------------------


@dataclass
class PrecomputeStats:
    plot: str
    considered: int = 0  # targets offered (stale runs, with --stale)
    cached: int = 0  # already ok, skipped without opening the file
    computed: int = 0  # newly computed ok (cache miss, --force or --stale)
    failed: int = 0  # recorded as a failed run, or failed without recording
    seconds: float = 0.0
    skipped: int = 0  # left alone, with the command that would fix it
    cancelled: int = 0  # never run: Ctrl-C, or a worker died
    unit: str = "shots"

    def summary(self, timed=True):
        parts = [
            f"{self.plot}: {self.considered} {self.unit}",
            f"{self.computed} computed",
            f"{self.cached} cached",
        ]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.cancelled:
            parts.append(f"{self.cancelled} cancelled")
        text = ", ".join(parts)
        return text + f" in {self.seconds:.1f}s" if timed else text


@dataclass
class Report:
    """How a fill went: one :class:`PrecomputeStats` per plot, in run order."""

    stats: list
    targets: int = 0
    workers: int = 1
    seconds: float = 0.0
    interrupted: bool = False  # Ctrl-C
    broken: str = ""  # why the pool broke, when a worker died
    killed: bool = False  # a second Ctrl-C killed workers mid-target

    def summary_lines(self):
        """The lines a fill ends with.

        One plot: its summary, timed, exactly as a fill has always ended. Several:
        one untimed line per plot, then the wall time of the whole fill, since
        the plots of one target share it.
        """
        if len(self.stats) == 1:
            lines = [self.stats[0].summary()]
        else:
            on = f" on {self.workers} workers" if self.workers > 1 else ""
            lines = [stats.summary(timed=False) for stats in self.stats]
            lines.append(f"{_count(self.targets, 'target')} in {self.seconds:.1f}s{on}")
        if self.broken:
            lines.append(
                f"stopped: a worker died ({self.broken}). Results already written"
                " are kept; rerun the same command to continue."
            )
        elif self.killed:
            lines.append(
                "killed: workers were stopped mid-target. A result they were"
                " writing may be incomplete; recompute the targets named above"
                " with --force."
            )
        elif self.interrupted:
            lines.append(
                "interrupted: nothing half-written was kept. Rerun the same"
                " command to continue; what is cached is skipped."
            )
        return lines


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def load_discharges():
    """``{shot: PlasmaDischarge}`` from the read-only descriptor, or ``{}``."""
    try:
        path = config.DISCHARGE_DB_PATH
    except RuntimeError:
        return {}
    if not os.path.exists(path):
        return {}
    return catalog.load_discharges(path)


def run_day(shot):
    """The run day of a C-Mod shot number ``1YYMMDDnnn``: its first seven digits."""
    return int(shot) // 1000


def _selected(shot, shots, run_days):
    """Whether ``shot`` is in the selection; ``--shot`` and ``--run-day`` add up."""
    if shots is None and run_days is None:
        return True
    return (shots is not None and shot in shots) or (
        run_days is not None and run_day(shot) in run_days
    )


def _accepts(spec, diagnostic, preprocessed):
    return diagnostic in spec.diagnostics and spec.accepts(preprocessed)


def _target(machine, row):
    return registry.Target(
        machine=machine,
        shot=row["shot"],
        diagnostic=row["diagnostic"],
        preprocessed=bool(row["preprocessed"]),
        path=row["path"],
        t_start=float("nan"),
        t_end=float("nan"),
        window_source="none",
    )


def select_targets(conn, specs, machine=None, shots=None, run_days=None):
    """One :class:`~fusion_ui.core.registry.Target` per indexed file any of
    ``specs`` accepts.

    ``shots`` and ``run_days`` are optional sets of shot numbers and of
    seven-digit run days; a file is selected when either names it, and every
    file when neither is given. ``machine`` defaults to ``config.MACHINE``. The
    time window is left ``NaN`` here -- it is derived from the descriptor (or
    the record itself) when the file is opened, exactly as the single-shot page
    does.

    The order is fixed by the query, not left to the query plan: a fill prints
    one line per shot and is watched (and resumed) by shot number, so two runs
    over the same index must walk it the same way.
    """
    machine = machine or config.MACHINE
    rows = conn.execute(
        "SELECT shot, diagnostic, preprocessed, path FROM shots WHERE machine = ?"
        " ORDER BY shot, diagnostic, preprocessed",
        (machine,),
    ).fetchall()
    return [
        _target(machine, row)
        for row in rows
        if _selected(row["shot"], shots, run_days)
        and any(_accepts(s, row["diagnostic"], row["preprocessed"]) for s in specs)
    ]


def targets_for(conn, spec, machine=None, shots=None, run_days=None):
    """:func:`select_targets` for one spec."""
    return select_targets(conn, [spec], machine, shots, run_days)


def in_dependency_order(specs):
    """``specs`` ordered so that each runs after every named spec it is built on.

    Sorted by the length of the ``requires`` chain, which an upstream's is
    always shorter than: ties keep the order given.
    """
    return sorted(specs, key=lambda spec: len(registry.chain(spec)))


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


# ---------------------------------------------------------------------------
# Parameters: the defaults, or a --params-json file
# ---------------------------------------------------------------------------

#: The file name :func:`command` puts after ``--params-json``; the page shows
#: the parameter set to save under it.
PARAMS_FILE = "params.json"


def command(spec, target, params=None, *flags):
    """The command line that fills ``spec`` for ``target`` with ``params``.

    What a page shows in place of a compute it must not start, a batch-only
    one. ``--shot`` names the shot; ``--machine`` is added when the target is
    not on the configured machine, and ``--params-json`` (see
    :data:`PARAMS_FILE`) when ``params`` are not the spec's defaults. ``flags``
    are appended as given, e.g. ``"--retry-failed"`` or ``"--force"``.
    """
    parts = ["fusion-ui", "precompute", spec.key, "--shot", str(target.shot)]
    if target.machine != config.MACHINE:
        parts += ["--machine", target.machine]
    if params is not None and not is_default(spec, params):
        parts += ["--params-json", PARAMS_FILE]
    return " ".join([*parts, *flags])


def is_default(spec, params):
    """Whether ``params`` hash the same as ``spec``'s default parameter set."""
    digest, _ = params_ui.hash_params(spec.key, params)
    return digest == params_ui.hash_params(spec.key, spec.params())[0]


class ParamsError(ValueError):
    """A ``--params-json`` file that cannot be applied as given.

    Raised before anything is computed: a parameter set that is ambiguous, or
    does not fit a plot named, must not become a batch of results nobody asked
    for.
    """


def _sets_reference_pixel(values):
    """Dotted paths of every ``refx``/``refy`` in a partial values tree."""
    found = []

    def walk(tree, prefix):
        for name, value in tree.items():
            path = f"{prefix}.{name}" if prefix else name
            if name in ("refx", "refy"):
                found.append(path)
            elif isinstance(value, dict):
                walk(value, path)

    walk(values, "")
    return found


def params_from_file(path, specs, pixel=None):
    """``{plot key: params}`` for ``specs`` from a ``--params-json`` file.

    The file holds one of two things.

    **A complete parameter set**, as ``param_sets.params_json`` stores it and
    the single-shot page shows it (``{"plot", "params"}``), or the bare
    ``{"__type__", "values"}`` inside it. It names one plot and fixes every
    knob, so it is used only when that plot is the one plot named: it says
    nothing about any other, and guessing would compute results nobody asked
    for. ``--pixel`` cannot change it.

    **The fields to change**, as part of a canonical ``values`` tree --
    ``{"averages": {"window": 30}}`` -- applied to every plot named, on top of
    its defaults (and ``--pixel``). Every path must exist in every plot named:
    a change that fits one plot and not another would leave the two computed
    with different settings, so it is refused, as is a ``refx``/``refy`` that
    ``--pixel`` also sets.

    Either way each leaf is checked strictly (see
    :func:`~fusion_ui.core.params_ui.from_canonical`), and the parameters come
    back as instances, so what the store records is ``hash_params`` of the
    parameters actually used -- which is how the pages find the results.
    """
    try:
        with open(path, encoding="utf-8-sig") as handle:
            body = json.load(handle)
    except OSError as error:
        raise ParamsError(f"--params-json {path}: {error.strerror or error}") from None
    except ValueError as error:
        raise ParamsError(f"--params-json {path}: not valid JSON ({error})") from None
    if not isinstance(body, dict):
        raise ParamsError(
            f"--params-json {path}: expected a JSON object, got {type(body).__name__}"
        )

    names = ", ".join(spec.key for spec in specs)
    if "plot" in body or "params" in body or "__type__" in body:
        what = body.get("plot") or body.get("__type__") or "one plot"
        if len(specs) != 1:
            raise ParamsError(
                f"--params-json {path} is the complete parameter set of {what},"
                f" which says nothing about the other plots named ({names}). Name"
                " that plot alone, or write only the fields to change (a partial"
                ' values tree such as {"averages": {"window": 30}}), which is'
                " applied to every plot named."
            )
        (spec,) = specs
        if "plot" in body and body["plot"] != spec.key:
            raise ParamsError(
                f"--params-json {path} is the parameter set of plot"
                f" {body['plot']!r}; the plot named is {spec.key!r}"
            )
        if pixel is not None:
            raise ParamsError(
                f"--pixel cannot change the complete parameter set in {path};"
                " set refx/refy there instead"
            )
        try:
            params = params_ui.from_canonical(spec.params, body, plot_key=spec.key)
        except (TypeError, ValueError) as error:
            raise ParamsError(f"--params-json {path}: {error}") from None
        return {spec.key: params}

    if pixel is not None and _sets_reference_pixel(body):
        raise ParamsError(
            f"--pixel and {path} both set the reference pixel"
            f" ({', '.join(_sets_reference_pixel(body))}); set it in one place"
        )
    out = {}
    for spec in specs:
        try:
            out[spec.key] = params_ui.with_values(default_params(spec, pixel), body)
        except (TypeError, ValueError) as error:
            several = (
                " Every plot named takes the whole file: run plots that need"
                " different changes separately."
                if len(specs) > 1
                else ""
            )
            raise ParamsError(
                f"--params-json {path} does not fit {spec.key}: {error}.{several}"
            ) from None
    return out


# ---------------------------------------------------------------------------
# Plans: what to run on which target, decided before anything runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One plot with one parameter set, on the target of its :class:`Job`."""

    plot: str
    params: object
    params_hash: str
    #: With ``--stale``: why the stored result is stale. It is recomputed in
    #: place, cached or not.
    stale: str = ""
    #: Set when planning decided to leave it alone: the line saying why.
    skip: str = ""


@dataclass(frozen=True)
class Job:
    """Everything one target needs, in the order it runs."""

    target: registry.Target
    index: int  # the target's place in the selection: [index/total]
    steps: tuple
    #: What ``loader.time_window`` reads off a discharge (``t_start``,
    #: ``t_end``), or ``None`` when the descriptor has no entry.
    window: object = None
    size: int = 0  # bytes, as the index has them


@dataclass(frozen=True)
class Options:
    """What every step of a fill shares; small, since each job carries it."""

    total: int
    force: bool = False
    retry_failed: bool = False
    multi: bool = False  # several plots: name the plot on each line
    hashed: frozenset = frozenset()  # plots run with several parameter sets


@dataclass
class Plan:
    jobs: list  # in shot order
    stats: dict  # plot -> PrecomputeStats, in run order
    headers: list  # the lines printed first, without their timestamps
    options: Options = field(default_factory=lambda: Options(total=0))
    #: With --stale: upstreams, not named, whose staleness left results alone.
    unnamed_upstreams: list = field(default_factory=list)


def _window(discharge):
    if discharge is None:
        return None
    return types.SimpleNamespace(t_start=discharge.t_start, t_end=discharge.t_end)


def _sizes(conn, machine):
    return {
        (row["shot"], row["diagnostic"], bool(row["preprocessed"])): row["bytes"] or 0
        for row in conn.execute(
            "SELECT shot, diagnostic, preprocessed, bytes FROM shots WHERE machine = ?",
            (machine,),
        )
    }


def _size(sizes, target):
    return sizes.get((target.shot, target.diagnostic, target.preprocessed), 0)


def plan_fill(conn, plots, targets, force=False, retry_failed=False):
    """A :class:`Plan` computing every plot in ``plots`` on the targets it accepts.

    ``plots`` is ``[(spec, params), …]`` in dependency order (see
    :func:`in_dependency_order`); ``targets`` come from :func:`select_targets`.
    Records each parameter set in ``param_sets`` once, so the header can name
    its hash. Opens no file.
    """
    discharges = load_discharges()
    recorded = [
        (spec, params, store.record_params(conn, spec.key, params)[0])
        for spec, params in plots
    ]
    stats = {spec.key: PrecomputeStats(plot=spec.key) for spec, _, _ in recorded}
    sizes = _sizes(conn, targets[0].machine) if targets else {}
    jobs = []
    for index, target in enumerate(targets, start=1):
        steps = tuple(
            Step(spec.key, params, digest)
            for spec, params, digest in recorded
            if _accepts(spec, target.diagnostic, target.preprocessed)
        )
        for step in steps:
            stats[step.plot].considered += 1
        if steps:
            jobs.append(
                Job(
                    target=target,
                    index=index,
                    steps=steps,
                    window=_window(discharges.get(target.shot)),
                    size=_size(sizes, target),
                )
            )
    headers = [
        f"{spec.key}: {stats[spec.key].considered} targets,"
        f" params {digest[:12]}, force={force} retry_failed={retry_failed}"
        for spec, _, digest in recorded
    ]
    options = Options(
        total=len(targets),
        force=force,
        retry_failed=retry_failed,
        multi=len(recorded) > 1,
    )
    return Plan(jobs=jobs, stats=stats, headers=headers, options=options)


def _settled(run):
    """Whether ``run`` is resolved from the ledger alone: cached, or failed."""
    return run is not None and (
        run["status"] == "failed"
        or (bool(run["blob_path"]) and os.path.exists(run["blob_path"]))
    )


def _stale_beneath(conn, spec, target, params, stale_by_id, scheduled):
    """``[(link, reason), …]``: stale links below ``spec`` this fill leaves stale.

    Walks the chain as the store resolves it: a link it would compute (no row,
    or an ``ok`` row whose blob is gone) is passed through to the link below;
    a current one ends the walk, since nothing beneath it is resolved; one
    this fill recomputes first ends it too. A stale one that is not scheduled
    is listed, and the walk goes on, so that the fix names every stale link.
    """
    blockers = []
    links = registry.chain(spec)
    current = params
    for upper, link in zip(links, links[1:]):
        current = upper.upstream_params(current)
        digest, _ = params_ui.hash_params(link.key, current)
        run = store.find_run(conn, target, link.key, digest)
        if run is None or not _settled(run):
            continue  # computed on the way, from whatever is beneath it
        if run["id"] in scheduled or run["id"] not in stale_by_id:
            break
        blockers.append((link, stale_by_id[run["id"]]["stale"]))
    return blockers


def _with_upstreams(spec, blockers, named):
    """The plots a ``--stale`` must name to make ``spec`` current: every link
    from the deepest stale one up, and the plots already named, in
    dependency order."""
    links = registry.chain(spec)
    deepest = max(links.index(link) for link, _ in blockers)
    keys = [link.key for link in reversed(links[: deepest + 1])]
    keys = list(dict.fromkeys([*keys, *named]))
    return [s.key for s in in_dependency_order([registry.get(k) for k in keys])]


def _fix_command(plots, target):
    parts = ["fusion-ui", "precompute", *plots, "--stale", "--shot", str(target.shot)]
    if target.machine != config.MACHINE:
        parts += ["--machine", target.machine]
    return " ".join(parts)


def plan_stale(conn, specs, machine=None, shots=None, run_days=None, only=None):
    """A :class:`Plan` recomputing in place the stale runs of ``specs``.

    ``specs`` are in dependency order. Every stale run of theirs on the
    selected targets is taken, of any status and any parameter set -- or only
    the parameter sets in ``only`` (``{plot: params}``), when given -- and
    recomputed with the parameters it was stored with, rebuilt from
    ``param_sets``.

    A run is planned as a skip, with a line saying why, when

    - its stored parameters no longer rebuild to the same hash (the params
      class changed since): nothing could recompute *that* set;
    - a link beneath it is stale and not recomputed first in this fill (see
      :func:`_stale_beneath`): recomputed on a stale upstream it would still
      be stale, so the line names the command that recomputes the chain.

    Reads the registry, so import :mod:`fusion_ui.plots` first. Opens no file.
    """
    machine = machine or config.MACHINE
    order = {spec.key: position for position, spec in enumerate(specs)}
    wanted = {
        key: params_ui.hash_params(key, params)[0]
        for key, params in (only or {}).items()
    }
    stale = store.stale_runs(conn)
    stale_by_id = {row["id"]: row for row in stale}
    candidates = sorted(
        (
            row
            for row in stale
            if row["plot"] in order
            and row["machine"] == machine
            and _selected(row["shot"], shots, run_days)
            and (row["plot"] not in wanted or row["params_hash"] == wanted[row["plot"]])
        ),
        key=lambda row: (order[row["plot"]], row["id"]),
    )
    index = {
        (row["shot"], row["diagnostic"], bool(row["preprocessed"])): row
        for row in conn.execute(
            "SELECT shot, diagnostic, preprocessed, path, bytes FROM shots"
            " WHERE machine = ?",
            (machine,),
        )
    }
    discharges = load_discharges()
    stats = {spec.key: PrecomputeStats(plot=spec.key, unit="stale") for spec in specs}
    reasons = {spec.key: {} for spec in specs}
    hashes = {spec.key: set() for spec in specs}
    scheduled = set()
    unnamed = set()  # stale upstreams that would have had to be named
    by_target = {}
    for row in candidates:
        spec = registry.get(row["plot"])
        key = (row["shot"], row["diagnostic"], bool(row["preprocessed"]))
        if key not in index:  # stale_runs judges indexed inputs only
            continue
        target = _target(machine, index[key])
        stats[spec.key].considered += 1
        reasons[spec.key][row["stale"]] = reasons[spec.key].get(row["stale"], 0) + 1
        hashes[spec.key].add(row["params_hash"])
        skip, params = "", None
        try:
            params = params_ui.from_canonical(
                spec.params,
                store.params_json(conn, row["params_hash"]) or "",
                plot_key=spec.key,
            )
            if params_ui.hash_params(spec.key, params)[0] != row["params_hash"]:
                raise ValueError("it no longer hashes as it was stored")
        except (TypeError, ValueError) as error:
            skip = (
                f"not recomputed: its stored parameter set"
                f" {row['params_hash'][:12]} no longer fits"
                f" {spec.params.__name__} ({error})"
            )
        if not skip:
            blockers = _stale_beneath(
                conn, spec, target, params, stale_by_id, scheduled
            )
            if blockers:
                link, reason = blockers[0]
                plots = _with_upstreams(spec, blockers, order)
                unnamed.update(plot for plot in plots if plot not in order)
                skip = (
                    f"not recomputed: stale ({row['stale']}), and so is its"
                    f" upstream {link.key} ({reason}); recomputed on a stale"
                    " upstream it would still be stale. Run"
                    f" `{_fix_command(plots, target)}`"
                )
        if not skip:
            scheduled.add(row["id"])
        by_target.setdefault(key, []).append(
            Step(spec.key, params, row["params_hash"], stale=row["stale"], skip=skip)
        )

    jobs = []
    for position, key in enumerate(sorted(by_target), start=1):
        target = _target(machine, index[key])
        jobs.append(
            Job(
                target=target,
                index=position,
                steps=tuple(by_target[key]),
                window=_window(discharges.get(target.shot)),
                size=index[key]["bytes"] or 0,
            )
        )
    headers = []
    for spec in specs:
        counts = reasons[spec.key]
        if not counts:
            headers.append(f"{spec.key}: nothing stale")
            continue
        why = ", ".join(f"{reason} {n}" for reason, n in sorted(counts.items()))
        headers.append(
            f"{spec.key}: {stats[spec.key].considered} stale ({why}),"
            " recomputed in place"
        )
    options = Options(
        total=len(jobs),
        multi=len(specs) > 1,
        hashed=frozenset(key for key, seen in hashes.items() if len(seen) > 1),
    )
    plan = Plan(jobs=jobs, stats=stats, headers=headers, options=options)
    plan.unnamed_upstreams = [
        s.key for s in in_dependency_order([registry.get(k) for k in unnamed])
    ]
    return plan


# ---------------------------------------------------------------------------
# Lines and errors
# ---------------------------------------------------------------------------


def _now_label():
    # Wall-clock prefix for overnight-log reading: tailing the log shows not
    # just which shot is in flight but since when.
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _count(n, noun):
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _one_line(message):
    return str(message).replace("\n", " | ")


def _prefix(job, step, options, tag=""):
    """``<time> [<tag>] [i/N] <target>[ · <plot>][ params <hash>]``.

    The plot is named only when several run, and the hash only when one plot
    runs with several parameter sets, so a single plot's lines read as they
    always have.
    """
    parts = []
    if options.multi:
        parts.append(step.plot)
    if step.plot in options.hashed:
        parts.append(f"params {step.params_hash[:12]}")
    what = f" · {' '.join(parts)}" if parts else ""
    who = f" {tag}" if tag else ""
    return f"{_now_label()}{who} [{job.index}/{options.total}] {job.target.label}{what}"


#: Errnos that mean "the setup is wrong", not "the analysis failed": a cache
#: directory the other writer owns, a read-only mount, a missing group.
_INFRA_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


def _is_infra_error(error):
    """Whether ``error`` is the environment, not the analysis.

    An infrastructure failure must not become a ``failed`` ledger row: it would
    be skipped without ``--force`` ever after, although nothing about the shot
    or the parameters is wrong. A database error is one too -- a lock held past
    the busy timeout, a full disk -- and recording it would mostly fail anyway.
    """
    return (
        isinstance(error, PermissionError)
        or (isinstance(error, OSError) and error.errno in _INFRA_ERRNOS)
        or isinstance(error, sqlite3.Error)
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


def _infra_hint(error):
    if isinstance(error, sqlite3.Error):
        return (
            "the app database refused the write: another process may hold its"
            " lock, or the disk or the file is not writable. Nothing was"
            " recorded, so rerunning is enough once that is fixed"
        )
    return _permission_hint()


def _error_text(error):
    return _one_line(f"{type(error).__name__}: {error}")


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


def _chain_with_params(spec, params):
    """``[(link, link params, link hash), …]`` down ``spec``'s chain."""
    out = []
    links = registry.chain(spec)
    current = params
    for position, link in enumerate(links):
        if position:
            current = links[position - 1].upstream_params(current)
        out.append((link, current, params_ui.hash_params(link.key, current)[0]))
    return out


def _unwritable(conn, spec, target, params):
    """The first blob directory this step would write and cannot, or ``""``.

    The step's own, and each upstream's that would be computed on the way:
    the walk stops at a link the ledger resolves (cached, or failed), as the
    store's does. An upstream bank that took an hour must not end in a
    ``PermissionError`` -- and a ``failed`` row -- on its save either.
    """
    for position, (link, _, digest) in enumerate(_chain_with_params(spec, params)):
        if position and _settled(store.find_run(conn, target, link.key, digest)):
            break
        writable, parent = _output_writable(link, target, digest)
        if not writable:
            return parent
    return ""


# ---------------------------------------------------------------------------
# Ctrl-C
# ---------------------------------------------------------------------------


class _Interrupts:
    """Ctrl-C, held back while the store writes.

    A SIGINT sets :attr:`pending`. It raises ``KeyboardInterrupt`` at once only
    inside :meth:`interruptible` -- opening and loading the input, and a spec's
    compute, where stopping loses nothing. Anywhere else the store may be part
    way through a write (the blob renamed into place, then the run row, then
    its scalars, each committed on its own), so the interruption waits for the
    next :meth:`check`, which comes before anything new starts. A compute that
    raised because it was interrupted counts as interrupted, never as a failed
    run.

    Installed only on the main thread, where Python delivers signals, and
    never over an ignored SIGINT (a fill started in the background of a
    script must stay deaf to the terminal's Ctrl-C); otherwise Ctrl-C behaves
    as it always does.
    """

    def __init__(self):
        self.pending = False
        self.armed = False
        self._installed = False
        self._previous = None

    def _handle(self, signum, frame):
        self.pending = True
        if self.armed:
            self.armed = False
            raise KeyboardInterrupt

    def install(self):
        if (
            threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGINT) is not signal.SIG_IGN
        ):
            self._previous = signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        return self

    def restore(self):
        if self._installed:
            signal.signal(signal.SIGINT, self._previous)
            self._installed = False

    def check(self):
        if self.pending:
            raise KeyboardInterrupt

    @contextlib.contextmanager
    def interruptible(self):
        self.check()
        self.armed = True
        try:
            yield
        except Exception:
            if self.pending:
                raise KeyboardInterrupt from None
            raise
        finally:
            self.armed = False
        self.check()


def _interruptible_compute(compute, interrupts):
    @functools.wraps(compute)
    def compute_until_interrupted(*args, **kwargs):
        with interrupts.interruptible():
            return compute(*args, **kwargs)

    return compute_until_interrupted


def _wrap_registry(interrupts):
    """Wrap every cached spec's compute in :meth:`_Interrupts.interruptible`.

    Through the registry, since that is where the store finds an upstream.
    Returns ``{key: (original, wrapped)}``.
    """
    swapped = {}
    for key, spec in list(registry.REGISTRY.items()):
        if spec.cached:
            wrapped = dataclasses.replace(
                spec, compute=_interruptible_compute(spec.compute, interrupts)
            )
            registry.REGISTRY[key] = wrapped
            swapped[key] = (spec, wrapped)
    return swapped


@contextlib.contextmanager
def _interruptible_registry(interrupts):
    swapped = _wrap_registry(interrupts)
    try:
        yield
    finally:
        for key, (original, wrapped) in swapped.items():
            if registry.REGISTRY.get(key) is wrapped:
                registry.REGISTRY[key] = original


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------


class _Record:
    """The target's input, opened at the first step that computes and shared
    by the rest; loaded into memory when ``load`` is set (pool workers)."""

    def __init__(self, job, load, interrupts):
        self._job = job
        self._load = load
        self._interrupts = interrupts
        self._stack = contextlib.ExitStack()
        self._data = None
        self._error = None

    def get(self):
        if self._error is not None:
            raise self._error
        if self._data is None:
            try:
                with self._interrupts.interruptible():
                    ds = self._stack.enter_context(
                        xr.open_dataset(self._job.target.path)
                    )
                    t_start, t_end, _ = loader.time_window(ds, self._job.window)
                    windowed = (
                        loader.sliced(ds, t_start, t_end)
                        if loader.TIME_DIM in ds.dims
                        else ds
                    )
                    self._data = windowed.load() if self._load else windowed
            except Exception as error:  # noqa: BLE001 - the steps record it
                self._error = error
                raise
        return self._data

    def close(self):
        self._data = None
        self._stack.close()


def _from_ledger(conn, target, step, options, prefix, emit):
    """The outcome the ledger alone settles for ``step``, or ``None`` to compute.

    A cache hit needs an ``ok`` run *and* its blob on disk: the ledger alone is
    not enough, because a cleared cache directory would otherwise make a fill
    skip a result it was asked to warm. A ``failed`` run is skipped without
    reopening its file, because ``store.result`` would only hand the same
    failure back -- unless ``--force`` or ``--retry-failed`` say otherwise.
    """
    if step.skip:
        emit(f"{prefix}: {step.skip}")
        return SKIPPED
    if step.stale:
        return None
    existing = store.find_run(conn, target, step.plot, step.params_hash)
    if existing is None or options.force:
        return None
    if existing["status"] == "ok":
        if existing["blob_path"] and os.path.exists(existing["blob_path"]):
            emit(f"{prefix}: cached, skipping")
            return CACHED
        return None
    if existing["status"] == "failed" and not options.retry_failed:
        emit(
            f"{prefix}: previous failure recorded, skipping"
            " (--force or --retry-failed to retry):"
            f" {_one_line(existing['error'] or '')[:200]}"
        )
        return FAILED
    return None


def _clear(conn, run, prefix, emit):
    """Drop ``run`` before recomputing it; ``False`` if the environment refused."""
    try:
        store.delete_run(conn, run)
    except OSError as error:
        if not _is_infra_error(error):
            raise
        emit(
            f"{prefix}: cannot clear the previous result: {_error_text(error)}"
            f" ({_permission_hint()})"
        )
        return False
    return True


def _failed(conn, target, step, prefix, emit, error, started):
    """Report an error that escaped the store; record it unless infrastructure.

    The file was removed since rescan, is unreadable, or is not a dataset at
    all (a corrupt file raises ValueError out of ``xr.open_dataset``, not
    OSError). Recorded, so one bad file cannot abort the fill -- unless it is
    the environment that is broken, which is reported and left unrecorded (see
    :func:`_is_infra_error`).
    """
    if _is_infra_error(error):
        emit(
            f"{prefix}: failed without recording a run: {_error_text(error)}"
            f" ({_infra_hint(error)})"
        )
        return FAILED
    try:
        # Every column is set explicitly so a retried `ok` row does not keep
        # stale timing/version values under its new `failed` status.
        store.record_run(
            conn,
            target,
            step.plot,
            step.params_hash,
            blob_path=None,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            seconds=None,
            code_version=store._code_version(),
        )
    except sqlite3.Error as db_error:
        emit(
            f"{prefix}: failed without recording a run: {_error_text(error)};"
            f" recording it failed too: {_error_text(db_error)}"
            f" ({_infra_hint(db_error)})"
        )
        return FAILED
    emit(
        f"{prefix}: failed after {time.perf_counter() - started:.1f}s:"
        f" {_error_text(error)}"
    )
    return FAILED


def _run_step(conn, job, step, options, emit, record, tag):
    """One plot on the job's target: the outcome, after logging it."""
    target = job.target
    prefix = _prefix(job, step, options, tag)
    settled = _from_ledger(conn, target, step, options, prefix, emit)
    if settled is not None:
        return settled
    if not step.stale:
        # Retrying a failure, or --force, drops the row first: store.result
        # hands a recorded failure (or a cached result) straight back and
        # would never recompute.
        existing = store.find_run(conn, target, step.plot, step.params_hash)
        if existing is not None and (options.force or existing["status"] == "failed"):
            if not _clear(conn, existing, prefix, emit):
                return FAILED

    spec = registry.get(step.plot)
    unwritable = _unwritable(conn, spec, target, step.params)
    if unwritable:
        emit(
            f"{prefix}: not computing -- {unwritable} is not writable by this"
            f" user ({_permission_hint()})"
        )
        return FAILED

    emit(
        f"{prefix}: recomputing ({step.stale})…"
        if step.stale
        else f"{prefix}: computing…"
    )
    started = time.perf_counter()
    try:
        data = record.get()
        # A batch job: the one caller allowed to compute batch-only specs,
        # including a batch-only upstream of the plot asked for.
        if step.stale:
            _, run = store.compute_and_store(
                conn, spec, target, step.params, data, batch=True
            )
        else:
            _, run = store.result(conn, spec, target, step.params, data, batch=True)
    except Exception as error:  # noqa: BLE001 - recorded through the ledger
        return _failed(conn, target, step, prefix, emit, error, started)

    elapsed = time.perf_counter() - started
    if run is not None and run["status"] == "failed":
        emit(f"{prefix}: failed after {elapsed:.1f}s: {_one_line(run['error'])}")
        return FAILED
    emit(f"{prefix}: ok in {elapsed:.1f}s")
    return COMPUTED


def _run_job(conn, job, options, emit, interrupts, tag="", load=False):
    """Every step of one target, in order: ``[(plot, outcome), …]``.

    Ctrl-C ends the job: the step in flight, if any, is reported interrupted,
    and it and the rest come back ``cancelled``. Commits stay per-step atomic:
    a line saying ``ok`` means the blob and the ledger row are on disk.
    """
    outcomes = []
    record = _Record(job, load, interrupts)
    try:
        for step in job.steps:
            interrupts.check()
            try:
                outcome = _run_step(conn, job, step, options, emit, record, tag)
            except KeyboardInterrupt:
                emit(
                    f"{_prefix(job, step, options, tag)}: interrupted, nothing recorded"
                )
                raise
            outcomes.append((step.plot, outcome))
    except KeyboardInterrupt:
        outcomes += [(step.plot, CANCELLED) for step in job.steps[len(outcomes) :]]
    finally:
        record.close()
    return outcomes


def _tally(stats, outcomes):
    for plot, outcome in outcomes:
        entry = stats[plot]
        setattr(entry, outcome, getattr(entry, outcome) + 1)


# ---------------------------------------------------------------------------
# Running a plan
# ---------------------------------------------------------------------------


def _quiet(log):
    """``log``, never raising: a closed pipe (tee killed by the same Ctrl-C)
    must not turn a finished compute into an error."""

    def emit(message):
        if log is None:
            return
        try:
            log(message)
        except (OSError, ValueError):
            pass

    return emit


def _renice(niceness):
    """Set this process's niceness to ``niceness``: absolute, never added.

    ``nice -n 10 fusion-ui precompute … --workers 7`` then leaves the workers
    at 10, not 20. Every thread is set, since numpy's BLAS threads are started
    at import, before a worker could renice its main thread, and Linux keeps a
    niceness per thread. Lowering needs privileges, so a niceness already
    higher is kept. Returns the niceness now in force.
    """
    try:
        threads = [int(name) for name in os.listdir("/proc/self/task")]
    except OSError:
        threads = [0]
    for thread in threads:
        try:
            if niceness > os.getpriority(os.PRIO_PROCESS, thread):
                os.setpriority(os.PRIO_PROCESS, thread, niceness)
        except OSError:
            pass
    return os.getpriority(os.PRIO_PROCESS, 0)


def execute(conn, plan, log=None, workers=1, nice=None, modules=None):
    """Run ``plan``: in this process, or on a pool of ``workers`` processes.

    ``log`` is called with each line this process writes; pool workers print
    their own lines to standard output, flushed, each with its worker's tag.
    ``nice`` sets the niceness the computes run at (see :func:`_renice`).
    ``modules`` are what a worker imports to register the specs it runs
    (default :data:`WORKER_MODULES`).
    """
    emit = _quiet(log)
    for header in plan.headers:
        emit(f"{_now_label()} {header}")
    report = Report(
        stats=list(plan.stats.values()), targets=plan.options.total, workers=workers
    )
    started = time.perf_counter()
    try:
        if workers > 1:
            _execute_pool(conn, plan, emit, workers, nice, modules, report)
        else:
            _execute_here(conn, plan, emit, nice, report)
    except KeyboardInterrupt:
        # Ctrl-C in this process outside a worker's care: while planning the
        # dispatch, before any job started.
        report.interrupted = True
    finally:
        report.seconds = time.perf_counter() - started
        for stats in report.stats:
            stats.seconds = report.seconds
            done = stats.computed + stats.cached + stats.failed + stats.skipped
            stats.cancelled = stats.considered - done
    return report


def _execute_here(conn, plan, emit, nice, report):
    if nice is not None:
        _renice(nice)
    interrupts = _Interrupts().install()
    try:
        with _interruptible_registry(interrupts):
            for job in plan.jobs:
                interrupts.check()
                _tally(plan.stats, _run_job(conn, job, plan.options, emit, interrupts))
                interrupts.check()
    except KeyboardInterrupt:
        report.interrupted = True
    finally:
        interrupts.restore()


# --- the pool ---------------------------------------------------------------


@dataclass
class _Worker:
    conn: sqlite3.Connection
    interrupts: _Interrupts
    tag: str


#: This process's state, when it is a pool worker.
_WORKER = None


def _worker_connection(database):
    """A connection of a worker's own, waiting out the other workers' writes."""
    conn = db.connect(database)
    conn.execute(f"PRAGMA busy_timeout = {WORKER_BUSY_TIMEOUT_MS:d}")
    return conn


def _init_worker(database, environment, modules, nice, counter):
    """A pool worker's start: the parent's configuration, then the specs."""
    global _WORKER
    # First, so that a Ctrl-C while the imports below run waits for the first
    # job, rather than killing the worker and breaking the pool.
    interrupts = _Interrupts().install()
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if nice is not None:
        _renice(nice)
    for module in modules:
        importlib.import_module(module)
    _wrap_registry(interrupts)
    with counter.get_lock():
        counter.value += 1
        number = counter.value
    _WORKER = _Worker(
        conn=_worker_connection(database), interrupts=interrupts, tag=f"w{number}"
    )


def _print_line(line):
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def _work(job, options):
    """One job, in a pool worker: ``(outcomes, whether Ctrl-C reached it)``.

    A Ctrl-C that arrived while the store wrote let the job finish, so the
    outcomes alone would not show it; the flag does, and the parent stops.
    """
    worker = _WORKER
    missing = sorted(
        {step.plot for step in job.steps if step.plot not in registry.REGISTRY}
    )
    if missing:
        raise RuntimeError(
            f"plot(s) {', '.join(missing)} are not registered in the worker:"
            " pass the module that registers them in `modules`"
        )
    outcomes = _run_job(
        worker.conn,
        job,
        options,
        _print_line,
        worker.interrupts,
        tag=worker.tag,
        load=True,
    )
    return outcomes, worker.interrupts.pending


def _database_file(conn):
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            return row[2]
    raise RuntimeError("a pool of workers needs the app database in a file")


def _processes(pool):
    return list((getattr(pool, "_processes", None) or {}).values())


def largest_first(jobs):
    """``jobs`` by input size, largest first, so the last shots of a fill do
    not start alone; equal sizes keep their order."""
    return sorted(jobs, key=lambda job: -job.size)


def _execute_pool(conn, plan, emit, workers, nice, modules, report):
    """The parent's side: skip from the ledger, dispatch the rest, collect."""
    options = plan.options
    dispatch = []
    for job in plan.jobs:
        remaining = []
        for step in job.steps:
            outcome = _from_ledger(
                conn, job.target, step, options, _prefix(job, step, options), emit
            )
            if outcome is None:
                remaining.append(step)
            else:
                _tally(plan.stats, [(step.plot, outcome)])
        if remaining:
            dispatch.append(dataclasses.replace(job, steps=tuple(remaining)))
    if not dispatch:
        report.workers = 1  # none was started
        return
    dispatch = largest_first(dispatch)
    workers = min(workers, len(dispatch))
    report.workers = workers
    niceness = os.getpriority(os.PRIO_PROCESS, 0)
    if nice is not None:
        niceness = max(nice, niceness)
    emit(
        f"{_now_label()} {len(dispatch)} of {_count(options.total, 'target')} to"
        f" compute on {_count(workers, 'worker')} at nice {niceness}, largest first"
    )

    context = multiprocessing.get_context("spawn")
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            _database_file(conn),
            {name: os.environ.get(name) for name in _CONFIG_VARIABLES},
            tuple(WORKER_MODULES if modules is None else modules),
            nice,
            context.Value("i", 0),
        ),
    )
    futures = {}
    try:
        _collect(pool, dispatch, futures, plan, emit, report)
    except KeyboardInterrupt:
        _kill(pool, futures, emit, report)  # Ctrl-C again, while winding down
    finally:
        try:
            pool.shutdown(wait=not report.killed, cancel_futures=True)
        except KeyboardInterrupt:
            _kill(pool, futures, emit, report)
            pool.shutdown(wait=False, cancel_futures=True)


class _Stop(Exception):
    """A worker was interrupted, or died: stop the rest."""


def _absorb(future, job, plan, emit, report):
    """Count one finished job; ``True`` when its worker was interrupted."""
    try:
        outcomes, interrupted = future.result()
    except concurrent.futures.CancelledError:
        return False
    except concurrent.futures.process.BrokenProcessPool as error:
        report.broken = str(error) or type(error).__name__
        return False
    except Exception as error:  # noqa: BLE001 - the worker's own code failed
        emit(
            f"{_now_label()} [{job.index}/{plan.options.total}] {job.target.label}:"
            f" worker error, nothing recorded: {_error_text(error)}"
        )
        _tally(plan.stats, [(step.plot, FAILED) for step in job.steps])
        return False
    _tally(plan.stats, outcomes)
    return interrupted or any(outcome == CANCELLED for _, outcome in outcomes)


def _collect(pool, dispatch, futures, plan, emit, report):
    """Submit every job and count each as it finishes, until done or stopped.

    ``futures`` (future -> job) is filled as the jobs are submitted.
    """
    pending = set()
    try:
        for job in dispatch:
            future = pool.submit(_work, job, plan.options)
            futures[future] = job
            pending.add(future)
        for future in concurrent.futures.as_completed(futures):
            pending.discard(future)
            interrupted = _absorb(future, futures[future], plan, emit, report)
            if interrupted or report.broken:
                raise _Stop
    except (KeyboardInterrupt, _Stop):
        if not report.broken:
            report.interrupted = True
        _wind_down(pool, futures, pending, plan, emit, report)


def _wind_down(pool, futures, pending, plan, emit, report):
    """Cancel what has not started; let what runs stop at a safe point.

    A future handed to a worker cannot be cancelled, so each worker is sent
    SIGINT too -- it already has it when the Ctrl-C came from the terminal --
    and its job stops as :class:`_Interrupts` allows. A Ctrl-C while this waits
    is left to the caller, which kills the workers (:func:`_kill`).
    """
    if report.broken:
        emit(f"{_now_label()} a worker died ({report.broken}); stopping")
    else:
        emit(
            f"{_now_label()} interrupted: cancelling the targets not started,"
            " waiting for the running ones to stop (Ctrl-C again to kill them)"
        )
    running = [future for future in pending if not future.cancel()]
    if not report.broken:
        for process in _processes(pool):
            with contextlib.suppress(OSError):
                os.kill(process.pid, signal.SIGINT)
    for future in concurrent.futures.as_completed(running):
        _absorb(future, futures[future], plan, emit, report)


def _kill(pool, futures, emit, report):
    """A second Ctrl-C: kill the workers, then sweep away the temporary files
    a blob write they were in could have left."""
    report.killed = report.interrupted = True
    for process in _processes(pool):
        with contextlib.suppress(OSError):
            process.kill()
    unfinished = [job for future, job in futures.items() if not future.done()]
    for job in unfinished:
        _sweep(job)
    names = ", ".join(f"[{job.index}] {job.target.label}" for job in unfinished)
    emit(
        f"{_now_label()} killed the workers; in flight or queued then:"
        f" {names or 'none'}"
    )


def _sweep(job):
    """Remove the temporary files a killed job's blob writes may have left."""
    for step in job.steps:
        if step.params is None:
            continue
        spec = registry.get(step.plot)
        for link, _, digest in _chain_with_params(spec, step.params):
            try:
                path = store.blob_path(link.key, digest, job.target)
            except RuntimeError:
                return
            for leftover in glob.glob(glob.escape(path) + ".tmp.*"):
                with contextlib.suppress(OSError):
                    os.remove(leftover)


# ---------------------------------------------------------------------------
# One plot, one process: the original entry point
# ---------------------------------------------------------------------------


def run(conn, spec, targets, params, force=False, retry_failed=False, log=None):
    """Compute ``spec`` with ``params`` on every target, skipping cache hits.

    A target whose compute raises still gets a ``failed`` row (via the store),
    so a broken shot does not stop the rest of the fill. ``--force``
    recomputes everything; ``--retry-failed`` recomputes only the failed rows
    (e.g. after fixing a full disk or a permissions problem that poisoned the
    ledger).

    ``log``, when given, is called with one line per event -- a header, then
    each target's skip/computing/done line -- so a fill watched through
    ``tail -f`` shows which shot is in flight. :func:`plan_fill` and
    :func:`execute` do the work, and take several plots and workers.
    """
    plan = plan_fill(
        conn, [(spec, params)], targets, force=force, retry_failed=retry_failed
    )
    execute(conn, plan, log=log)
    return plan.stats[spec.key]
