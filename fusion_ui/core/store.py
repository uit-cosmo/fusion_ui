"""The result store: the run ledger, the netCDF blobs and the scalar rows.

Every result of a cached :class:`~fusion_ui.core.registry.PlotSpec` lands here:
one ``runs`` row per (plot, parameters, target), the dataset as netCDF under
``CACHE_DIR``, and what ``spec.scalars()`` returns in ``scalars``, which the
multi-shot view reads.

- The cache never evicts. Cleanup is deliberate: :func:`delete_run`,
  :func:`prune` and :func:`prune_scalars`.
- ``code_version`` is recorded but is not in the cache key, or every commit
  would invalidate every result.
- A compute that raises is stored as a ``failed`` row with its message, and
  later calls return that row instead of raising again.
- A spec with ``requires`` is resolved depth first and only on a miss. Each
  link keeps its own row and blob.
- A batch-only spec is computed only with ``batch=True``, which ``fusion-ui
  precompute`` passes. Otherwise a miss raises :class:`BatchOnlyError`, and a
  page reads such results with :func:`lookup`.
- A run records its input file's mtime and the upstream run it was built on.
  Neither is in the cache key. :func:`stale_runs` compares them with the index
  and the ledger as they are now.
"""

import contextlib
import math
import os
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
import xarray as xr

from fusion_ui import config
from fusion_ui.core import params_ui, registry, shared, versions

#: The x and y of a scalar that belongs to the shot, not to one pixel. Not NULL:
#: SQLite allows NULLs in a non-INTEGER primary key, which would break the
#: uniqueness the table depends on.
SHOT_LEVEL = -1


def _now():
    # To the microsecond: stale_runs orders a run against its upstream by
    # created_at, and the two can be written within the same second.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@lru_cache(maxsize=None)
def _code_version():
    """Each repository's commit as one string, or ``None``. Read once per process."""
    # A missing checkout must not stop a result being stored.
    try:
        return versions.as_text(versions.code_version())
    except Exception:  # noqa: BLE001
        return None


def blob_path(plot, params_hash, target, suffix=".nc"):
    """``CACHE_DIR/runs/<plot>/<params_hash>/<target key><suffix>``.

    The hash is in the path, so two parameter sets for one shot cannot
    overwrite each other on disk.
    """
    return os.path.join(
        config.CACHE_DIR, "runs", plot, params_hash, f"{target.key}{suffix}"
    )


# ---------------------------------------------------------------------------
# param_sets and runs
# ---------------------------------------------------------------------------


def record_params(conn, plot, params):
    """``(params_hash, params_json)``, inserting the ``param_sets`` row once."""
    digest, text = params_ui.hash_params(plot, params)
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO param_sets (hash, plot, params_json, created_at)"
            " VALUES (?, ?, ?, ?)",
            (digest, plot, text, _now()),
        )
    return digest, text


def params_json(conn, params_hash):
    row = conn.execute(
        "SELECT params_json FROM param_sets WHERE hash = ?", (params_hash,)
    ).fetchone()
    return row["params_json"] if row else None


def find_run(conn, target, plot, params_hash):
    return conn.execute(
        "SELECT * FROM runs WHERE machine = ? AND shot = ? AND diagnostic = ?"
        "   AND preprocessed = ? AND plot = ? AND params_hash = ?",
        (
            target.machine,
            target.shot,
            target.diagnostic,
            int(target.preprocessed),
            plot,
            params_hash,
        ),
    ).fetchone()


def input_mtime(conn, target):
    """The input file's mtime as the index (``shots.mtime``) has it, or ``None``.

    A run stores the same string in ``runs.input_mtime``, so the two compare
    directly. ``None`` for a target that is not indexed.
    """
    row = conn.execute(
        "SELECT mtime FROM shots WHERE machine = ? AND shot = ? AND diagnostic = ?"
        "   AND preprocessed = ?",
        (target.machine, target.shot, target.diagnostic, int(target.preprocessed)),
    ).fetchone()
    return None if row is None else row["mtime"]


def record_run(conn, target, plot, params_hash, **columns):
    """Write the run row and return it, replacing any earlier attempt.

    An earlier attempt is updated in place, keeping its ``id``, and its scalars
    are cleared. ``input_mtime`` and ``upstream_run_id`` are written as ``NULL``
    unless given, so the row never keeps the previous attempt's.
    """
    fields = {
        "machine": target.machine,
        "shot": target.shot,
        "diagnostic": target.diagnostic,
        "preprocessed": int(target.preprocessed),
        "plot": plot,
        "params_hash": params_hash,
        "created_at": _now(),
        "input_mtime": None,
        "upstream_run_id": None,
        **columns,
    }
    names = list(fields)
    updates = ", ".join(f"{n} = excluded.{n}" for n in names if n != "machine")
    with conn:
        conn.execute(
            f"INSERT INTO runs ({', '.join(names)})"
            f" VALUES ({', '.join('?' * len(names))})"
            f" ON CONFLICT (machine, shot, diagnostic, preprocessed, plot,"
            f"               params_hash)"
            f" DO UPDATE SET {updates}",
            [fields[n] for n in names],
        )
        run = find_run(conn, target, plot, params_hash)
        conn.execute("DELETE FROM scalars WHERE run_id = ?", (run["id"],))
    return run


def delete_run(conn, run):
    """Drop a run, its scalars and its blob.

    The rows go even if the blob cannot be removed: a leftover blob is
    overwritten by the next compute, but a leftover row would keep returning
    the old result. Runs built on this one lose their ``upstream_run_id`` (the
    foreign key sets it NULL), which is how :func:`stale_runs` finds them.
    """
    if run["blob_path"]:
        with contextlib.suppress(OSError):
            os.remove(run["blob_path"])
    with conn:
        conn.execute("DELETE FROM scalars WHERE run_id = ?", (run["id"],))
        conn.execute("DELETE FROM runs WHERE id = ?", (run["id"],))


# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------


def write_scalars(conn, run_id, mapping):
    """Write ``mapping`` as this run's scalars and return how many rows.

    A key is a ``str`` for a shot-level scalar or an ``(x, y, name)`` tuple for
    one that belongs to a pixel. A NaN, or a value that is not a number, is
    stored as NULL: SQLite has no NaN.
    """
    rows = []
    for key, value in mapping.items():
        x, y, name = (SHOT_LEVEL, SHOT_LEVEL, key) if isinstance(key, str) else key
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        rows.append(
            (run_id, int(x), int(y), str(name), None if math.isnan(number) else number)
        )
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO scalars (run_id, x, y, name, value)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


_SCALAR_COLUMNS = [
    "machine",
    "shot",
    "diagnostic",
    "preprocessed",
    "plot",
    "params_hash",
    "x",
    "y",
    "name",
    "value",
]

_SCALAR_QUERY = """
SELECT r.machine, r.shot, r.diagnostic, r.preprocessed, r.plot, r.params_hash,
       s.x, s.y, s.name, s.value
  FROM scalars s JOIN runs r ON r.id = s.run_id
 WHERE r.status = 'ok'
"""


def scalar_frame(conn, names=None, machine=None, plot=None, params_hash=None):
    """The scalars of every ``ok`` run in long form, which the multi-shot view reads."""
    query, args = _SCALAR_QUERY, []
    for column, value in (
        ("r.machine", machine),
        ("r.plot", plot),
        ("r.params_hash", params_hash),
    ):
        if value is not None:
            query += f" AND {column} = ?"
            args.append(value)
    if names:
        query += f" AND s.name IN ({', '.join('?' * len(names))})"
        args.extend(names)
    return pd.DataFrame(
        [dict(row) for row in conn.execute(query, args)], columns=_SCALAR_COLUMNS
    )


# ---------------------------------------------------------------------------
# Blobs
# ---------------------------------------------------------------------------


def load_result(conn, run):
    """The stored dataset for ``run``, loaded into memory, or ``None``.

    ``None`` when the blob is gone or cannot be read, so that the caller
    recomputes instead of crashing the page.
    """
    path = run["blob_path"]
    if not path or not os.path.exists(path):
        return None
    try:
        with xr.open_dataset(path) as stored:
            return stored.load()
    except Exception:  # noqa: BLE001 - a corrupt blob must recompute, not crash
        return None


def _write_blob(result, path, plot, params_hash, text, code_version, created_at):
    directory = os.path.dirname(path)
    shared.makedirs(directory)
    result = result.copy()
    # The parameters go in as their canonical JSON string (netCDF attributes
    # cannot nest), so a blob found without the database describes itself.
    result.attrs.update(
        {
            "fusion_ui_plot": plot,
            "fusion_ui_params_hash": params_hash,
            "fusion_ui_params_json": text,
            "fusion_ui_code_version": code_version or "unknown",
            "fusion_ui_created_at": created_at,
        }
    )
    # Written aside and renamed into place. Two accounts write this directory
    # (see fusion_ui.core.shared), and a blob the other one left is not writable
    # by this one. A rename needs write permission on the directory only.
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f"{os.path.basename(path)}.tmp."
    )
    os.close(fd)
    try:
        result.to_netcdf(tmp_path)
        shared.share_file(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise
    return path


# ---------------------------------------------------------------------------
# Computing
# ---------------------------------------------------------------------------


class BatchOnlyError(RuntimeError):
    """A batch-only spec is missing, and the caller is not a batch job.

    ``spec`` and ``params`` name the batch-only link, which may be an upstream
    of the spec that was asked for. Raised before any run is recorded: the
    analysis did not fail, it was not run.
    """

    def __init__(self, spec, target, params):
        self.spec = spec
        self.target = target
        self.params = params
        super().__init__(
            f"{spec.label} ({spec.key!r}) is computed in batch only, by"
            f" `fusion-ui precompute {spec.key}`, and is not cached for"
            f" {target.label} with these parameters"
        )


def compute_and_store(conn, spec, target, params, ds, batch=False):
    """Run ``spec.compute``, store what it produced, return ``(result, run)``.

    A failure is recorded and comes back as ``(None, run)`` with
    ``run["status"] == "failed"``. A batch-only ``spec``, or a batch-only
    upstream that has to be computed, raises :class:`BatchOnlyError` unless
    ``batch`` is true.

    The row records ``input_mtime`` and ``upstream_run_id`` on a failure too,
    so that retrying a failed upstream marks the downstream failure stale.
    """
    if spec.batch_only and not batch:
        raise BatchOnlyError(spec, target, params)

    params_hash, text = record_params(conn, spec.key, params)
    columns = {
        "code_version": _code_version(),
        "input_mtime": input_mtime(conn, target),
        "upstream_run_id": None,
    }

    def failed(message, seconds=None):
        run = record_run(
            conn,
            target,
            spec.key,
            params_hash,
            blob_path=None,
            status="failed",
            error=message,
            seconds=seconds,
            **columns,
        )
        return None, run

    # The upstream is resolved before the clock starts: ``seconds`` is this
    # analysis alone.
    arguments = (ds, params)
    if spec.requires is not None:
        upstream, upstream_run = result(
            conn,
            registry.get(spec.requires),
            target,
            spec.upstream_params(params),
            ds,
            batch=batch,
        )
        if upstream_run is not None:
            columns["upstream_run_id"] = upstream_run["id"]
        if upstream is None:
            reason = (
                upstream_run["error"]
                if upstream_run is not None
                else "it produced nothing"
            )
            return failed(
                f"upstream {spec.requires!r} did not produce a result: {reason}"
            )
        arguments = (ds, params, upstream)

    path = blob_path(spec.key, params_hash, target)
    started = time.perf_counter()
    try:
        result_ds = spec.compute(*arguments)
    except Exception as error:  # noqa: BLE001 - a failure is a row, not a crash
        return failed(f"{type(error).__name__}: {error}", time.perf_counter() - started)
    try:
        created_at = _now()
        _write_blob(
            result_ds,
            path,
            spec.key,
            params_hash,
            text,
            columns["code_version"],
            created_at,
        )
        run = record_run(
            conn,
            target,
            spec.key,
            params_hash,
            blob_path=path,
            status="ok",
            error=None,
            seconds=time.perf_counter() - started,
            created_at=created_at,
            **columns,
        )
        if spec.scalars is not None:
            write_scalars(conn, run["id"], spec.scalars(result_ds))
    except Exception as error:  # noqa: BLE001 - a failure is a row, not a crash
        # The blob write, the ledger write or scalars() raised after a compute
        # that succeeded. A blob already in place has no `ok` row to name it.
        with contextlib.suppress(OSError):
            os.remove(path)
        return failed(f"{type(error).__name__}: {error}", time.perf_counter() - started)
    return result_ds, run


def result(conn, spec, target, params, ds, batch=False):
    """``(result, run)`` for one spec on one target.

    A live spec (``compute is None``) gets ``ds`` straight back, with no run.
    A cached spec is looked up first and computed only on a miss. A recorded
    failure is returned as it is, ``(None, run)``. See
    :func:`compute_and_store` for ``batch``.
    """
    if spec.compute is None:
        return ds, None

    params_hash, _ = record_params(conn, spec.key, params)
    run = find_run(conn, target, spec.key, params_hash)
    if run is not None:
        if run["status"] == "failed":
            return None, run
        stored = load_result(conn, run)
        if stored is not None:
            return stored, run
        # The row is there but the blob is not: recompute.
    return compute_and_store(conn, spec, target, params, ds, batch=batch)


# ---------------------------------------------------------------------------
# Reading without computing
# ---------------------------------------------------------------------------


def lookup(conn, spec, target, params):
    """``(result, run)`` from the ledger alone, for a page that must not compute.

    Never computes, never resolves an upstream and writes nothing, not even the
    ``param_sets`` row.

    - ``(None, None)``: nothing is stored for these parameters on this target.
    - ``(None, run)``: a recorded failure, or an ``ok`` row whose blob is gone
      or unreadable.
    - ``(result, run)``: the stored result, loaded into memory.

    A live spec stores nothing, so asking for one is a ``ValueError``.
    """
    if not spec.cached:
        raise ValueError(f"{spec.key!r} is a live spec: nothing is stored for it")
    params_hash, _ = params_ui.hash_params(spec.key, params)
    run = find_run(conn, target, spec.key, params_hash)
    if run is None or run["status"] != "ok":
        return None, run
    return load_result(conn, run), run


def missing_batch_upstreams(conn, spec, target, params):
    """The batch-only links in ``spec``'s chain that a compute here would need.

    ``[(link, link_params, run), ...]`` from ``spec`` itself down its
    ``requires`` chain. ``run`` is the link's ledger row: ``None``, a
    ``failed`` row, or an ``ok`` row whose blob has gone.

    The walk follows :func:`result`: it ends at a link whose result is cached,
    since nothing beneath a cache hit is resolved, and at a recorded failure.
    Every batch-only link up to there is listed, a failed one included (only
    the command line can retry it). ``[]`` means :func:`result` can resolve
    ``spec`` without starting a batch-only compute.

    Opens no blob, since a page asks this on every rerun. One that exists but
    cannot be read is caught later, by :class:`BatchOnlyError`.
    """
    links = registry.chain(spec)
    if not any(link.batch_only for link in links):
        return []
    missing = []
    for index, link in enumerate(links):
        if index:
            params = links[index - 1].upstream_params(params)
        params_hash, _ = params_ui.hash_params(link.key, params)
        run = find_run(conn, target, link.key, params_hash)
        status = None if run is None else run["status"]
        if status == "ok" and os.path.exists(run["blob_path"] or ""):
            break
        if link.batch_only:
            missing.append((link, params, run))
        if status == "failed":
            break
    return missing


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

#: The input file's mtime in the index is not the one the run recorded.
STALE_INPUT = "input changed"
#: The run's upstream row was deleted. What replaced it is not what this
#: result was built on.
STALE_UPSTREAM_DELETED = "upstream deleted"
#: The upstream row was written after this one: recomputed in place.
STALE_UPSTREAM_RECOMPUTED = "upstream recomputed"
#: Nothing about this run itself, but the run it was built on is stale.
STALE_UPSTREAM_STALE = "upstream stale"

_STALE_QUERY = """
SELECT r.*, s.mtime AS index_mtime
  FROM runs r
  LEFT JOIN shots s
    ON s.machine = r.machine AND s.shot = r.shot
   AND s.diagnostic = r.diagnostic AND s.preprocessed = r.preprocessed
 ORDER BY r.id
"""


def _instant(text):
    """``created_at`` as an aware datetime (UTC when it says nothing), or ``None``."""
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _chained(plot):
    """Whether runs of ``plot`` are always written with an upstream link."""
    spec = registry.REGISTRY.get(plot)
    if spec is None or spec.requires is None:
        return False
    upstream = registry.REGISTRY.get(spec.requires)
    return upstream is not None and upstream.cached


def _own_staleness(run, rows):
    """Why ``run`` itself is stale, or ``None``."""
    if run["index_mtime"] != run["input_mtime"]:
        return STALE_INPUT
    upstream_id = run["upstream_run_id"]
    if upstream_id is None:
        # Every chained run is written with its link, so none means the
        # upstream row was deleted since.
        return STALE_UPSTREAM_DELETED if _chained(run["plot"]) else None
    upstream = rows.get(upstream_id)
    if upstream is None:
        # A dangling link: deleted on a connection with foreign keys off.
        return STALE_UPSTREAM_DELETED
    mine, theirs = _instant(run["created_at"]), _instant(upstream["created_at"])
    if mine is not None and theirs is not None and theirs > mine:
        return STALE_UPSTREAM_RECOMPUTED
    return None


def stale_runs(conn, plot=None):
    """Every run whose stored result a recompute might not reproduce.

    ``[dict(run, stale=reason), ...]`` in ``id`` order, of any status (a
    failure on an input that has changed may now succeed). ``plot`` keeps one
    plot's runs. ``reason`` is :data:`STALE_INPUT`,
    :data:`STALE_UPSTREAM_DELETED`, :data:`STALE_UPSTREAM_RECOMPUTED`, or
    :data:`STALE_UPSTREAM_STALE` when only the run it was built on is stale,
    however far down the chain.

    Only a run that recorded its ``input_mtime``, and whose input is still in
    the index, is judged. Any other is never listed and passes no staleness
    on: without ``input_mtime`` nothing says what it was computed from, and
    without the input it could not be recomputed.

    Whether a plot is chained is read from the registry, so
    :mod:`fusion_ui.plots` must be imported first. A plot that is not
    registered is judged by the other rules.
    """
    rows = {row["id"]: dict(row) for row in conn.execute(_STALE_QUERY)}
    reasons = {}

    def reason(run_id, visiting):
        if run_id in reasons:
            return reasons[run_id]
        run = rows[run_id]
        found = None
        if run["input_mtime"] is not None and run["index_mtime"] is not None:
            found = _own_staleness(run, rows)
            upstream_id = run["upstream_run_id"]
            if found is None and upstream_id in rows and upstream_id not in visiting:
                if reason(upstream_id, visiting | {run_id}) is not None:
                    found = STALE_UPSTREAM_STALE
        reasons[run_id] = found
        return found

    stale = []
    for run_id, run in rows.items():
        why = reason(run_id, frozenset())
        if why is None or (plot is not None and run["plot"] != plot):
            continue
        listed = {k: v for k, v in run.items() if k != "index_mtime"}
        listed["stale"] = why
        stale.append(listed)
    return stale


# ---------------------------------------------------------------------------
# Pruning a whole plot
# ---------------------------------------------------------------------------


class PruneError(RuntimeError):
    """:func:`prune` or :func:`prune_scalars` refused to start, or could not write.

    The message says which, and what state that leaves behind.
    """


def _inside(path, root):
    """Whether ``path`` is an entry of the tree ``root``, which is already resolved.

    The directories on the way are resolved, so neither ``..`` nor a symlinked
    directory leads out of the tree. The last component is not: removing a
    link removes the link, never what it points at.
    """
    entry = os.path.join(
        os.path.realpath(os.path.dirname(path)), os.path.basename(path)
    )
    return os.path.commonpath([root, entry]) == root


def _runs_per_plot(conn):
    """``{plot: its runs}`` for every plot in the ledger, for a mistyped key's hint."""
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT plot, COUNT(*) FROM runs GROUP BY plot ORDER BY plot"
        )
    }


@dataclass
class PrunePlan:
    """Everything one plot owns, counted and not yet deleted: see :func:`plan_prune`."""

    plot: str
    runs: list  # the plot's ``runs`` rows, in id order
    scalars: dict  # run id -> its ``scalars`` rows, for the runs that have any
    present: list  # blob paths that are on disk
    missing: list  # blob paths the ledger lists that are not on disk
    outside: list  # (run id, blob path) lying outside ``CACHE_DIR``: refused
    locked: list  # present blob paths in a directory this user cannot write
    params: list  # hashes of the ``param_sets`` rows nothing will reference
    dependents: list  # (run id, plot, upstream run id) of other plots' runs on these
    plots: dict  # every plot in the ledger -> its runs

    @property
    def empty(self):
        """Nothing to delete: no run of the plot and no parameter set of it."""
        return not self.runs and not self.params

    @property
    def refusal(self):
        """Why :func:`prune` will not run on this plan, or ``None``."""
        if not self.outside:
            return None
        shown = "\n".join(f"  run {i}: {path}" for i, path in self.outside[:5])
        more = len(self.outside) - 5
        return (
            f"Refusing to prune {self.plot!r}: {len(self.outside)} of its runs list a"
            f" blob outside the result cache ({config.CACHE_DIR}), and prune never"
            f" deletes there:\n{shown}"
            + (f"\n  ... and {more} more" if more > 0 else "")
            + "\nNothing was deleted. Check FUSION_UI_CACHE against the ledger's"
            " blob paths."
        )

    def lines(self):
        """The counts, a line each, for whoever is about to say ``--yes``."""
        statuses = Counter(run["status"] for run in self.runs)
        kinds = ", ".join(f"{name} {n}" for name, n in sorted(statuses.items()))
        rows = [
            ("plot", self.plot),
            ("runs", f"{len(self.runs)}  ({kinds})"),
            ("scalar rows", sum(self.scalars.values())),
            (
                "blob files",
                f"{len(self.present)} on disk, {len(self.missing)} listed but"
                " already missing",
            ),
        ]
        if self.locked:
            folders = sorted({os.path.dirname(path) for path in self.locked})
            more = f" and {len(folders) - 3} more" if len(folders) > 3 else ""
            rows.append(
                (
                    "unwritable",
                    f"{len(self.locked)} blob files in {len(folders)} directories"
                    " this user cannot write, so their runs would be kept:"
                    f" {', '.join(folders[:3])}{more}",
                )
            )
        rows.append(
            ("param sets", f"{len(self.params)} that nothing will reference any more")
        )
        if self.outside:
            rows.append(("outside cache", f"{len(self.outside)} blob paths: refused"))
        if self.dependents:
            built = Counter(plot for _, plot, _ in self.dependents)
            names = ", ".join(f"{plot} {n}" for plot, n in sorted(built.items()))
            rows.append(
                (
                    "built on them",
                    f"{len(self.dependents)} runs of other plots ({names}): they lose"
                    " their upstream link, so the store treats them as stale",
                )
            )
        else:
            rows.append(("built on them", "no run of another plot"))
        return [f"{label:<14}{value}" for label, value in rows]


@dataclass
class PruneReport:
    """What :func:`prune` did."""

    plot: str
    runs: int = 0  # ledger rows deleted
    scalars: int = 0  # scalar rows that went with them
    blobs: int = 0  # blob files removed
    gone: int = 0  # blob files that were already missing
    params: int = 0  # ``param_sets`` rows deleted
    stale: int = 0  # runs of other plots that lost their upstream
    #: (run id, blob path, why) for each run kept because its blob could not be removed
    kept: list = field(default_factory=list)

    def lines(self):
        """What was done, a line each; what was not, and what to do about it, last."""
        lines = [
            f"{'deleted':<14}{self.runs} runs, {self.scalars} scalar rows,"
            f" {self.blobs} blob files, {self.params} param sets"
        ]
        if self.gone:
            lines.append(
                f"{'already gone':<14}{self.gone} blob files the ledger listed were"
                " not on disk"
            )
        if self.stale:
            lines.append(
                f"{'now stale':<14}{self.stale} runs of other plots lost their upstream"
            )
        if self.kept:
            lines.append(
                f"{'KEPT':<14}{len(self.kept)} runs, because their blob file could"
                " not be removed:"
            )
            lines += [f"  {path}  {why}" for _, path, why in self.kept[:10]]
            if len(self.kept) > 10:
                lines.append(f"  ... and {len(self.kept) - 10} more")
            lines.append(
                "Their ledger rows stay, so that no blob is left that nothing names."
                " Fix the permissions on those files' directories and run the same"
                " command again."
            )
        return lines


def plan_prune(conn, plot):
    """Count what pruning ``plot`` would delete, and delete nothing.

    ``plot`` is matched against ``runs.plot`` and need not be registered: a
    removed spec leaves its results behind. Needs ``CACHE_DIR``, to tell the
    cache's blobs from any that are not.

    ``params`` is every ``param_sets`` row of the plot that no other plot's
    run and no preset references, whether these runs leave it unreferenced or
    it already was.
    """
    root = os.path.realpath(config.CACHE_DIR)
    runs = conn.execute(
        "SELECT * FROM runs WHERE plot = ? ORDER BY id", (plot,)
    ).fetchall()
    scalars = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT s.run_id, COUNT(*) FROM scalars s JOIN runs r ON r.id = s.run_id"
            " WHERE r.plot = ? GROUP BY s.run_id",
            (plot,),
        )
    }
    present, missing, outside = [], [], []
    for run in runs:
        path = run["blob_path"]
        if not path:
            continue
        if not _inside(path, root):
            outside.append((run["id"], path))
        elif os.path.lexists(path):
            present.append(path)
        else:
            missing.append(path)
    # A forecast only: removing a file needs write and search permission on its
    # directory, and one the other writer made may grant this user neither.
    locked = [
        path
        for path in present
        if not os.access(os.path.dirname(path), os.W_OK | os.X_OK)
    ]
    params = [
        row[0]
        for row in conn.execute(
            "SELECT hash FROM param_sets WHERE plot = ?"
            "   AND hash NOT IN (SELECT params_hash FROM runs WHERE plot <> ?)"
            "   AND hash NOT IN (SELECT params_hash FROM presets)"
            " ORDER BY hash",
            (plot, plot),
        )
    ]
    dependents = [
        (row["id"], row["plot"], row["upstream_run_id"])
        for row in conn.execute(
            "SELECT id, plot, upstream_run_id FROM runs WHERE plot <> ?"
            "   AND upstream_run_id IN (SELECT id FROM runs WHERE plot = ?)"
            " ORDER BY id",
            (plot, plot),
        )
    ]
    return PrunePlan(
        plot=plot,
        runs=runs,
        scalars=scalars,
        present=present,
        missing=missing,
        outside=outside,
        locked=locked,
        params=params,
        dependents=dependents,
        plots=_runs_per_plot(conn),
    )


def prune(conn, plan):
    """Delete what ``plan`` counted, and return a :class:`PruneReport`.

    Blobs first, then the ledger in one transaction. A run whose blob cannot
    be removed (a directory the other writer made) keeps its row and is listed
    in the report, so the same command can finish the job later. The other way
    round, that blob would stay on disk with no row to name it. A row whose
    blob is gone is a state the store already handles.

    Only files under ``CACHE_DIR`` are removed, and a plan with a blob outside
    it is refused whole. Scalars go by ``ON DELETE CASCADE`` and the runs built
    on a deleted one lose their link by ``ON DELETE SET NULL``, so a connection
    with foreign keys off is refused too.

    Run it while nothing is computing the plot: a run recomputed in place
    since the plan was made is deleted by its id and leaves its new blob
    unlisted.

    Raises :class:`PruneError` when the plan is refused, or when the ledger
    cannot be written; the ledger is then unchanged, and the message says how
    many blobs were removed already.
    """
    if plan.refusal:
        raise PruneError(plan.refusal)
    if not conn.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise PruneError(
            "Refusing to prune on a connection with foreign keys off: the scalars"
            " would not follow their runs. Open it with fusion_ui.core.db.open_db."
            " Nothing was deleted."
        )
    report = PruneReport(plan.plot)
    deletable = []
    for run in plan.runs:
        path = run["blob_path"]
        if path:
            try:
                os.remove(path)
                report.blobs += 1
            except FileNotFoundError:
                report.gone += 1
            except OSError as error:
                why = f"{type(error).__name__}: {error.strerror or error}"
                report.kept.append((run["id"], path, why))
                continue
        deletable.append(run["id"])

    try:
        with conn:
            report.runs = conn.executemany(
                "DELETE FROM runs WHERE id = ?", [(run_id,) for run_id in deletable]
            ).rowcount
            report.params = conn.execute(
                "DELETE FROM param_sets WHERE plot = ?"
                "   AND hash NOT IN (SELECT params_hash FROM runs)"
                "   AND hash NOT IN (SELECT params_hash FROM presets)",
                (plan.plot,),
            ).rowcount
    except sqlite3.Error as error:
        raise PruneError(
            f"The ledger could not be updated ({error}), so it is unchanged, but"
            f" {report.blobs} blob files were removed already and those runs now"
            " read as not stored. Run the same command again to finish."
        ) from error
    deleted = set(deletable)
    report.scalars = sum(plan.scalars.get(run_id, 0) for run_id in deleted)
    report.stale = sum(1 for _, _, upstream in plan.dependents if upstream in deleted)
    return report


# ---------------------------------------------------------------------------
# Pruning some of a plot's scalars
# ---------------------------------------------------------------------------


@dataclass
class ScalarPrunePlan:
    """Some names' scalar rows under one plot key, counted: see :func:`plan_prune_scalars`."""

    plot: str
    names: tuple  # the names asked for, in the order given, each once
    rows: dict  # name -> {run id: rows of that name in it}: exactly what would go
    runs: int  # the plot's runs: every one of them stays, with its blob
    kept: int  # the plot's scalar rows of other names: they stay
    elsewhere: dict  # plot -> rows of these names under another plot: untouched
    held: list  # the names the plot's scalars have, for a mistyped name's hint
    plots: dict  # every plot in the ledger -> its runs

    @property
    def counts(self):
        """``{name: scalar rows that would go}``, a name with none counted as 0."""
        return {name: sum(self.rows[name].values()) for name in self.names}

    @property
    def total(self):
        return sum(self.counts.values())

    @property
    def empty(self):
        """The ledger holds no run of the plot at all: the key is mistyped."""
        return self.runs == 0

    def lines(self):
        """The counts, a line each, for whoever is about to say ``--yes``."""
        width = max([14, *(len(name) + 2 for name in self.names)])
        rows = [("plot", self.plot)]
        for name in self.names:
            held = self.rows[name]
            rows.append((name, f"{sum(held.values())} scalar rows in {len(held)} runs"))
        rows += [
            ("to delete", f"{self.total} scalar rows"),
            (
                "stays",
                f"all {self.runs} runs of the plot, with their blobs, and its"
                f" {self.kept} scalar rows of other names",
            ),
        ]
        if self.elsewhere:
            where = ", ".join(
                f"{plot} {n}" for plot, n in sorted(self.elsewhere.items())
            )
            rows.append(
                ("elsewhere", f"the same names under other plots, not touched: {where}")
            )
        else:
            rows.append(("elsewhere", "none of these names under another plot"))
        absent = [name for name in self.names if not self.rows[name]]
        if absent:
            # Not an error, but a typo looks the same as a job already done.
            shown = ", ".join(self.held[:20]) or "none"
            more = f" and {len(self.held) - 20} more" if len(self.held) > 20 else ""
            rows.append(
                (
                    "no rows",
                    f"{', '.join(absent)}: nothing of that name to delete. The scalar"
                    f" names this plot has: {shown}{more}",
                )
            )
        return [f"{label:<{width}}{value}" for label, value in rows]


@dataclass
class ScalarPruneReport:
    """What :func:`prune_scalars` did."""

    plot: str
    deleted: dict = field(default_factory=dict)  # name -> scalar rows deleted

    @property
    def scalars(self):
        return sum(self.deleted.values())

    def lines(self):
        names = ", ".join(f"{name} {n}" for name, n in self.deleted.items())
        return [f"{'deleted':<14}{self.scalars} scalar rows ({names})"]


def plan_prune_scalars(conn, plot, names):
    """Count the scalar rows of ``names`` under ``plot`` that would go, and delete nothing.

    ``plot`` is matched against ``runs.plot`` and need not be registered.
    ``names`` are exact scalar names, a repeated one counted once. A name with
    no row is counted as 0, which is not an error. What stays is counted too:
    the plot's runs, its scalars of other names, and the same names under
    every other plot (``elsewhere``).
    """
    names = tuple(dict.fromkeys(str(name) for name in names))
    if not names:
        raise ValueError("no scalar name to prune")
    marks = ", ".join("?" * len(names))
    rows = {name: {} for name in names}
    for run_id, name, n in conn.execute(
        "SELECT s.run_id, s.name, COUNT(*) FROM scalars s"
        "  JOIN runs r ON r.id = s.run_id"
        f" WHERE r.plot = ? AND s.name IN ({marks})"
        " GROUP BY s.run_id, s.name ORDER BY s.run_id",
        (plot, *names),
    ):
        rows[name][run_id] = n
    gone = sum(sum(held.values()) for held in rows.values())
    runs = conn.execute("SELECT COUNT(*) FROM runs WHERE plot = ?", (plot,)).fetchone()
    scalars = conn.execute(
        "SELECT COUNT(*) FROM scalars s JOIN runs r ON r.id = s.run_id"
        " WHERE r.plot = ?",
        (plot,),
    ).fetchone()
    elsewhere = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT r.plot, COUNT(*) FROM scalars s JOIN runs r ON r.id = s.run_id"
            f" WHERE r.plot <> ? AND s.name IN ({marks})"
            " GROUP BY r.plot ORDER BY r.plot",
            (plot, *names),
        )
    }
    held = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT s.name FROM scalars s JOIN runs r ON r.id = s.run_id"
            " WHERE r.plot = ? ORDER BY s.name",
            (plot,),
        )
    ]
    return ScalarPrunePlan(
        plot=plot,
        names=names,
        rows=rows,
        runs=runs[0],
        kept=scalars[0] - gone,
        elsewhere=elsewhere,
        held=held,
        plots=_runs_per_plot(conn),
    )


def prune_scalars(conn, plan):
    """Delete what ``plan`` counted, and return a :class:`ScalarPruneReport`.

    The rows go in one transaction: all of them, or none, with a
    :class:`PruneError`. Only the runs and names the plan counted are touched.
    The runs, their blobs, their other scalars and the same names under other
    plots stay. A connection with foreign keys off is refused, as in
    :func:`prune`. Run it while nothing is writing the plot.
    """
    if not conn.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise PruneError(
            "Refusing to prune scalars on a connection with foreign keys off: it did"
            " not come from fusion_ui.core.db.open_db, whose rules the store assumes."
            " Nothing was deleted."
        )
    report = ScalarPruneReport(plan.plot)
    try:
        with conn:
            for name in plan.names:
                report.deleted[name] = conn.executemany(
                    "DELETE FROM scalars WHERE run_id = ? AND name = ?",
                    [(run_id, name) for run_id in plan.rows[name]],
                ).rowcount
    except sqlite3.Error as error:
        raise PruneError(
            f"The ledger could not be updated ({error}), so it is unchanged and"
            " nothing was deleted."
        ) from error
    return report
