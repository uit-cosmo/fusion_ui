"""The result store: the run ledger, the netCDF blobs, and the scalar writer.

Everything a cached :class:`~fusion_ui.core.registry.PlotSpec` produces lands
here. One ``runs`` row per (what was computed, on what) is the provenance
record; the derived dataset goes to netCDF under ``CACHE_DIR``; whatever
``spec.scalars()`` returns goes to the ``scalars`` table, which is what the
multi-shot view reads.

Three decisions worth knowing before changing anything here.

**The cache never evicts.** With tens of terabytes free, an unevicted cache is
strictly better, and eviction would solve a problem we do not have. ``runs`` is
the ledger; deliberate cleanup is an operator decision, not a background
process that silently throws away a four-minute computation.

**``code_version`` is recorded but never hashed.** Putting the git hash in the
cache key would invalidate every result on every commit. Store it, show it
under the figure, and let the person looking at the plot decide whether the
version matters for what they are doing.

**A failure is a row, not an exception.** A compute that raises writes a
``status='failed'`` row carrying the message. Subsequent loads return that row
so the page can show the error and offer Recompute, rather than re-raising the
same traceback on every rerun.

**A spec with ``requires`` is resolved depth first**, and only when the
downstream result is actually missing -- a cache hit on the derived quantity
must not pay for its upstream. Each link in the chain keeps its own ledger row,
so the 2DCA average that four different plots are built on is computed and
stored exactly once.

**A batch-only spec is computed by batch jobs alone.** :func:`result` and
:func:`compute_and_store` raise :class:`BatchOnlyError` rather than compute one
unless the caller passes ``batch=True``, which only ``fusion-ui precompute``
does -- including when the batch-only spec is an upstream reached through a
chain. A page reads such results with :func:`lookup`, and asks
:func:`missing_batch_upstreams` before computing anything built on one.

**A result records what it was computed from** (schema v4): the input file's
mtime as the index recorded it, and the upstream run it was built on. Neither
is in the cache key. :func:`stale_runs` compares them with the index and the
ledger as they are now.
"""

import math
import os
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
import xarray as xr

from fusion_ui import config
from fusion_ui.core import params_ui, shared, versions

#: Sentinel for a scalar that belongs to the shot rather than to one pixel.
#: Not NULL: SQLite permits NULLs in a non-INTEGER primary key, which would
#: silently break the uniqueness the table depends on.
SHOT_LEVEL = -1


def _now():
    # ISO 8601 in UTC, as catalog writes shots.mtime, but to the microsecond:
    # stale_runs orders a run against its upstream by created_at, and an
    # upstream recomputed within the same second as its downstream was
    # written would otherwise not read as newer.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@lru_cache(maxsize=None)
def _code_version():
    """Each repository's commit as one string, or ``None``.

    Read once per process: the code a process runs is the code it imported,
    and a long batch must not run git again for every target. Defensive,
    because a missing checkout must not stop a result being stored. See
    :mod:`fusion_ui.core.versions`.
    """
    try:
        return versions.as_text(versions.code_version())
    except Exception:  # noqa: BLE001 - provenance is nice to have, not required
        return None


# ---------------------------------------------------------------------------
# Where a blob lives
# ---------------------------------------------------------------------------


def blob_path(plot, params_hash, target, suffix=".nc"):
    """``CACHE_DIR/runs/<plot>/<params_hash>/<target key><suffix>``.

    The hash is in the path, not just in the database, so two parameter sets
    for the same shot cannot overwrite each other on disk -- and so a stray
    directory can be read back to the run that made it.
    """
    return os.path.join(
        config.CACHE_DIR, "runs", plot, params_hash, f"{target.key}{suffix}"
    )


# ---------------------------------------------------------------------------
# param_sets
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


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


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
    """The input file's mtime as the index (``shots.mtime``) records it, or ``None``.

    What a run stores in ``runs.input_mtime``: the same string, so that
    :func:`stale_runs` can tell a rescan that saw the file rewritten by
    comparing the two directly. ``None`` for a target that is not indexed.
    """
    row = conn.execute(
        "SELECT mtime FROM shots WHERE machine = ? AND shot = ? AND diagnostic = ?"
        "   AND preprocessed = ?",
        (target.machine, target.shot, target.diagnostic, int(target.preprocessed)),
    ).fetchone()
    return None if row is None else row["mtime"]


def record_run(conn, target, plot, params_hash, **columns):
    """Write the run row and return it, replacing any earlier attempt.

    An earlier attempt is genuinely superseded -- a failure that now succeeds,
    or a recompute under a newer ``code_version`` -- so the row is updated in
    place (keeping its ``id``) and its scalars are cleared rather than
    accumulating. ``input_mtime`` and ``upstream_run_id`` are written on every
    call, as ``NULL`` unless given, so an update in place never keeps the
    previous attempt's provenance.
    """
    fields = {
        "machine": target.machine,
        "shot": target.shot,
        "diagnostic": target.diagnostic,
        # Part of the run's identity, not of its parameters: the raw and the
        # preprocessed file are different data and give different answers.
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
    """Drop a run, its scalars (by cascade) and its blob. The Recompute path.

    The ledger rows are always removed, even if the blob cannot be unlinked
    (permissions, concurrent delete): a leftover blob under a hash path is
    overwritten on the next compute, but a leftover row would short-circuit
    ``result()`` back to the stale result forever.

    Runs built on this one keep their results but lose their link to it
    (``upstream_run_id`` is set NULL by the foreign key), which is what
    :func:`stale_runs` reports them by.
    """
    path = run["blob_path"]
    if path:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    with conn:
        conn.execute("DELETE FROM scalars WHERE run_id = ?", (run["id"],))
        conn.execute("DELETE FROM runs WHERE id = ?", (run["id"],))


# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------


def _scalar_rows(run_id, mapping):
    for key, value in mapping.items():
        if isinstance(key, str):
            x, y, name = SHOT_LEVEL, SHOT_LEVEL, key
        else:
            x, y, name = key
            x, y = int(x), int(y)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        # SQLite has no NaN; binding one stores NULL anyway. Being explicit
        # keeps "the fit did not converge" readable in the table.
        yield run_id, x, y, str(name), None if math.isnan(number) else number


def write_scalars(conn, run_id, mapping):
    """Replace this run's scalars with ``mapping``.

    Keys are a ``str`` for a shot-level scalar or an ``(x, y, name)`` tuple for
    one that belongs to a pixel.
    """
    rows = list(_scalar_rows(run_id, mapping))
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO scalars (run_id, x, y, name, value)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


_SCALAR_QUERY = """
SELECT r.machine, r.shot, r.diagnostic, r.preprocessed, r.plot, r.params_hash,
       s.x, s.y, s.name, s.value
  FROM scalars s JOIN runs r ON r.id = s.run_id
 WHERE r.status = 'ok'
"""


def scalar_frame(conn, names=None, machine=None, plot=None, params_hash=None):
    """Long-form scalars as a DataFrame -- what the multi-shot view reads."""
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
        [dict(row) for row in conn.execute(query, args)],
        columns=[
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
        ],
    )


# ---------------------------------------------------------------------------
# Blobs
# ---------------------------------------------------------------------------


def load_result(conn, run):
    """The stored dataset for ``run``, or ``None`` if the blob is gone.

    Loaded into memory and the file closed: these are derived results, small
    next to the ~500 MB inputs, and holding a handle open would keep a deleted
    cache file alive. A corrupt/unreadable blob also returns ``None`` so the
    caller falls through and recomputes rather than crashing the page.
    """
    path = run["blob_path"]
    if not path or not os.path.exists(path):
        return None
    try:
        with xr.open_dataset(path) as stored:
            return stored.load()
    except Exception:  # noqa: BLE001 - corrupt cache must recompute, not crash
        return None


def _write_blob(result, path, plot, params_hash, text, code_version, created_at):
    directory = os.path.dirname(path)
    shared.makedirs(directory)
    result = result.copy()
    # netCDF attributes cannot hold nested structures, so the parameters go in
    # as their canonical JSON string -- which makes the blob self-describing if
    # it is ever found without the database beside it.
    result.attrs.update(
        {
            "fusion_ui_plot": plot,
            "fusion_ui_params_hash": params_hash,
            "fusion_ui_params_json": text,
            "fusion_ui_code_version": code_version or "unknown",
            "fusion_ui_created_at": created_at,
        }
    )
    # Write aside and rename into place: the blob directory is shared by two
    # accounts, and a blob left there by the other writer is not writable by
    # this one, so saving straight to `path` fails on the overwrite. Renaming
    # needs write permission on the directory only -- which the group-writable
    # setup provides -- whatever mode the previous blob carries.
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f"{os.path.basename(path)}.tmp."
    )
    os.close(fd)
    try:
        result.to_netcdf(tmp_path)
        # The service account and whoever runs `fusion-ui precompute` both write
        # here; see fusion_ui.core.shared.
        shared.share_file(tmp_path)
        os.replace(tmp_path, path)
        shared.share_file(path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


class BatchOnlyError(RuntimeError):
    """A batch-only spec is missing, and the caller is not a batch job.

    Raised by :func:`compute_and_store` before it records a run, whether the
    batch-only spec was asked for directly or is an upstream reached through a
    chain, so ``runs`` gains no row for either: the analysis did not fail, it
    was not run. ``spec`` and ``params`` name the batch-only link. A page
    catches this and shows the command that fills the cache instead.
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


def _fail(
    conn, target, spec, params_hash, message, seconds, code_version, **provenance
):
    return None, record_run(
        conn,
        target,
        spec.key,
        params_hash,
        blob_path=None,
        status="failed",
        error=message,
        seconds=seconds,
        code_version=code_version,
        **provenance,
    )


def compute_and_store(conn, spec, target, params, ds, batch=False):
    """Run ``spec.compute``, store what it produced, return ``(result, run)``.

    On failure the exception is recorded and ``(None, run)`` comes back with
    ``run["status"] == "failed"``.

    ``batch=True`` says the caller is a batch job (``fusion-ui precompute``),
    the only thing allowed to compute a batch-only spec: otherwise a batch-only
    ``spec``, or a batch-only upstream that has to be computed, raises
    :class:`BatchOnlyError` before any run is recorded.

    The row records what the result was computed from: ``input_mtime``, the
    input file's mtime as the index has it now, and ``upstream_run_id``, the
    run its upstream result came from -- also on a failure, so that retrying a
    failed upstream marks the downstream failure stale too.
    """
    from fusion_ui.core import registry

    if spec.batch_only and not batch:
        raise BatchOnlyError(spec, target, params)

    params_hash, text = record_params(conn, spec.key, params)
    code_version = _code_version()
    provenance = {"input_mtime": input_mtime(conn, target), "upstream_run_id": None}

    # An upstream is resolved before the clock starts, so ``seconds`` measures
    # this analysis and not the one it was waiting on -- each has its own row.
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
            provenance["upstream_run_id"] = upstream_run["id"]
        if upstream is None:
            reason = (
                upstream_run["error"]
                if upstream_run is not None
                else "it produced nothing"
            )
            return _fail(
                conn,
                target,
                spec,
                params_hash,
                f"upstream {spec.requires!r} did not produce a result: {reason}",
                None,
                code_version,
                **provenance,
            )
        arguments = (ds, params, upstream)

    started = time.perf_counter()
    try:
        result_ds = spec.compute(*arguments)
    except Exception as error:  # noqa: BLE001 - reported through the ledger
        return _fail(
            conn,
            target,
            spec,
            params_hash,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
            code_version,
            **provenance,
        )

    created_at = _now()
    try:
        path = _write_blob(
            result_ds,
            blob_path(spec.key, params_hash, target),
            spec.key,
            params_hash,
            text,
            code_version,
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
            code_version=code_version,
            created_at=created_at,
            **provenance,
        )
        if spec.scalars is not None:
            write_scalars(conn, run["id"], spec.scalars(result_ds))
    except Exception as error:  # noqa: BLE001 - a failure is a row, not a crash
        # Blob write, ledger write, or scalars() raised after a successful
        # compute (disk full, PermissionError, bad attrs, non-numeric mapping).
        # Record it so the page shows the error with Recompute instead of a
        # traceback, and remove a half-written blob if there is one.
        try:
            partial = blob_path(spec.key, params_hash, target)
            if os.path.exists(partial):
                os.remove(partial)
        except OSError:
            pass
        return _fail(
            conn,
            target,
            spec,
            params_hash,
            f"{type(error).__name__}: {error}",
            time.perf_counter() - started,
            code_version,
            **provenance,
        )
    return result_ds, run


def result(conn, spec, target, params, ds, batch=False):
    """``(result, run)`` for one spec on one target -- the single entry point.

    A live spec (``compute is None``) gets its time-sliced input straight back
    and has no run row. A cached spec is looked up first, computed only if the
    ledger has nothing usable, and a recorded failure is returned as-is for the
    page to surface. A batch-only spec, or one whose chain reaches a missing
    batch-only link, is computed only with ``batch=True``; otherwise a miss
    raises :class:`BatchOnlyError` (see :func:`compute_and_store`).
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
        # The row survived but the blob did not -- someone cleared the cache
        # directory. Fall through and recompute rather than reporting nothing.
    return compute_and_store(conn, spec, target, params, ds, batch=batch)


# ---------------------------------------------------------------------------
# Reading without computing
# ---------------------------------------------------------------------------


def lookup(conn, spec, target, params):
    """``(result, run)`` from the ledger alone, for a page that must not compute.

    Never computes, never resolves an upstream, and writes nothing -- not even
    the ``param_sets`` row :func:`result` records.

    - ``(None, None)``: nothing is stored for these parameters on this target.
    - ``(None, run)``: a recorded failure (``run["status"] == "failed"``), or
      an ``ok`` row whose blob is gone or unreadable, which a compute would
      replace.
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


def _blob_on_disk(run):
    return (
        run is not None
        and run["status"] == "ok"
        and bool(run["blob_path"])
        and os.path.exists(run["blob_path"])
    )


def missing_batch_upstreams(conn, spec, target, params):
    """The batch-only links in ``spec``'s chain that a compute here would need.

    ``[(link, link_params, run), …]``, from ``spec`` itself down its
    ``requires`` chain, with each link's parameters lifted out of the one above
    it as the store lifts them. ``run`` is the link's ledger row: ``None`` when
    nothing is stored, a ``failed`` row, or an ``ok`` row whose blob has gone.

    The walk follows :func:`result`: a link whose result is cached ends it,
    because nothing beneath a cache hit is resolved, and so does a recorded
    failure, which is handed back as it is. Every batch-only link before that
    point is listed. ``[]`` therefore means :func:`result` can resolve ``spec``
    without starting a batch-only compute, which is when a page may compute a
    derived spec inline. A failed batch-only link is listed too: only the
    command line can retry it.

    Reads the ledger and checks that blobs exist, but opens none: a page asks
    this on every rerun. A blob that exists but cannot be read is caught later,
    by :class:`BatchOnlyError`.
    """
    from fusion_ui.core import registry

    links = registry.chain(spec)
    if not any(link.batch_only for link in links):
        return []
    missing = []
    current = params
    for index, link in enumerate(links):
        if index:
            current = links[index - 1].upstream_params(current)
        params_hash, _ = params_ui.hash_params(link.key, current)
        run = find_run(conn, target, link.key, params_hash)
        failed = run is not None and run["status"] == "failed"
        if not failed and _blob_on_disk(run):
            break
        if link.batch_only:
            missing.append((link, current, run))
        if failed:
            break
    return missing


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

#: The input file's mtime in the index differs from the one the run recorded:
#: a rescan saw the file rewritten (re-preprocessed, a corrected mask).
STALE_INPUT = "input changed"
#: The run's upstream row was deleted (Recompute, ``--force``, a retried
#: failure); whatever replaced it is not what this result was built on.
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
    from fusion_ui.core import registry

    spec = registry.REGISTRY.get(plot)
    if spec is None or spec.requires is None:
        return False
    upstream = registry.REGISTRY.get(spec.requires)
    return upstream is not None and upstream.cached


def _own_staleness(run, rows):
    """Why ``run`` itself is stale, ``None`` if it is not or nobody can tell."""
    if run["input_mtime"] is None:
        # Written before schema v4 (or for a target the index does not hold):
        # nothing says what it was computed from, so it is unknown, not stale.
        return None
    if run["index_mtime"] is not None and run["index_mtime"] != run["input_mtime"]:
        return STALE_INPUT
    upstream_id = run["upstream_run_id"]
    if upstream_id is None:
        # The store links every chained run when it writes it, so a missing
        # link on a v4 row means the upstream row was deleted since.
        return STALE_UPSTREAM_DELETED if _chained(run["plot"]) else None
    upstream = rows.get(upstream_id)
    if upstream is None:
        # A dangling link: the upstream was deleted by a connection that had
        # foreign keys off, so SET NULL never fired.
        return STALE_UPSTREAM_DELETED
    mine, theirs = _instant(run["created_at"]), _instant(upstream["created_at"])
    if mine is not None and theirs is not None and theirs > mine:
        return STALE_UPSTREAM_RECOMPUTED
    return None


def stale_runs(conn, plot=None):
    """Every run whose stored result a recompute might no longer reproduce.

    ``[dict(run, stale=reason), …]`` in ``id`` order, of any status: a failure
    on an input that has since changed may now succeed. ``plot`` restricts the
    list to one plot's runs; the rules still follow their chains through every
    other plot. ``reason`` is one of

    - :data:`STALE_INPUT`: ``runs.input_mtime`` differs from ``shots.mtime``;
    - :data:`STALE_UPSTREAM_DELETED`: the upstream link is NULL (or dangling)
      on a run of a chained spec;
    - :data:`STALE_UPSTREAM_RECOMPUTED`: the upstream row's ``created_at`` is
      later than this row's;
    - :data:`STALE_UPSTREAM_STALE`: none of those, but the upstream run is
      stale itself, by any rule -- so a stale 2DCA average marks every result
      built on it, however far down the chain.

    A run with no ``input_mtime`` -- written before schema v4, or for a target
    the index does not hold -- is unknown and never listed, so legacy results
    are not recomputed wholesale. Nor is a run whose input file has left the
    index, which nothing could recompute. Whether a plot is chained is read
    from the registry, so import :mod:`fusion_ui.plots` first, as every entry
    point does; a plot that is not registered is judged by the other rules.
    """
    rows = {row["id"]: dict(row) for row in conn.execute(_STALE_QUERY)}
    reasons = {}

    def reason(run_id, visiting):
        if run_id in reasons:
            return reasons[run_id]
        run = rows[run_id]
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
