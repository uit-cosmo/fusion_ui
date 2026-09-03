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
"""

import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import xarray as xr

from fusion_ui import config
from fusion_ui.core import params_ui

#: Sentinel for a scalar that belongs to the shot rather than to one pixel.
#: Not NULL: SQLite permits NULLs in a non-INTEGER primary key, which would
#: silently break the uniqueness the table depends on.
SHOT_LEVEL = -1


def _now():
    # Same format as catalog uses for shots.mtime.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _code_version():
    """``git describe`` for this app and imaging_methods, or ``None``.

    Imported lazily and defensively: this module is used from the CLI as well
    as from a page, and a missing git checkout must not stop a result being
    stored.
    """
    try:
        from fusion_ui import ui

        return " ".join(f"{k}={v}" for k, v in sorted(ui.code_version().items()))
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


def record_run(conn, target, plot, params_hash, **columns):
    """Write the run row and return it, replacing any earlier attempt.

    An earlier attempt is genuinely superseded -- a failure that now succeeds,
    or a recompute under a newer ``code_version`` -- so the row is updated in
    place and its scalars are cleared rather than accumulating.
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
    """Drop a run, its scalars (by cascade) and its blob. The Recompute path."""
    path = run["blob_path"]
    if path and os.path.exists(path):
        os.remove(path)
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
    cache file alive.
    """
    path = run["blob_path"]
    if not path or not os.path.exists(path):
        return None
    with xr.open_dataset(path) as stored:
        return stored.load()


def _write_blob(result, path, plot, params_hash, text, code_version, created_at):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    result.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def _fail(conn, target, spec, params_hash, message, seconds, code_version):
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
    )


def compute_and_store(conn, spec, target, params, ds):
    """Run ``spec.compute``, store what it produced, return ``(result, run)``.

    On failure the exception is recorded and ``(None, run)`` comes back with
    ``run["status"] == "failed"``.
    """
    from fusion_ui.core import registry

    params_hash, text = record_params(conn, spec.key, params)
    code_version = _code_version()

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
        )
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
        )

    created_at = _now()
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
    )
    if spec.scalars is not None:
        write_scalars(conn, run["id"], spec.scalars(result_ds))
    return result_ds, run


def result(conn, spec, target, params, ds):
    """``(result, run)`` for one spec on one target -- the single entry point.

    A live spec (``compute is None``) gets its time-sliced input straight back
    and has no run row. A cached spec is looked up first, computed only if the
    ledger has nothing usable, and a recorded failure is returned as-is for the
    page to surface.
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
    return compute_and_store(conn, spec, target, params, ds)
