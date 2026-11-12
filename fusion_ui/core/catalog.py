"""The shot catalog: an index of what is on disk, and the join to metadata.

:func:`rescan` fills the ``shots`` table, and pages read only that table, never
the filesystem. It **stats files, it never opens them**: a single APD file is
~425 MB and the tree is tens of terabytes. A file with no discharge-DB entry is
indexed with ``has_metadata = 0`` and shown flagged, not hidden.
"""

import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
from experimental_database import PlasmaDischargeManager
from experimental_database.diagnostics import Diagnostic

DIAGNOSTICS = tuple(d.subfolder for d in Diagnostic)

# <diagnostic>_<shot>.nc, or the same name with a "_preprocessed" suffix.
_FILENAME_RE = {
    d: re.compile(rf"^{re.escape(d.subfolder)}_(\d+)(_preprocessed)?\.nc$")
    for d in Diagnostic
}


@dataclass(frozen=True)
class RescanStats:
    """What one :func:`rescan` pass did. Printed by the CLI, shown in the UI."""

    machine: str
    seen: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    missing_folders: tuple = ()
    seconds: float = 0.0

    def summary(self):
        parts = [
            f"{self.seen} files",
            f"+{self.inserted}",
            f"~{self.updated}",
            f"={self.unchanged}",
            f"-{self.removed}",
        ]
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.missing_folders:
            parts.append(f"missing: {', '.join(self.missing_folders)}")
        return f"[{self.machine}] " + ", ".join(parts) + f" in {self.seconds:.2f}s"


# ---------------------------------------------------------------------------
# The discharge descriptor (read-only, hand-curated)
# ---------------------------------------------------------------------------


def load_discharges(discharge_db_path):
    """``{shot_number: PlasmaDischarge}`` from the descriptor, which is never
    written. It carries a few exact duplicate entries; the last one wins."""
    manager = PlasmaDischargeManager(discharge_file=str(discharge_db_path))
    return {d.shot_number: d for d in manager.discharges}


# ---------------------------------------------------------------------------
# rescan
# ---------------------------------------------------------------------------


def _scan_folder(data_folder, diagnostic):
    """``([(shot, preprocessed, path, bytes, mtime), …], files skipped)`` for one
    diagnostic, by stat only, or ``None`` when its subfolder does not exist."""
    folder = os.path.join(data_folder, diagnostic.subfolder)
    if not os.path.isdir(folder):
        return None
    pattern = _FILENAME_RE[diagnostic]
    found, skipped = [], 0
    with os.scandir(folder) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match is None:
                skipped += 1
                continue
            stat = entry.stat()
            found.append(
                (
                    int(match.group(1)),
                    1 if match.group(2) else 0,
                    entry.path,
                    stat.st_size,
                    datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                )
            )
    return found, skipped


def rescan(conn, data_folder, machine, discharge_db_path=None):
    """Refresh the ``shots`` table for one machine from the data tree.

    A row whose file has disappeared is deleted. Everything happens in one
    transaction, so a crashed scan leaves the previous index intact.
    """
    started = time.perf_counter()
    known_shots = (
        set(load_discharges(discharge_db_path)) if discharge_db_path else set()
    )

    seen, skipped, missing = {}, 0, []
    for diagnostic in Diagnostic:
        result = _scan_folder(data_folder, diagnostic)
        if result is None:
            missing.append(diagnostic.subfolder)
            continue
        found, folder_skipped = result
        skipped += folder_skipped
        for shot, preprocessed, path, size, mtime in found:
            seen[(shot, diagnostic.subfolder, preprocessed)] = (
                path,
                size,
                mtime,
                1 if shot in known_shots else 0,
            )

    existing = {
        (row["shot"], row["diagnostic"], row["preprocessed"]): (
            row["path"],
            row["bytes"],
            row["mtime"],
            row["has_metadata"],
        )
        for row in conn.execute(
            "SELECT shot, diagnostic, preprocessed, path, bytes, mtime, has_metadata"
            " FROM shots WHERE machine = ?",
            (machine,),
        )
    }

    inserted = [key for key in seen if key not in existing]
    updated = [k for k, v in seen.items() if k in existing and existing[k] != v]
    # Rows of a diagnostic whose folder is missing right now are kept: a lost
    # mount or a mistyped --data-folder would otherwise wipe its index. They go
    # once the folder is seen, present and empty.
    removed = [key for key in existing if key not in seen and key[1] not in missing]

    with conn:
        conn.executemany(
            "INSERT INTO shots"
            " (machine, shot, diagnostic, preprocessed, path, bytes, mtime, has_metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (machine, shot, diagnostic, preprocessed) DO UPDATE SET"
            " path = excluded.path, bytes = excluded.bytes,"
            " mtime = excluded.mtime, has_metadata = excluded.has_metadata",
            [(machine, *key, *value) for key, value in seen.items()],
        )
        conn.executemany(
            "DELETE FROM shots WHERE machine = ? AND shot = ?"
            " AND diagnostic = ? AND preprocessed = ?",
            [(machine, *key) for key in removed],
        )

    return RescanStats(
        machine=machine,
        seen=len(seen),
        inserted=len(inserted),
        updated=len(updated),
        unchanged=len(seen) - len(inserted) - len(updated),
        removed=len(removed),
        skipped=skipped,
        missing_folders=tuple(missing),
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# Phantom frame interval: measured once per file, read everywhere
# ---------------------------------------------------------------------------

# The one diagnostic whose frame rate varies from shot to shot, which is why it
# has a browser column.
PHANTOM_DIAGNOSTIC = "phantom"


def frame_dt(path):
    """The file's median frame interval in seconds, read off the time
    coordinate alone. ``None`` when the file is missing, unreadable, or has
    fewer than two increasing time stamps."""
    try:
        with xr.open_dataset(path) as ds:
            if "time" not in ds.coords:
                return None
            times = np.asarray(ds["time"].values, dtype=float)
    except Exception:  # noqa: BLE001 - a bad file measures as nothing, not a crash
        return None
    if times.size < 2:
        return None
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


@dataclass(frozen=True)
class DtBackfillStats:
    """What one :func:`backfill_dt` pass did."""

    considered: int = 0  # phantom rows in scope
    measured: int = 0  # newly filled
    skipped: int = 0  # already filled, kept without reopening the file
    failed: int = 0  # file missing/unreadable, left NULL

    def summary(self):
        return (
            f"{self.considered} phantom files, +{self.measured} measured,"
            f" ={self.skipped} already filled, !{self.failed} failed"
        )


def backfill_dt(conn, machine, shots=None, force=False):
    """Fill ``shots.dt`` for the phantom rows of ``machine`` that lack it.

    ``shots`` is an optional set of shot numbers to restrict to; ``force``
    measures again the rows that already hold a value. A file that cannot be
    read keeps the value it had.
    """
    rows = conn.execute(
        "SELECT machine, shot, diagnostic, preprocessed, path, dt FROM shots"
        " WHERE machine = ? AND diagnostic = ?"
        " ORDER BY shot, preprocessed",
        (machine, PHANTOM_DIAGNOSTIC),
    ).fetchall()

    considered, measured, skipped, failed = 0, 0, 0, 0
    for row in rows:
        if shots is not None and row["shot"] not in shots:
            continue
        considered += 1
        if row["dt"] is not None and not force:
            skipped += 1
            continue
        value = frame_dt(row["path"])
        if value is None:
            failed += 1
            continue
        with conn:
            conn.execute(
                "UPDATE shots SET dt = ? WHERE machine = ? AND shot = ?"
                " AND diagnostic = ? AND preprocessed = ?",
                (
                    value,
                    row["machine"],
                    row["shot"],
                    row["diagnostic"],
                    row["preprocessed"],
                ),
            )
        measured += 1
    return DtBackfillStats(
        considered=considered, measured=measured, skipped=skipped, failed=failed
    )


# ---------------------------------------------------------------------------
# The browser table: index joined to metadata
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "machine",
    "shot",
    *DIAGNOSTICS,
    "phantom_dt",
    "I_p",
    "n_e_bar",
    "f_GW",
    "f_GW_source",
    "mode",
    "t_start",
    "t_end",
    "mlp_mode",
    "has_metadata",
    "bytes",
]


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float_or_nan(value):
    """``float(value)``, or NaN for ``None``, a non-number or an infinity: the
    discharge DB's fields are sparsely filled."""
    return float(value) if _finite(value) else float("nan")


def greenwald_fraction(discharge):
    """``(value, source)``: f_GW as the descriptor stores it (``"db"``), else
    derived from I_p and n̄_e (``"derived"``), else ``(nan, "")``.

    Most shots store none, so the source is always shown beside the value.
    """
    if _finite(discharge.greenwald_fraction):
        return float(discharge.greenwald_fraction), "db"
    if _finite(discharge.line_averaged_density) and _finite(discharge.plasma_current):
        if float(discharge.plasma_current) != 0.0:
            value = discharge.greenwald_fraction_fun()
            if _finite(value):
                return float(value), "derived"
    return float("nan"), ""


def _availability(raw, preprocessed):
    return {(1, 1): "R+P", (1, 0): "R", (0, 1): "P"}.get((raw, preprocessed), "")


def shot_table(conn, discharge_db_path):
    """One row per (machine, shot): what is on disk, joined to the metadata.

    The discharge descriptor is read straight from JSON, never mirrored into
    SQLite, where a copy would go stale.
    """
    rows = conn.execute(
        "SELECT machine, shot,"
        "       diagnostic,"
        "       MAX(CASE WHEN preprocessed = 0 THEN 1 ELSE 0 END) AS raw,"
        "       MAX(CASE WHEN preprocessed = 1 THEN 1 ELSE 0 END) AS prep,"
        "       SUM(COALESCE(bytes, 0)) AS bytes,"
        "       MAX(has_metadata) AS has_metadata,"
        "       MAX(dt) AS dt"  # NULL until `fusion-ui backfill-dt` runs
        "  FROM shots GROUP BY machine, shot, diagnostic"
    ).fetchall()

    discharges = load_discharges(discharge_db_path) if discharge_db_path else {}
    nan = float("nan")

    records = {}
    for row in rows:
        key = (row["machine"], row["shot"])
        record = records.get(key)
        if record is None:
            record = records[key] = dict(
                machine=row["machine"],
                shot=row["shot"],
                **{diagnostic: "" for diagnostic in DIAGNOSTICS},
                phantom_dt=nan,
                I_p=nan,
                n_e_bar=nan,
                f_GW=nan,
                f_GW_source="",
                mode="",
                t_start=nan,
                t_end=nan,
                mlp_mode="",
                has_metadata=False,
                bytes=0,
            )
        record[row["diagnostic"]] = _availability(row["raw"], row["prep"])
        if row["diagnostic"] == PHANTOM_DIAGNOSTIC and row["dt"] is not None:
            record["phantom_dt"] = float(row["dt"])
        record["bytes"] += row["bytes"] or 0
        record["has_metadata"] |= bool(row["has_metadata"])

    for (_, shot), record in records.items():
        discharge = discharges.get(shot)
        if discharge is None:
            continue
        f_gw, source = greenwald_fraction(discharge)
        record.update(
            I_p=_as_float_or_nan(discharge.plasma_current),
            n_e_bar=_as_float_or_nan(discharge.line_averaged_density),
            f_GW=f_gw,
            f_GW_source=source,
            mode=discharge.comment or "",
            t_start=_as_float_or_nan(discharge.t_start),
            t_end=_as_float_or_nan(discharge.t_end),
            mlp_mode=discharge.mlp_mode or "",
        )

    table = pd.DataFrame(list(records.values()), columns=TABLE_COLUMNS)
    if table.empty:
        return table
    return table.sort_values("shot", ignore_index=True)


def index_fingerprint(conn):
    """A cheap value that changes whenever ``shots`` does: a cache key.

    ``COUNT(dt)`` is in it because a ``backfill-dt`` pass changes no byte count
    or mtime.
    """
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(mtime), ''),"
        "       COUNT(dt) FROM shots"
    ).fetchone()
    return tuple(row)
