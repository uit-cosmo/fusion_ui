"""The shot catalog: an index of what is on disk, and the join to metadata.

With shots arriving continuously, ``os.listdir`` per page load stops being
acceptable and the discharge DB stops being the authoritative shot list.
:func:`rescan` -- on cron plus a button in the UI -- fills the ``shots`` table;
the browser reads only from it. Files with no discharge-DB entry still appear,
flagged ``has_metadata = 0``, so a newly copied shot is visible the moment it
lands rather than waiting for someone to curate it.

:func:`rescan` **stats files, it never opens them.** A single APD file is
~425 MB and the tree is tens of terabytes.
"""

import math
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd
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

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# The discharge descriptor (read-only, hand-curated)
# ---------------------------------------------------------------------------


def load_discharges(discharge_db_path):
    """Return ``{shot_number: PlasmaDischarge}`` from the descriptor.

    The file carries a handful of exact duplicate entries; last one wins, which
    is safe because duplicates do not conflict. Never written back.
    """
    manager = PlasmaDischargeManager(discharge_file=str(discharge_db_path))
    return {d.shot_number: d for d in manager.discharges}


# ---------------------------------------------------------------------------
# rescan
# ---------------------------------------------------------------------------


def _scan_folder(data_folder, diagnostic):
    """Yield ``(shot, preprocessed, path, bytes, mtime)`` for one diagnostic.

    ``os.scandir`` on one directory, stat only. Returns ``None`` instead of a
    generator when the diagnostic subfolder does not exist at all.
    """
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

    Rows whose file has disappeared are deleted, so the index cannot outlive
    the data. Everything happens in one transaction: a crashed scan leaves the
    previous index intact rather than a half-emptied one.
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
    removed = [key for key in existing if key not in seen]

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
# The browser table: index joined to metadata
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "machine",
    "shot",
    *DIAGNOSTICS,
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


def greenwald_fraction(discharge):
    """``(value, source)`` -- stored if the descriptor has it, else derived.

    f_GW is filled in for only about a quarter of the curated shots but is
    derivable from I_p and n̄_e for most of the rest. Deriving it silently would
    be worse than not deriving it, hence the source flag: it ends up as a column
    so nobody sorts by a number whose origin they cannot see.
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

    The discharge descriptor is read straight from JSON rather than mirrored
    into SQLite -- it is small, it is hand-curated, and a copy in the app
    database would be one more thing that can go stale.
    """
    rows = conn.execute(
        "SELECT machine, shot,"
        "       diagnostic,"
        "       MAX(CASE WHEN preprocessed = 0 THEN 1 ELSE 0 END) AS raw,"
        "       MAX(CASE WHEN preprocessed = 1 THEN 1 ELSE 0 END) AS prep,"
        "       SUM(COALESCE(bytes, 0)) AS bytes,"
        "       MAX(has_metadata) AS has_metadata"
        "  FROM shots GROUP BY machine, shot, diagnostic"
    ).fetchall()

    discharges = load_discharges(discharge_db_path) if discharge_db_path else {}

    records = {}
    for row in rows:
        key = (row["machine"], row["shot"])
        record = records.get(key)
        if record is None:
            record = records[key] = dict.fromkeys(TABLE_COLUMNS)
            record.update(
                machine=row["machine"],
                shot=row["shot"],
                bytes=0,
                has_metadata=False,
                **{diagnostic: "" for diagnostic in DIAGNOSTICS},
            )
        record[row["diagnostic"]] = _availability(row["raw"], row["prep"])
        record["bytes"] += row["bytes"] or 0
        record["has_metadata"] |= bool(row["has_metadata"])

    for (_, shot), record in records.items():
        discharge = discharges.get(shot)
        if discharge is None:
            record.update(
                I_p=float("nan"),
                n_e_bar=float("nan"),
                f_GW=float("nan"),
                f_GW_source="",
                mode="",
                t_start=float("nan"),
                t_end=float("nan"),
                mlp_mode="",
            )
            continue
        f_gw, source = greenwald_fraction(discharge)
        record.update(
            I_p=float(discharge.plasma_current),
            n_e_bar=float(discharge.line_averaged_density),
            f_GW=f_gw,
            f_GW_source=source,
            mode=discharge.comment or "",
            t_start=float(discharge.t_start),
            t_end=float(discharge.t_end),
            mlp_mode=discharge.mlp_mode or "",
        )

    table = pd.DataFrame(list(records.values()), columns=TABLE_COLUMNS)
    if table.empty:
        return table
    return table.sort_values("shot", ignore_index=True)


def index_fingerprint(conn):
    """A cheap value that changes whenever ``shots`` does -- a cache key."""
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(mtime), '')"
        " FROM shots"
    ).fetchone()
    return tuple(row)
