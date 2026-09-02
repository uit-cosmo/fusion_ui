"""Import ``density_scan/results.json`` as seed rows in the scalar store.

Fifty shots' worth of blob parameters already exist, computed over months by
``density_scan``. Loading them means the multi-shot view has something to plot
on the day it is built, instead of waiting for every analysis to be re-run
through this app.

They are attributed honestly. Everything imported lands under one reserved plot
key, :data:`IMPORT_PLOT`, with ``code_version = "imported"`` and a parameter set
that records the file it came from rather than pretending to be the parameters
that produced it -- the fifteen names come from five different analyses whose
settings were never written down per-value. When phase 03's real ports start
writing ``vx_c`` under their own plot keys, the two never collide, and a plot
can always say which it is showing.

The parameter set is keyed on the *content* of the file, so re-running the
import is a no-op and a genuinely changed ``results.json`` produces a fresh,
distinguishable set of rows beside the old ones.
"""

import hashlib
import os
from dataclasses import dataclass

from fusion_ui.core import registry, store

#: The reserved plot key every imported row carries.
IMPORT_PLOT = "density_scan_import"

#: density_scan ran its analysis on the preprocessed APD file
#: (``manager.get_dataset(shot, diagnostic=apd, preprocessed=True)``), so that
#: is what these rows describe.
IMPORT_DIAGNOSTIC = "apd"
IMPORT_PREPROCESSED = True


@dataclass
class ImportedResults:
    """The "parameters" of an import: where the numbers came from.

    Not the analysis settings -- those were never recorded per value. Hashing
    the file's own digest is what makes the import idempotent.
    """

    source: str = ""
    sha1: str = ""


@dataclass
class ImportStats:
    path: str
    shots: int = 0
    skipped: int = 0
    pixels: int = 0
    scalars: int = 0

    def summary(self):
        return (
            f"{os.path.basename(self.path)}: {self.shots} shots, {self.pixels} "
            f"pixels, {self.scalars} scalars"
            + (f", {self.skipped} already imported" if self.skipped else "")
        )


def default_results_path():
    """``density_scan/results.json`` as ``fusion_scripts`` knows it.

    ``fusion_scripts`` installs its settings as a **top-level** module called
    ``config``, so this import is emphatically not :mod:`fusion_ui.config`.
    Imported inside the function and aliased so that neither reader nor linter
    has to work that out from the module header.
    """
    import config as fusion_scripts_config

    return str(fusion_scripts_config.DENSITY_SCAN_RESULTS)


def file_digest(path):
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_results(path):
    """``{shot: ShotData}`` from a density_scan results file.

    ``ResultManager`` is not exported from any package ``__init__``, hence the
    module path. The file itself contains bare ``NaN`` literals -- which Python
    reads and a strict JSON parser does not -- so it must go through this
    loader rather than through, say, SQLite's JSON functions.
    """
    from density_scan.discharge import ResultManager

    return ResultManager.from_json(path).shots


def import_results(conn, path=None, machine=None):
    """Write every ``(shot, refx, refy, name)`` value into ``scalars``."""
    from fusion_ui import config

    path = path or default_results_path()
    machine = machine or config.MACHINE
    params = ImportedResults(source=os.path.basename(path), sha1=file_digest(path))
    params_hash, _ = store.record_params(conn, IMPORT_PLOT, params)

    stats = ImportStats(path=path)
    for shot, shot_data in sorted(load_results(path).items()):
        target = registry.Target(
            machine=machine,
            shot=int(shot),
            diagnostic=IMPORT_DIAGNOSTIC,
            preprocessed=IMPORT_PREPROCESSED,
            path="",
            t_start=float("nan"),
            t_end=float("nan"),
            window_source="none",
        )
        existing = store.find_run(conn, target, IMPORT_PLOT, params_hash)
        if existing is not None and existing["status"] == "ok":
            stats.skipped += 1
            continue

        mapping = {}
        for refx, column in shot_data.blob_params.items():
            for refy, blob in column.items():
                if blob is None:
                    continue
                stats.pixels += 1
                for name, value in blob.to_dict().items():
                    mapping[(int(refx), int(refy), name)] = value

        run = store.record_run(
            conn,
            target,
            IMPORT_PLOT,
            params_hash,
            blob_path=None,
            status="ok",
            error=None,
            seconds=None,
            code_version="imported",
        )
        stats.scalars += store.write_scalars(conn, run["id"], mapping)
        stats.shots += 1
    return stats
