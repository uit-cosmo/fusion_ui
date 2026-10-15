"""What the one-off seed of June's ``density_scan/results.json`` left in the ledger.

Fifty shots' worth of blob parameters were computed over months by fusion_scripts'
``density_scan`` and imported, so that the multi-shot view had something to plot on the
day it was built. **The importer is retired** (L14: ``fusion-ui import-results`` and its
reading of ``results.json`` are gone, since the batch products have replaced the seed).
The rows stay in a ledger that holds them, readable on the multi-shot axis and removable
with ``fusion-ui prune``, and this module keeps the names they are known by.

They were attributed honestly: everything imported landed under one reserved plot key,
:data:`IMPORT_PLOT`, with ``code_version = "imported"`` and a parameter set
(:class:`ImportedResults`) that records the file it came from rather than pretending to be
the parameters that produced it. The file held fifteen names and twelve were imported:
:data:`NOT_IMPORTED` says which three were not, and why.
"""

from dataclasses import dataclass

#: The reserved plot key every imported row carries.
IMPORT_PLOT = "density_scan_import"

#: ``density_scan`` ran its analysis on the preprocessed APD file
#: (``manager.get_dataset(shot, diagnostic=apd, preprocessed=True)``), so that
#: is what these rows describe.
IMPORT_DIAGNOSTIC = "apd"
IMPORT_PREPROCESSED = True

#: The names in ``results.json`` that are not imported. ``density_scan/utils.py:
#: analysis`` stores the ellipse fitted to the contour at 0.3 of the maximum under
#: ``lx_f``, ``ly_f`` and ``theta_f``, not the Gaussian fit those names mean
#: everywhere else in this store (``gaussian_sizes``, ``blob_parameters``), and its
#: ``theta_f`` is in an older angle convention. One name on one axis must not be
#: two quantities, so the three are left out. What an earlier import wrote is
#: removed from the ledger by ``fusion-ui prune --plot density_scan_import --scalar
#: lx_f --scalar ly_f --scalar theta_f``.
NOT_IMPORTED = ("lx_f", "ly_f", "theta_f")

#: The fifteen names of ``results.json``'s per-pixel record (``density_scan.discharge.BlobParameters``, read
#: off it before the importer was retired, in the record's order). Twelve of them were imported.
RECORD_NAMES = (
    "vx_c",
    "vy_c",
    "area_c",
    "vx_2dca_tde",
    "vy_2dca_tde",
    "vx_tde",
    "vy_tde",
    "lx_f",
    "ly_f",
    "lr",
    "lz",
    "theta_f",
    "taud_psd",
    "lambda_psd",
    "number_events",
)


@dataclass
class ImportedResults:
    """The "parameters" of an import: where the numbers came from.

    Not the analysis settings -- those were never recorded per value. The importer hashed the
    file's own digest, which made it idempotent. Kept so that a seeded row can be built by hand.
    """

    source: str = ""
    sha1: str = ""
