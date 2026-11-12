"""The names of a retired one-off import of blob parameters.

The importer is gone. A ledger that holds its rows keeps them under the plot
key :data:`IMPORT_PLOT`: they show on the multi-shot axis and ``fusion-ui
prune`` removes them.
"""

from dataclasses import dataclass

#: The plot key every imported row carries. No spec has it.
IMPORT_PLOT = "density_scan_import"

#: The names of the imported file's per-pixel record, in its order.
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

#: The names of the record that were not imported: the file stored under them
#: the ellipse fitted to a contour, not the Gaussian fit they mean elsewhere.
NOT_IMPORTED = ("lx_f", "ly_f", "theta_f")


@dataclass
class ImportedResults:
    """The parameter set of the imported rows: the file they came from."""

    source: str = ""
    sha1: str = ""
