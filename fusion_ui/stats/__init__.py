"""Every statistic in the app, one module each.

Importing this package registers them all: the statistics page asks
:func:`fusion_ui.core.statistics.all_specs` what it can draw and never mentions
an individual statistic. Adding one is a new module here plus its import below
-- no page change, no new storage.
"""

from fusion_ui.stats import (  # noqa: F401 - imported to register
    trace,
    pdf,
    psd,
    acf,
    ccf,
)

__all__ = [
    "trace",
    "pdf",
    "psd",
    "acf",
    "ccf",
]
