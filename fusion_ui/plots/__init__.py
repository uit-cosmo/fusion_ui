"""Every analysis in the app, one module each.

Importing this package registers them all: a page asks
:func:`fusion_ui.core.registry.for_diagnostic` what it can draw and never
mentions an individual plot. Adding one is a new module here plus its import
below -- no UI change, no new storage.
"""

from fusion_ui.plots import probe, raw, spectra  # noqa: F401 - imported to register

__all__ = ["probe", "raw", "spectra"]
