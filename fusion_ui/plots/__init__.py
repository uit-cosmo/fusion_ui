"""Every analysis in the app, one module each.

Importing this package registers them all: a page asks
:func:`fusion_ui.core.registry.for_diagnostic` what it can draw and never
mentions an individual plot. Adding one is a new module here plus its import
below -- no UI change, no new storage.

Import order matters only where a spec declares ``requires``: the upstream must
already be registered, which is checked when the downstream registers.
"""

from fusion_ui.plots import (  # noqa: F401 - imported to register
    probe,
    raw,
    spectra,
    two_dca,
    velocity_contour,
)

__all__ = ["probe", "raw", "spectra", "two_dca", "velocity_contour"]
