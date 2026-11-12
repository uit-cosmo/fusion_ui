"""Every analysis in the app, one module each.

Importing this package registers them all: a page asks
:func:`fusion_ui.core.registry.for_diagnostic` what it can draw and never
mentions an individual plot. Adding one is a new module here plus its import
below.

Import order matters only where a spec declares ``requires``: the upstream must
already be registered, so ``pixel_averages`` comes before the two products built
on it. The first spec of a diagnostic is the one the single-shot page opens on.
"""

from fusion_ui.plots import (  # noqa: F401 - imported to register
    probe,
    raw,
    dead_pixels,
    thomson_profiles,
    mlp_profiles,
    spectra,
    velocity_tde,
    pixel_averages,
    method_fields,
    blob_parameters,
)

__all__ = [
    "probe",
    "raw",
    "dead_pixels",
    "thomson_profiles",
    "mlp_profiles",
    "spectra",
    "velocity_tde",
    "pixel_averages",
    "method_fields",
    "blob_parameters",
]
