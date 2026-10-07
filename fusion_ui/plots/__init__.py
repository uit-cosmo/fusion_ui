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
    dead_pixels,
    spectra,
    velocity_tde,
    two_dca,
    velocity_contour,
    fwhm_sizes,
    gaussian_sizes,
    velocity_2dca_tde,
    trajectories,
    two_sided_exp,
    velocity_field,
    pixel_averages,
    method_fields,
    blob_parameters,
)

# Order is not cosmetic below two_dca: register() rejects a spec whose
# upstream is not yet in the registry, so every requires="two_dca" spec has
# to be imported after it. velocity_tde and velocity_field are unchained --
# they run their own analysis off the raw record -- so they are free to sit
# anywhere; they are grouped with their nearest relatives instead.
#
# The same holds for the three decorrelation products at the end:
# method_fields and blob_parameters require pixel_averages, the only batch-only
# spec, so it comes first. Their parameter classes are module-level
# (fusion_ui.plots.<key>.<Key>Params) because a precompute pool worker receives
# them pickled, and because the class's module and name are part of every cache
# key -- the three keys and those names are as permanent as any other.
__all__ = [
    "probe",
    "raw",
    "dead_pixels",
    "spectra",
    "velocity_tde",
    "two_dca",
    "velocity_contour",
    "fwhm_sizes",
    "gaussian_sizes",
    "velocity_2dca_tde",
    "trajectories",
    "two_sided_exp",
    "velocity_field",
    "pixel_averages",
    "method_fields",
    "blob_parameters",
]
