"""The numbers at one pixel: every method's velocity, and the blob parameters when they are computed.

Plain tables of floats. A number a method did not produce is NaN, never text, so a column stays
numeric and sorts as one (``st.dataframe`` greys a missing numeric as ``None``; see CLAUDE.md).
"""

import numpy as np
import pandas as pd

from fusion_ui.views import pixel
from fusion_ui.views.methods import METHODS

METHOD_COLUMNS = [
    "method",
    "v_R [m/s]",
    "v_Z [m/s]",
    "|v| [m/s]",
    "lags",
    "events",
    "level",
    "CC",
]

#: The 13 blob parameters in the order ``apd<shot>_blobs.nc`` lists them: what each is, and its unit
#: (the file's ``units`` attribute says "m, m^2, s, rad").
BLOB_PARAMETERS = (
    ("nevents", "events the conditional average rests on", ""),
    ("level", "contour level, as a fraction of the maximum", ""),
    ("area", "area of the zero-lag contour", "m²"),
    ("lx_c", "ellipse fitted to the contour: size in x", "m"),
    ("ly_c", "ellipse fitted to the contour: size in y", "m"),
    ("theta_c", "ellipse fitted to the contour: tilt", "rad"),
    ("lr", "FWHM along the reference's row", "m"),
    ("lz", "FWHM along the reference's column", "m"),
    ("lx_f", "Gaussian fit: size in x", "m"),
    ("ly_f", "Gaussian fit: size in y", "m"),
    ("theta_f", "Gaussian fit: tilt", "rad"),
    ("taud", "duration time, from the PSD", "s"),
    ("lam", "asymmetry of the two-sided exponential pulse", ""),
)


def _at(bundle, name):
    values = bundle.field_values(name)
    if values is None:
        return float("nan")
    x, y = bundle.pixel
    return float(values[y, x])


def method_table(bundle):
    """One row per method: v_R, v_Z, |v| and what the estimate rests on (lags, events, level, CC)."""
    if bundle.fields is None or bundle.pixel is None:
        return pd.DataFrame(columns=METHOD_COLUMNS)
    events = _at(bundle, "nevents")
    rows = []
    for method in METHODS:
        vr, vz = _at(bundle, method.vr), _at(bundle, method.vz)
        rows.append(
            {
                "method": method.label,
                "v_R [m/s]": vr,
                "v_Z [m/s]": vz,
                "|v| [m/s]": float(np.hypot(vr, vz)),
                "lags": _at(bundle, method.nlags) if method.nlags else float("nan"),
                "events": events if method.uses_events else float("nan"),
                "level": (
                    _at(bundle, f"level_{method.track}")
                    if method.track
                    else float("nan")
                ),
                "CC": _at(bundle, "cc_tde") if method.colour == "cc" else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=METHOD_COLUMNS)


def blob_table(bundle):
    """The blob parameters at the pixel in view, or ``None`` when ``blob_parameters`` is not computed."""
    if bundle.blobs is None or bundle.pixel is None:
        return None
    x, y = bundle.pixel
    rows = []
    for name, meaning, unit in BLOB_PARAMETERS:
        if name not in bundle.blobs:
            continue
        rows.append(
            {
                "parameter": name,
                "value": float(
                    np.asarray(bundle.blobs[name].values, dtype=float)[y, x]
                ),
                "unit": unit,
                "meaning": meaning,
            }
        )
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "meaning"])


def numbers(bundle):
    """``[method table, blob table]`` for the pixel in view; a sentence instead when it has nothing to show.

    A dead pixel and a live one with no events are ordinary, not errors: the page puts the sentence
    where the tables would have been. The blob table is left out until ``blob_parameters`` is computed.
    """
    why = pixel.problem(bundle, need_bank=False)
    if why:
        return why
    tables = [method_table(bundle)]
    blobs = blob_table(bundle)
    if blobs is not None:
        tables.append(blobs)
    return tables
