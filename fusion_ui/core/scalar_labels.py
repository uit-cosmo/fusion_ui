"""What the multi-shot axis calls a stored scalar: the quantity and its unit.

The store's names are permanent keys (``vr_com``); a person reading a scatter needs to know what the
number is (``v_R, 2DCA centroid [m/s]``). :data:`LABELS` holds that for the 32 names the phase-06
products write (``method_fields`` 20, ``blob_parameters`` 12). A name without an entry, which is every
name an older spec or the seed writes alone, shows as its raw name: nothing is guessed for it.

**Every label carries a unit, or says outright that it has none.** The unit is the last thing in the
label, in square brackets, and one of :data:`UNITS`. A number with no unit (a count of events or lags,
a fraction of a maximum, a correlation coefficient, an asymmetry between 0 and 1) says ``[no unit]``,
so that a unit someone forgot cannot be mistaken for one that does not exist. The metres, seconds and
radians are what the products store (R and Z in metres, lags in seconds). For the blob parameters they
are also what ``fusion_ui.views.numbers.BLOB_PARAMETERS`` gives, which a test compares.

**Labels are per name, not per source.** Eight of the 32 names are written by an older source too
(:data:`SHARED`), and the axis shows the same label whichever source is picked. Each of those is
therefore worded by what the quantity is, never by the product that happens to write it, and carries
the unit all its sources give it. G2 settled that they are the same estimators in the same units; the
wording follows the older sources' own docstrings (``fwhm_sizes``, ``gaussian_sizes``, ``taud_psd``
and the seed's ``density_scan``).

Nothing in this module touches Streamlit, the database or the filesystem.
"""

import re
import textwrap
from dataclasses import dataclass

#: The unit of a number that has none, said outright: a count, a fraction, a coefficient.
NO_UNIT = "no unit"

#: The units a label may carry.
UNITS = ("m/s", "m", "m²", "s", "rad", NO_UNIT)


@dataclass(frozen=True)
class Label:
    """A quantity and its unit. ``str()`` is what an axis shows: ``quantity [unit]``."""

    quantity: str
    unit: str

    def __str__(self):
        return f"{self.quantity} [{self.unit}]"


#: The names that an older source writes as well as a phase-06 product, and which sources those are:
#: plot keys, and the seed's ``density_scan_import``. A test checks that each source named does write the
#: name. A label here has to read true for every one of them.
SHARED = {
    "number_events": ("two_dca", "density_scan_import"),
    "taud_psd": ("taud_psd", "density_scan_import"),
    "lambda_psd": ("taud_psd", "density_scan_import"),
    "lr": ("fwhm_sizes", "density_scan_import"),
    "lz": ("fwhm_sizes", "density_scan_import"),
    "lx_f": ("gaussian_sizes", "density_scan_import"),
    "ly_f": ("gaussian_sizes", "density_scan_import"),
    "theta_f": ("gaussian_sizes", "density_scan_import"),
}

#: The method names are the Fields page's panel titles (``fusion_ui.views.methods``), so that a point
#: on the scatter and the panel it opens call a method the same thing. Where a panel says "on the CA",
#: the label says conditional average: an axis title is read without the page beside it.
LABELS = {
    # -- method_fields, the three 2DCA tracks: the velocity each reads off the average, and the lags it
    # rests on. The contour level exists only for the centroid track.
    "vr_max": Label("v_R, 2DCA max", "m/s"),
    "vz_max": Label("v_Z, 2DCA max", "m/s"),
    "nlags_max": Label("number of lags the 2DCA max slope rests on", NO_UNIT),
    "vr_com": Label("v_R, 2DCA centroid", "m/s"),
    "vz_com": Label("v_Z, 2DCA centroid", "m/s"),
    "nlags_com": Label("number of lags the 2DCA centroid slope rests on", NO_UNIT),
    "level_com": Label(
        "contour level of the 2DCA centroid track, fraction of the average's maximum",
        NO_UNIT,
    ),
    "vr_2dcc": Label("v_R, 2DCC", "m/s"),
    "vz_2dcc": Label("v_Z, 2DCC", "m/s"),
    "nlags_2dcc": Label("number of lags the 2DCC slope rests on", NO_UNIT),
    # The events the conditional average rests on: the count two_dca and the seed write at a reference
    # pixel, and the blob's ``nevents``.
    "number_events": Label("number of events in the conditional average", NO_UNIT),
    # -- method_fields, time delay estimation off the record's cross-correlation, and on the
    # conditional average.
    "vr3_tde": Label("v_R, 3TDE (CC), from the record", "m/s"),
    "vz3_tde": Label("v_Z, 3TDE (CC), from the record", "m/s"),
    "vr2_tde": Label("v_R, 2TDE (CC), from the record", "m/s"),
    "vz2_tde": Label("v_Z, 2TDE (CC), from the record", "m/s"),
    "cc_tde": Label("peak cross-correlation of the TDE from the record", NO_UNIT),
    "vr3_catde": Label("v_R, 3TDE on the conditional average", "m/s"),
    "vz3_catde": Label("v_Z, 3TDE on the conditional average", "m/s"),
    "vr2_catde": Label("v_R, 2TDE on the conditional average", "m/s"),
    "vz2_catde": Label("v_Z, 2TDE on the conditional average", "m/s"),
    # -- blob_parameters, the structure behind each pixel's average. The contour is the one read at a
    # level set per pixel, not at a fixed fraction of the maximum (velocity_contour's area_c).
    "level": Label(
        "contour level read off the average, fraction of its maximum", NO_UNIT
    ),
    "area": Label("area of the zero-lag contour", "m²"),
    "lx_c": Label("ellipse fitted to the zero-lag contour, size in x", "m"),
    "ly_c": Label("ellipse fitted to the zero-lag contour, size in y", "m"),
    "theta_c": Label("ellipse fitted to the zero-lag contour, tilt", "rad"),
    # R varies along a row of the array and Z along a column: lr and lz are the FWHM in each direction.
    "lr": Label("radial FWHM of the conditional average", "m"),
    "lz": Label("poloidal FWHM of the conditional average", "m"),
    "lx_f": Label("Gaussian fit to the conditional average, size in x", "m"),
    "ly_f": Label("Gaussian fit to the conditional average, size in y", "m"),
    "theta_f": Label("Gaussian fit to the conditional average, tilt", "rad"),
    # The two-sided exponential pulse fitted to the PSD of the pixel's own record.
    "taud_psd": Label("duration time τ_d from the PSD fit", "s"),
    "lambda_psd": Label("pulse asymmetry λ from the PSD fit, between 0 and 1", NO_UNIT),
}


def has_label(name):
    """Whether ``name`` has a label."""
    return name in LABELS


def label(name):
    """The text an axis shows for ``name``: its label, or the raw name when it has none."""
    entry = LABELS.get(name)
    return name if entry is None else str(entry)


def axis_title(name, collapse=None):
    """The y-axis title of the multi-shot scatter: the label, then how the pixels were collapsed.

    ``collapse`` is the text of the aggregate (``multishot.AGGREGATES``), or ``None`` for a scalar that
    is already one number per shot. A name without a label shows as it always did, ``name (collapse)``.
    """
    text = label(name)
    return f"{text} ({collapse})" if collapse else text


#: How many characters a line of a rotated axis title can hold on the scatter, which is 520 pixels high.
TITLE_WIDTH = 44

_KEPT_WHOLE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def wrapped(text, width=TITLE_WIDTH):
    """``text`` broken into lines of at most ``width`` characters, joined with ``<br>`` as Plotly takes them.

    A title is one line to Plotly, and a rotated one that is longer than the plot is high is cut off at
    both ends: the longest of these, with its aggregate, runs to 107 characters. The unit in square
    brackets and the aggregate in parentheses are never broken in two, since ``[no`` on one line and
    ``unit]`` on the next reads as nothing. A word longer than ``width`` is left whole.
    """
    glued = _KEPT_WHOLE.sub(lambda m: m.group().replace(" ", " "), text)
    lines = textwrap.wrap(
        glued, width=width, break_long_words=False, break_on_hyphens=False
    )
    return "<br>".join(line.replace(" ", " ") for line in lines)
