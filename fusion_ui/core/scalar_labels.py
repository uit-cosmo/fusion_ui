"""What the multi-shot axis calls a stored scalar: the quantity and its unit.

The store's names are permanent keys (``vr_com``); a person reading a scatter needs to know what the
number is (``v_R, 2DCA centroid [m/s]``). :data:`LABELS` holds that for the names ``method_fields`` and
``blob_parameters`` write. A name without an entry shows as its raw name: nothing is guessed for it.

Every label ends in its unit in square brackets, one of :data:`UNITS`. A number with no unit says
``[no unit]``, so that a forgotten unit cannot be mistaken for none. A label is per name, not per source:
the axis shows it whichever spec wrote the value, so it says what the quantity is and names no spec.
"""

import re
import textwrap
from dataclasses import dataclass

#: The unit of a number that has none: a count, a fraction, a coefficient.
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


#: The method names are the Fields page's panel titles (``fusion_ui.views.methods``), so that a point on
#: the scatter and the panel it opens call a method the same thing. Where a panel says "on the CA", the
#: label says conditional average: an axis title is read without the page beside it.
LABELS = {
    # method_fields: the three 2DCA tracks
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
    "number_events": Label("number of events in the conditional average", NO_UNIT),
    # method_fields: time delay estimation off the record, and on the conditional average
    "vr3_tde": Label("v_R, 3TDE (CC), from the record", "m/s"),
    "vz3_tde": Label("v_Z, 3TDE (CC), from the record", "m/s"),
    "vr2_tde": Label("v_R, 2TDE (CC), from the record", "m/s"),
    "vz2_tde": Label("v_Z, 2TDE (CC), from the record", "m/s"),
    "cc_tde": Label("peak cross-correlation of the TDE from the record", NO_UNIT),
    "vr3_catde": Label("v_R, 3TDE on the conditional average", "m/s"),
    "vz3_catde": Label("v_Z, 3TDE on the conditional average", "m/s"),
    "vr2_catde": Label("v_R, 2TDE on the conditional average", "m/s"),
    "vz2_catde": Label("v_Z, 2TDE on the conditional average", "m/s"),
    # blob_parameters. R varies along a row of the array and Z along a column.
    "area": Label("area of the zero-lag contour", "m²"),
    "lr": Label("radial FWHM of the conditional average", "m"),
    "lz": Label("poloidal FWHM of the conditional average", "m"),
    "taud_psd": Label("duration time τ_d from the PSD fit", "s"),
    "lambda_psd": Label("pulse asymmetry λ from the PSD fit, between 0 and 1", NO_UNIT),
}


def label(name):
    """What an axis shows for ``name``: its label, or the raw name when it has none."""
    return str(LABELS.get(name, name))


def option(name):
    """What the scalar picker shows for ``name``: ``name — label``, or the raw name when it has no label.

    The name comes first so that the picker can be searched by it.
    """
    return f"{name} — {LABELS[name]}" if name in LABELS else name


def axis_title(name, collapse=None):
    """The y-axis title of the multi-shot scatter: the label, then how the pixels were collapsed.

    ``collapse`` is the text of the aggregate (``multishot.AGGREGATES``), or ``None`` for a scalar that
    is already one number per shot.
    """
    text = label(name)
    return f"{text} ({collapse})" if collapse else text


#: How many characters a line of a rotated axis title can hold on the scatter, which is 520 pixels high.
TITLE_WIDTH = 44

_KEPT_WHOLE = re.compile(r"\[[^\]]*\]|\([^)]*\)")

#: ``textwrap`` breaks only at ASCII whitespace, so a no-break space keeps a unit or an aggregate on one
#: line. Written as a code point, since the character is invisible.
_NBSP = chr(0xA0)


def wrapped(text, width=TITLE_WIDTH):
    """``text`` broken into lines of at most ``width`` characters, joined with ``<br>`` as Plotly takes them.

    A title is one line to Plotly, and a rotated one longer than the plot is high is cut off at both ends.
    The unit in square brackets and the aggregate in parentheses are never broken in two, since ``[no`` on
    one line and ``unit]`` on the next reads as nothing. A word longer than ``width`` is left whole.
    """
    glued = _KEPT_WHOLE.sub(lambda m: m.group().replace(" ", _NBSP), text)
    lines = textwrap.wrap(
        glued, width=width, break_long_words=False, break_on_hyphens=False
    )
    return "<br>".join(line.replace(_NBSP, " ") for line in lines)
