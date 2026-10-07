"""The conditional average at every live pixel of a shot, as the reference: the product the others read.

The 2DCA is over 95% of the cost of every decorrelation quantity and depends on four settings only
(``Averages``). Computed once per (shot, settings) and stored, it turns a change of tracking, TDE or blob
settings from a two-hour recompute into minutes: ``method_fields`` and ``blob_parameters`` declare
``requires="pixel_averages"`` and read it back. It is the argument ``two_dca`` makes for one pixel, made
for all of them: one blob per shot rather than a ``two_dca`` run per pixel, which ``requires`` cannot
express and which would list 68 sources per shot on the multi-shot page.

**Batch only.** One reference costs about 30 s on one core, because ``imaging_methods.find_events_and_2dca``
recomputes the reference's cross-correlation with every pixel over the whole record each time: 35 to 70
minutes a shot. Inside the process that serves the whole group that must never start from a page, so the
store computes it only for ``fusion-ui precompute``; a page looks it up, or shows the command that fills
it. Nothing here prints or reports progress: a compute stays pure.

**Preprocessed files only.** The 2DCA's threshold is in standard deviations of the *normalised* record, and
the dead-pixel mask is the one preprocessing stored: the raw file has neither. Without this, ``precompute
--run-day`` would also try every raw file of the day.

The blob, as ``decorrelation.pipeline.stack`` makes it (R and Z in metres)::

    dims    ref_y, ref_x, y, x, time (the lags, in seconds)
    cond_av, cond_repr, cross_corr   (ref_y, ref_x, y, x, time)  float64, NaN at skipped references
    nevents                          (ref_y, ref_x)              int64, 0 where none or skipped
    computed                         (ref_y, ref_x)              bool, False at dead references
    dead                             (y, x)                      bool, the mask used
    attrs   averages (the settings, stamped), dead_mask_source

No scalars: nothing in it is one number per pixel. ``compute`` returns the API's Dataset unchanged (no
renaming, no casting, nothing dropped), since the paper's code reads these blobs by name and a derived
product computed straight after the 2DCA has to be bit-identical to one computed from the stored blob.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from fusion_ui.core import registry
from fusion_ui.plots._pipeline import Averages, pipeline, record
from fusion_ui.views import lag_strip
from fusion_ui.views.bundle import Bundle
from fusion_ui.views.figures import message_figure


@dataclass
class PixelAveragesParams:
    """
    averages: What one reference's 2DCA depends on, bar the pixel: the settings of the conditional
        average, which every product built on this one is lifted from.
    """

    averages: Averages = field(default_factory=Averages)


def lifted_from(params):
    """``PixelAveragesParams`` out of any params that carry ``averages``: what ``method_fields`` and
    ``blob_parameters`` hand the store as their upstream's. Read out of the downstream's own, never built
    from defaults, so that changing the 2DCA threshold gives the bank and everything on it a new key.
    """
    return PixelAveragesParams(averages=copy.deepcopy(params.averages))


def compute(ds, params):
    """The 2DCA at every live reference of the record, stacked into one bank.

    ``ds`` is the preprocessed record sliced to the discharge window. The reference pixels are every one
    the record's mask does not mark dead (``dead_mask``: the mask stored in the file, or the hand-made
    one for a 1160616 shot, or it raises); a live reference with no events keeps an empty average and
    ``nevents`` 0.
    """
    record_m = record(ds)
    dead = pipeline.dead_mask(record_m)
    found = {
        (x, y): pipeline.average(record_m, params.averages, x, y)
        for x, y in pipeline.references(dead.values)
    }
    return pipeline.stack(found, record_m, params.averages, dead)


def best_pixel(bank):
    """``(x, y)`` of the reference whose average rests on the most events; ``None`` if no reference has one.

    What the single-shot view draws, since a pure render has no way to be told which pixel the person
    wants: the one with the best statistics is the one worth looking at first. The Fields page opens any.
    """
    events = np.asarray(bank["nevents"].values)
    if not events.size or events.max() <= 0:
        return None
    y, x = np.unravel_index(int(np.argmax(events)), events.shape)
    return int(x), int(y)


def render(result, params, target):
    """The average and the cross-correlation at a few lags, at the reference with the most events.

    Pure: it returns the lag strip of ``fusion_ui.views`` (one colour scale across the lags of a row, the
    reference marked, the contour at the level the average implies) and never touches Streamlit. The pixel
    cannot be picked here, and a slider needs state this has nowhere to keep, so it shows the pixel with the
    best statistics at the strip's default lags; the Fields page has the picker and the lag controls.
    """
    pixel = best_pixel(result)
    if pixel is None:
        return message_figure(
            "No reference pixel has events: every conditional average of this shot is empty."
        )
    x, y = pixel
    figure = lag_strip.lag_strip(Bundle(shot=target.shot, bank=result, pixel=pixel))
    events = int(np.asarray(result["nevents"].values)[y, x])
    figure.update_layout(
        title=dict(
            text=(
                f"Reference pixel (x={x}, y={y}), the one with the most events ({events})."
                " The Fields page opens any pixel."
            ),
            x=0.0,
            xanchor="left",
            font=dict(size=13),
        ),
        margin=dict(t=80),
    )
    return figure


SPEC = registry.register(
    registry.PlotSpec(
        key="pixel_averages",
        label="Conditional average at every pixel",
        diagnostics=("apd",),
        params=PixelAveragesParams,
        render=render,
        compute=compute,
        batch_only=True,
        preprocessed=True,
        description=(
            "The 2DCA at every live reference pixel: the average, its representativeness and the"
            " cross-correlation at every lag. Batch only, 35-70 min a shot on one core. The input"
            " every other decorrelation product is read off."
        ),
    )
)
