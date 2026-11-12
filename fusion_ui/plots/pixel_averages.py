"""The conditional average (2DCA) at every live pixel of a shot, as the reference: one bank per shot.

The 2DCA is nearly all the cost of every decorrelation quantity and depends only on ``Averages``. Stored
once per (shot, settings), it lets ``method_fields`` and ``blob_parameters`` be recomputed in minutes:
both declare ``requires="pixel_averages"`` and read it back.

**Batch only.** One reference costs about 30 s on one core, 35 to 70 minutes a shot, so only
``fusion-ui precompute`` computes it; a page looks it up, or shows the command that fills it.

**Preprocessed files only.** The threshold is in standard deviations of the normalised record, and the
dead-pixel mask is the one preprocessing stored.

The blob is ``imaging_methods.decorrelation.stack``'s Dataset, unchanged (R and Z in metres)::

    dims    ref_y, ref_x, y, x, time (the lags, in seconds)
    cond_av, cond_repr, cross_corr   (ref_y, ref_x, y, x, time)  float64, NaN at skipped references
    nevents                          (ref_y, ref_x)              int64, 0 where none or skipped
    computed                         (ref_y, ref_x)              bool, False at dead references
    dead                             (y, x)                      bool, the mask used
    attrs   averages (the settings, stamped), dead_mask_source

It writes no scalars: nothing in it is one number per pixel.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from fusion_ui.core import registry
from fusion_ui.plots._pipeline import Averages, hand_made, pipeline, record
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
    """``PixelAveragesParams`` out of any params that carry ``averages``: the ``upstream_params`` of
    ``method_fields`` and ``blob_parameters``. Read out of the downstream's own, never built from
    defaults, so that changing the 2DCA threshold gives the bank and everything on it a new key.
    """
    return PixelAveragesParams(averages=copy.deepcopy(params.averages))


def compute(ds, params):
    """The 2DCA at every live reference of the record, stacked into one bank.

    ``ds`` is the preprocessed record cut to the discharge window. The references are the pixels the
    record's mask does not mark dead (``dead_mask``: the mask stored in the file, or the hand-made one
    of its run day, or it raises). A live reference with no events keeps an empty average.
    """
    record_m = record(ds)
    dead = pipeline.dead_mask(record_m, hand_made=hand_made())
    found = {
        (x, y): pipeline.average(record_m, params.averages, x, y)
        for x, y in pipeline.references(dead.values)
    }
    return pipeline.stack(found, record_m, params.averages, dead)


def best_pixel(bank):
    """``(x, y)`` of the reference whose average rests on the most events; ``None`` if no reference has one."""
    events = np.asarray(bank["nevents"].values)
    if not events.size or events.max() <= 0:
        return None
    y, x = np.unravel_index(int(np.argmax(events)), events.shape)
    return int(x), int(y)


def render(result, params, target):
    """The average and the cross-correlation at a few lags, at the reference with the most events.

    The lag strip of ``fusion_ui.views`` at its default lags. A pure render cannot be told which pixel
    the person wants, so it shows the one with the best statistics; the Fields page opens any.
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
