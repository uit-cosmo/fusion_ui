"""Every method's velocity field over the whole array, read off the stored conditional averages.

Seven ways to read a velocity at every live pixel, on one record:

- three 2DCA tracks, through the conditional average and the cross-correlation of ``pixel_averages``: the
  average tracked by its sub-pixel maximum (suffix ``_max``), the average by the centroid of its contour
  (``_com``) and the cross-correlation by its maximum (``_2dcc``);
- three- and two-point time delay estimation off the record's cross-correlation (``_tde``);
- the same TDE applied to the conditional average itself (``_catde``).

``compute`` is ``imaging_methods.decorrelation.fields``, returned unchanged: velocities in m/s, R and Z
in metres, lags in seconds. It takes 1 to 3 minutes a shot once the bank exists, so a page may compute it.

The blob (``dims y, x, time``)::

    for each track T in max, com, 2dcc:
      vr_T, vz_T, nlags_T      (y, x)        the velocity [m/s] and the lags its slope rests on
      level_T                  (y, x)        the contour level [fraction of the maximum]; NaN for a maximum track
      pos_r_T, pos_z_T         (y, x, time)  the tracked position [m], NaN where the tracker found nothing
      fit_T                    (y, x, time)  bool: the lags the slope rests on
    nevents                                  (y, x)  events behind the conditional average
    vr3_tde vz3_tde vr2_tde vz2_tde cc_tde   (y, x)  TDE off the record: three- and two-point v_R, v_Z, and the
                                                     peak cross-correlation [dimensionless]
    vr3_catde vz3_catde vr2_catde vz2_catde  (y, x)  TDE on the conditional average
    dead                                     (y, x)  bool: the mask the bank was computed with
    attrs   min_cc, averages, tracking, tde (the settings, stamped), dead_mask_source

**Settings.** ``averages`` selects the bank, so moving the 2DCA threshold gives the bank and this a new
entry, while ``tracking`` and ``tde`` move only this one and reuse the bank. Which pixels are drawn
(minimum lags, minimum events) is view state, never a parameter.
"""

from dataclasses import dataclass, field

from fusion_ui.core import registry
from fusion_ui.plots import pixel_averages
from fusion_ui.plots._pipeline import (
    Averages,
    Tde,
    Tracking,
    average_at,
    bank_mask,
    live_scalars,
    pipeline,
    record,
)
from fusion_ui.views import panels
from fusion_ui.views.bundle import Bundle, Cuts

#: The scalars this writes per live pixel: each name, as the store records it, next to the variable of the
#: blob it is read from. Written out rather than read off the API: a name is permanent once a batch has
#: written it. ``level_max`` and ``level_2dcc`` are not scalars: a maximum track has no level.
SCALARS = {
    "vr_max": "vr_max",
    "vz_max": "vz_max",
    "nlags_max": "nlags_max",
    "vr_com": "vr_com",
    "vz_com": "vz_com",
    "nlags_com": "nlags_com",
    "level_com": "level_com",
    "vr_2dcc": "vr_2dcc",
    "vz_2dcc": "vz_2dcc",
    "nlags_2dcc": "nlags_2dcc",
    "number_events": "nevents",
    "vr3_tde": "vr3_tde",
    "vz3_tde": "vz3_tde",
    "vr2_tde": "vr2_tde",
    "vz2_tde": "vz2_tde",
    "cc_tde": "cc_tde",
    "vr3_catde": "vr3_catde",
    "vz3_catde": "vz3_catde",
    "vr2_catde": "vr2_catde",
    "vz2_catde": "vz2_catde",
}


@dataclass
class MethodFieldsParams:
    """
    averages: The 2DCA the fields are read off, as ``pixel_averages`` was computed with. It selects the
        bank this is built on.
    tracking: How the three 2DCA tracks are read off an average: the filters, the velocity estimator and
        the contour level's neighbour step.
    tde: Time delay estimation, off the record and on the conditional average.
    """

    averages: Averages = field(default_factory=Averages)
    tracking: Tracking = field(default_factory=Tracking)
    tde: Tde = field(default_factory=Tde)


def compute(ds, params, upstream):
    """Every method's field at every live pixel, from the stored bank and the record.

    ``upstream`` is the ``pixel_averages`` result: the mask is the bank's own, and every live
    reference's average is read back out of it. No ``pixels`` is passed: the TDE's result depends on
    the order of the pixels, and the API's default order is the one to keep.
    """
    return pipeline.fields(
        record(ds),
        average_at(upstream),
        params.averages,
        params.tracking,
        params.tde,
        bank_mask(upstream),
    )


def scalars(result):
    """The :data:`SCALARS` at every live pixel."""
    return live_scalars(result, SCALARS)


def render(result, params, target):
    """One v_R map per method, on one diverging scale.

    The seven panels of ``fusion_ui.views`` at the cuts the Fields page starts from. The line across
    the top of the figure says which cuts those are, since they differ by method, so the title does not.
    """
    bundle = Bundle(
        shot=target.shot,
        fields=result,
        neighbour_step=params.tracking.neighbour_step,
        cuts=Cuts(),
    )
    figure = panels.velocity_panels(bundle, mode="vr")
    figure.update_layout(
        title=dict(
            text=(
                "v_R of every method, on one scale.<br>"
                "The Fields page sets the cuts and draws the arrows."
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
        key="method_fields",
        label="Velocity fields of every method",
        diagnostics=("apd",),
        params=MethodFieldsParams,
        render=render,
        compute=compute,
        scalars=scalars,
        requires="pixel_averages",
        upstream_params=pixel_averages.lifted_from,
        preprocessed=True,
        description=(
            "Every method's velocity at every live pixel: three 2DCA tracks (maximum, contour centroid,"
            " cross-correlation maximum) and three- and two-point TDE off the record and on the"
            " conditional average. Read off pixel_averages; emits twenty names per live pixel."
        ),
    )
)
