"""The Langmuir probe view: one (quantity, position) trace at a time.

Also a live spec. ASP/FSP files are ragged -- every quantity x position carries
its own time dimension, 107k-207k samples each, with no shared axis -- so the
choice of what to plot is data-dependent in a way no annotation can express.
That is what :attr:`PlotSpec.choices` is for: the quantity list comes from the
opened file, and the position list from the quantity chosen a moment earlier.
"""

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from fusion_ui.core import decimate, probes, registry


@dataclass
class ProbeTraceParams:
    """
    quantity: Measured quantity, e.g. ne, Te, Vf.
    position: Probe position index along the reciprocating head.
    """

    quantity: str = ""
    position: int = 0


def choices(ds, path, chosen):
    """Quantity and position lists, read off the file rather than declared.

    ``chosen`` carries the fields decided so far in this pass, so the position
    list narrows to the quantity that was just picked.
    """
    available = probes.quantities_and_positions(ds)
    if path == "quantity":
        return tuple(available) or None
    if path == "position":
        quantity = chosen.get("quantity")
        if quantity in available:
            return tuple(available[quantity])
        return None
    return None


def render(ds, params, target):
    available = probes.quantities_and_positions(ds)
    if not available:
        st.warning("No probe quantities found in this file.", icon="⚠️")
        return None
    if params.quantity not in available:
        st.warning("Pick a quantity in the sidebar.", icon="⚠️")
        return None

    trace = probes.load_trace(ds, params.quantity, int(params.position))
    x_decimated, y_decimated = decimate.envelope(trace.time, trace.value)

    figure = go.Figure(go.Scatter(x=x_decimated, y=y_decimated, mode="lines"))
    figure.update_layout(
        xaxis_title="time [s]",
        yaxis_title=f"{trace.quantity}_{trace.position}",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        f"{trace.time.size} samples on this position's own time base, "
        f"{x_decimated.size} plotted after min/max-envelope decimation."
    )

    if trace.rho is not None:
        with st.expander("Flux coordinate ρ for this position"):
            st.caption(
                "ρ is computed on its own, coarser time base -- not a "
                "resampling of the trace above."
            )
            rho_x, rho_y = decimate.envelope(trace.rho_time, trace.rho)
            rho_figure = go.Figure(go.Scatter(x=rho_x, y=rho_y, mode="lines"))
            rho_figure.update_layout(
                xaxis_title="time [s]",
                yaxis_title="ρ",
                height=280,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(rho_figure, use_container_width=True)

    geometry = probes.probe_geometry(ds)
    if geometry["probe_type"]:
        origin = geometry["probe_origin"]
        st.caption(
            geometry["probe_type"]
            + (f" · origin {origin}" if origin is not None else "")
        )
    return None


SPEC = registry.register(
    registry.PlotSpec(
        key="probe_trace",
        label="Probe trace",
        diagnostics=("asp", "fsp"),
        params=ProbeTraceParams,
        render=render,
        choices=choices,
        description="One quantity at one probe position, on its own time base.",
    )
)
