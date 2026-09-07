"""The Langmuir probe view: one (quantity, position) trace at a time.

Also a live spec. ASP/FSP files are ragged -- every quantity x position carries
its own time dimension, 107k-207k samples each, with no shared axis -- so the
choice of what to plot is data-dependent in a way no annotation can express.
That is what :attr:`PlotSpec.choices` is for: the quantity list comes from the
opened file, and the position list from the quantity chosen a moment earlier.
"""

from dataclasses import dataclass

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
    decimate.zoomable_trace(
        trace.time,
        trace.value,
        key=f"probe.{target.key}.{trace.quantity}.{trace.position}",
        x_label="time [s]",
        y_label=f"{trace.quantity}_{trace.position}",
        height=380,
    )

    if trace.rho is not None:
        with st.expander("Flux coordinate ρ for this position"):
            st.caption(
                "ρ is computed on its own, coarser time base -- not a "
                "resampling of the trace above."
            )
            decimate.zoomable_trace(
                trace.rho_time,
                trace.rho,
                key=f"rho.{target.key}.{trace.quantity}.{trace.position}",
                x_label="time [s]",
                y_label="ρ",
                height=280,
            )

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
