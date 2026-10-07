"""The tracks: where each method put the structure at each lag, and the fits and TDE lines against it.

R(tau) above and Z(tau) below, as ``figures.fig_pixel`` draws them through
``plotting_scripts.plot_trajectories``: for each of the three 2DCA tracks every lag it tracked
(faint), the lags its slope rests on (highlighted), and the straight line that slope is. The two
TDE velocities appear as the straight lines they imply through the reference pixel, so the three
numbers can be held against what the track actually did. A velocity is one number and looks equally
plausible whatever it is; a track that latched onto the array edge or never straightens out is
obvious here.

Positions are drawn as displacements from the reference pixel in millimetres, against the lag in
microseconds. The fitted line is the stored velocity as a line through the mean of the fitted
points: that is the least-squares line itself when the estimator is ``lsq`` (the deck's), and the
mean slope the velocity stands for otherwise.
"""

import numpy as np
import plotly.graph_objects as go

from fusion_ui.views import pixel
from fusion_ui.views.figures import (
    TDE_COLOURS,
    TRACK_COLOURS,
    TRACK_SYMBOLS,
    message_figure,
)
from fusion_ui.views.methods import TRACKS

#: Metres to millimetres, seconds to microseconds: m/s * us = 1e-3 mm.
MM = 1e3
US = 1e6
CORNER_STEP = 0.11

COMPONENTS = (("r", "R", "v_R", "vr"), ("z", "Z", "v_Z", "vz"))


def _lag_axis(bundle):
    fields = bundle.fields
    if fields is not None and "time" in fields.coords:
        return np.asarray(fields["time"].values, dtype=float)
    if bundle.bank is not None:
        return pixel.lags(bundle)
    return np.arange(fields.sizes["time"], dtype=float)


def _pad(low, high):
    span = max(high - low, 1.0)
    return low - 0.12 * span, high + 0.12 * span


def tracks_figure(bundle):
    """R(tau) and Z(tau) of the three tracks at the pixel in view, with the fits and the TDE lines."""
    if bundle.fields is None:
        return message_figure("The velocity fields of this shot are not computed.")
    why = pixel.problem(bundle, need_bank=False)
    if why:
        return message_figure(why)
    fields = bundle.fields
    x, y = bundle.pixel
    R, Z = bundle.grid
    reference = {"r": float(R[y, x]), "z": float(Z[y, x])}
    lag = _lag_axis(bundle) * US
    events = (
        float(fields["nevents"].values[y, x]) if "nevents" in fields else float("nan")
    )

    # What every track did, before anything is drawn: the vertical range comes from the tracks, not
    # from the TDE lines, which can be far outside it and are clipped.
    shown = {}
    for track in TRACKS:
        names = [f"pos_r_{track.key}", f"pos_z_{track.key}", f"fit_{track.key}"]
        if not all(name in fields for name in names):
            continue
        r = (np.asarray(fields[names[0]].values)[y, x] - reference["r"]) * MM
        z = (np.asarray(fields[names[1]].values)[y, x] - reference["z"]) * MM
        fit = np.asarray(fields[names[2]].values)[y, x].astype(bool)
        if np.isfinite(r).any() or np.isfinite(z).any():
            shown[track.key] = dict(track=track, r=r, z=z, fit=fit)
    if not shown:
        return message_figure(f"No track could be followed at pixel (x={x}, y={y}).")

    layout = dict(
        xaxis=dict(
            domain=[0.07, 0.98],
            anchor="y",
            matches="x2",
            showticklabels=False,
            zeroline=False,
        ),
        xaxis2=dict(
            domain=[0.07, 0.98],
            anchor="y2",
            title=dict(text="lag [µs]", standoff=4),
            zeroline=False,
        ),
        yaxis=dict(domain=[0.57, 0.97], anchor="x", zeroline=False),
        yaxis2=dict(domain=[0.1, 0.5], anchor="x2", zeroline=False),
        # The reference lag, tau = 0, on both panels.
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=0,
                y1=1,
                xref=x_axis,
                yref=f"{y_axis} domain",
                line=dict(color="grey", dash="dot", width=1),
            )
            for x_axis, y_axis in (("x", "y"), ("x2", "y2"))
        ],
    )
    traces, annotations = [], []
    corners = {}
    for row, (component, axis_label, speed_label, speed_key) in enumerate(COMPONENTS):
        axes = ("x" if row == 0 else "x2", "y" if row == 0 else "y2")
        values = [info[component] for info in shown.values()]
        finite = np.concatenate([v[np.isfinite(v)] for v in values])
        low, high = _pad(
            *((float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0))
        )
        layout[axes[1].replace("y", "yaxis")]["range"] = [low, high]
        layout[axes[1].replace("y", "yaxis")]["title"] = dict(
            text=f"{axis_label} − reference [mm]", standoff=4
        )

        traces.append(
            go.Scatter(
                x=[lag[0], lag[-1]],
                y=[0, 0],
                mode="lines",
                line=dict(color="grey", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
                xaxis=axes[0],
                yaxis=axes[1],
            )
        )
        for key, info in shown.items():
            track, position, fit = info["track"], info[component], info["fit"]
            tracked = np.isfinite(position)
            colour, symbol = TRACK_COLOURS[key], TRACK_SYMBOLS[key]
            fitted = tracked & fit
            traces.append(
                go.Scatter(
                    x=lag[tracked],
                    y=position[tracked],
                    mode="markers",
                    marker=dict(color=colour, symbol=symbol, size=6, opacity=0.5),
                    name=track.label,
                    legendgroup=key,
                    showlegend=row == 0,
                    hovertemplate=f"{track.label}<br>τ %{{x:.1f}} µs, %{{y:.2f}} mm<extra></extra>",
                    xaxis=axes[0],
                    yaxis=axes[1],
                )
            )
            traces.append(
                go.Scatter(
                    x=lag[fitted],
                    y=position[fitted],
                    mode="lines+markers",
                    line=dict(color=colour, width=3),
                    marker=dict(color=colour, symbol=symbol, size=8),
                    name=f"{track.label}: lags of the fit",
                    legendgroup=key,
                    showlegend=False,
                    hovertemplate=f"{track.label}, in the fit<br>τ %{{x:.1f}} µs, %{{y:.2f}} mm<extra></extra>",
                    xaxis=axes[0],
                    yaxis=axes[1],
                )
            )
            speed = (
                float(fields[f"{speed_key}_{key}"].values[y, x])
                if f"{speed_key}_{key}" in fields
                else np.nan
            )
            if np.isfinite(speed) and fitted.any():
                # The velocity as a line through the mean of the fitted points: the least-squares
                # line itself when the estimator is lsq.
                t_fit = lag[fitted]
                line = (speed / MM) * (t_fit - t_fit.mean()) + position[fitted].mean()
                traces.append(
                    go.Scatter(
                        x=t_fit,
                        y=line,
                        mode="lines",
                        line=dict(color=colour, width=1.5, dash="dash"),
                        legendgroup=key,
                        showlegend=False,
                        hoverinfo="skip",
                        xaxis=axes[0],
                        yaxis=axes[1],
                    )
                )
                side = (0.02, "left") if speed >= 0 else (0.98, "right")
                depth = corners.get((row, side[1]), 0)
                corners[(row, side[1])] = depth + 1
                annotations.append(
                    dict(
                        text=f"{speed_label} = {speed:.0f} m/s",
                        xref="x domain" if row == 0 else "x2 domain",
                        yref="y domain" if row == 0 else "y2 domain",
                        x=side[0],
                        y=0.96 - CORNER_STEP * depth,
                        xanchor=side[1],
                        yanchor="top",
                        showarrow=False,
                        font=dict(size=11, color=colour),
                        bgcolor="rgba(255,255,255,0.75)",
                    )
                )
        for points, tde_label in (("3", "3TDE"), ("2", "2TDE")):
            name = f"{speed_key[:2]}{points}_tde"
            speed = float(fields[name].values[y, x]) if name in fields else np.nan
            if not np.isfinite(speed):
                continue
            traces.append(
                go.Scatter(
                    x=[lag[0], lag[-1]],
                    y=[speed * lag[0] / MM, speed * lag[-1] / MM],
                    mode="lines",
                    line=dict(color=TDE_COLOURS[points], width=1.6, dash="dashdot"),
                    name=f"{tde_label} (CC)",
                    legendgroup=f"tde{points}",
                    showlegend=row == 0,
                    hovertemplate=f"{tde_label}: {speed:.0f} m/s<extra></extra>",
                    xaxis=axes[0],
                    yaxis=axes[1],
                )
            )
    annotations.append(
        dict(
            text=f"{events:.0f} events" if np.isfinite(events) else "",
            xref="paper",
            yref="paper",
            x=0.98,
            y=1.0,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color="dimgrey"),
        )
    )
    layout.update(
        annotations=annotations,
        height=620,
        margin=dict(l=10, r=10, t=30, b=70),
        legend=dict(orientation="h", x=0.0, y=-0.01, yanchor="top"),
        title=dict(
            text=f"tracks at reference pixel (x={x}, y={y})",
            font=dict(size=14),
            x=0.07,
            xanchor="left",
        ),
    )
    return go.Figure(data=traces, layout=go.Layout(**layout))
