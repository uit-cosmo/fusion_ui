"""A synthetic ASP file in the layout of ``asp_<shot>.nc``, as ``cmod_functions.asp_functions.get_asp_dataset`` writes it.

The layout is the real files' (``fusion_ui/core/mlp.py`` says what they hold): the six MLP quantities ``ne``, ``Te``,
``Js``, ``Is``, ``Vp`` and ``Vf`` at four pins, each on its own axis ``time_<q>_<pin>``, recorded only while the probe is
in; each with its ρ companion ``rho_<q>_<pin>`` on ``rho_time_<q>_<pin>`` at 10 kHz over the whole shot, exactly 0 where
there is no plasma; and the attributes ``plunge_time``, ``plunge_depth`` (m, at 10 kHz), ``probe_origin``, ``probe_type``
and ``shot_number``. No variable has a units attribute, as in the real files.

Nothing here is a measurement. The probe moves as the 1160616 scans do: parked, in to a base depth, three plunges 4.2 cm
deeper at 0.5 m/s with their tips at 0.7, 1.0 and 1.3 s, back to the base between them, and out to park at the end (or,
with ``dwell=True``, in to one depth that it holds from 0.5 to 1.5 s). ρ falls 0.95 m per metre of depth, as in the
files, from 3 cm at the base depth. The profiles are pure exponentials, so that a fit can be checked against the lengths
they were made with: n_e falls with :data:`L_N` and T_e with :data:`L_T`. The pins differ as the real ones do, by fixed
factors on n_e (and so on J_s and I_s) that average to one, and by offsets of V_f that average to zero, so the mean of
the four is the profile itself.

The MLP's samples are 50 µs apart here, not 0.3 µs, to keep the file small: 20 to a millisecond.
"""

import numpy as np
import xarray as xr

SHOT = 1160616011

#: The e-folding lengths the profiles are made with, in metres.
L_N, L_T = 0.03, 0.02
#: The profiles at ρ = 0: n_e in m^-3 and T_e in eV.
NE0, TE0 = 2.0e19, 60.0
#: ρ at the base depth, and how far it falls per metre of depth.
RHO_BASE, RHO_PER_DEPTH = 0.03, 0.95
PARK, BASE, TIP = 0.006, 0.082, 0.124
TIPS = (0.7, 1.0, 1.3)
SPEED = 0.5  # m/s
#: The probe depth at which the MLP starts and stops recording, in a scan.
RECORDS_FROM = 0.085
#: How the pins differ: a factor on n_e, J_s and I_s, and an offset of V_f in volts; each averages out over the four.
PIN_FACTOR = {0: 0.95, 1: 0.87, 2: 1.13, 3: 1.05}
PIN_VF = {0: -3.0, 1: -2.0, 2: 3.0, 3: 2.0}
PIN_AREA = {0: 0.87e-6, 1: 1.05e-6, 2: 0.86e-6, 3: 1.04e-6}
#: Each pin starts and stops recording a little apart from the others, as on the real files.
PIN_LAG = {0: 0.0, 1: 0.0004, 2: 0.0011, 3: 0.0016}
STEP = 50e-6
E, M_I = 1.602176634e-19, 2 * 1.67262192e-27


def depth_at(time, dwell=False):
    """The probe's depth (m) at ``time`` (s)."""
    time = np.asarray(time, dtype=float)
    if dwell:
        held = 0.109
        depth = np.interp(time, [0.0, 0.3, 0.3 + (held - PARK) / SPEED, 1.5, 1.5 + (held - PARK) / SPEED, 2.1],
                          [PARK, PARK, held, held, PARK, PARK])
        return depth
    ramp = (TIP - BASE) / SPEED
    knots_t, knots_d = [0.0, 0.3, 0.3 + (BASE - PARK) / SPEED], [PARK, PARK, BASE]
    for tip in TIPS:
        knots_t += [tip - ramp, tip, tip + ramp]
        knots_d += [BASE, TIP, BASE]
    knots_t += [1.5, 1.5 + (BASE - PARK) / SPEED, 2.1]
    knots_d += [BASE, PARK, PARK]
    return np.interp(time, knots_t, knots_d)


def rho_at(depth):
    """ρ (m) of a pin at ``depth`` (m)."""
    return RHO_BASE - RHO_PER_DEPTH * (np.asarray(depth) - BASE)


def asp_dataset(shot=SHOT, *, dwell=False, track=True, as_text=False, quantities=None, drop_pin=None):
    """The dataset of one file over a shot of 2.1 s.

    ``dwell`` holds the probe still from 0.5 to 1.5 s instead of scanning; ``track=False`` leaves the plunge attributes
    out; ``as_text`` stores them as the text of their arrays; ``quantities`` keeps only those; ``drop_pin`` leaves one
    pin's every quantity out of the file.
    """
    quantities = quantities or ("ne", "Te", "Js", "Is", "Vp", "Vf")
    rho_time = np.round(np.arange(-0.04, 2.1, 1e-4), 7)
    track_depth = depth_at(rho_time, dwell)
    data_vars, coords = {}, {}
    for pin in range(4):
        if pin == drop_pin:
            continue
        time = np.round(np.arange(0.0, 2.1, STEP), 9)
        depth = depth_at(time, dwell)
        if dwell:
            on = (time >= 0.45 + PIN_LAG[pin]) & (time <= 1.55 - PIN_LAG[pin])
        else:
            on = depth >= RECORDS_FROM + PIN_LAG[pin] * SPEED
        time, depth = time[on], depth[on]
        rho = rho_at(depth)
        ne = PIN_FACTOR[pin] * NE0 * np.exp(-rho / L_N)
        te = TE0 * np.exp(-rho / L_T)
        js = 0.5 * E * ne * np.sqrt(2 * te * E / M_I)
        values = {
            "ne": ne,
            "Te": te,
            "Js": js,
            "Is": js * PIN_AREA[pin],
            "Vf": PIN_VF[pin] - 0.5 * te,
            "Vp": PIN_VF[pin] - 0.5 * te + 2.7 * te,
        }
        rho_track = rho_at(track_depth).astype("float32")
        rho_track[(rho_time < 0.1) | (rho_time > 1.9)] = 0.0  # no mapping without a plasma
        for quantity in quantities:
            coords[f"time_{quantity}_{pin}"] = time
            data_vars[f"{quantity}_{pin}"] = (f"time_{quantity}_{pin}", values[quantity].astype("float32"))
            coords[f"rho_time_{quantity}_{pin}"] = rho_time.astype("float32")
            data_vars[f"rho_{quantity}_{pin}"] = (f"rho_time_{quantity}_{pin}", rho_track)
    attrs = {
        "shot_number": shot,
        "probe_origin": np.array([0.976224, 0.110719, 3.12], dtype="float32"),
        "probe_type": "Mirror Langmuir Probe (MLP)",
    }
    if track:
        plunge_time = rho_time.astype("float32")
        if as_text:
            attrs["plunge_time"] = " ".join(f"{v:.7g}" for v in plunge_time)
            attrs["plunge_depth"] = " ".join(f"{v:.9g}" for v in track_depth)
        else:
            attrs["plunge_time"] = plunge_time
            attrs["plunge_depth"] = track_depth
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def write(folder, ds, shot=SHOT):
    """Write ``ds`` as ``<folder>/asp/asp_<shot>.nc``, netCDF-4 as the real files are, and return the path."""
    target = folder / "asp"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"asp_{shot}.nc"
    ds.to_netcdf(path)
    return path
