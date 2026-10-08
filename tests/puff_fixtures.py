"""A raw APD record with a gas puff in it, for the tests of the dead-pixel view on the puff window.

Built the way the real ones look: every pixel sits at the digitizer's dark level (about -1.1 V) until the puff, and
then lights up with red fluctuations on top of it; one pixel is dead (white noise at the dark level, whatever the
plasma does) and one has no samples. The record is short and fast enough to run through the whole analysis in a
second or two: 120 ms at 1 MHz, whose Nyquist frequency still leaves the estimator's noise band (300-900 kHz) its
lower part.

``SCENARIOS`` are discharge windows that tell the whole record from a cut of it. The gas puff is found against the
dark level at the *start of the record*, so a window that begins after the record does can give another answer when
the record is cut to it first, and a window that begins with the record cannot:

``two puffs``
    The window starts in the dark gap between two puffs. On the whole record the first puff has come and gone: the
    gap is a dip inside the window, reported and not cut. Cut to the window, the record starts dark and the first
    puff does not exist, so the gap would be cut as the dark before the puff.
``just before the rise``
    The window starts 3 ms before the puff rises, within the 5 ms the dark level is read from: cut to the window,
    that stretch is half light, and the record "starts with the light on".
``from the start``
    The control: a window that starts with the record, in the dark. Cutting it changes nothing.

``discharge_entry`` is the discharge-DB entry that gives a shot such a window, and ``before_the_puff_window`` ages a
result to what the server cached before the view looked for the puff.
"""

from collections import namedtuple

import numpy as np
import xarray as xr
from scipy import signal

DT = 1e-6
STEP = 1e-3  # the digitizer's step: the raw files are quantised to it
DARK = -1.11  # V, the dark level of the C-Mod array
LIGHT = 0.8  # V above the dark level, at the array mean
DEAD = (0, 3)
EMPTY = (2, 0)
SHAPE = (3, 4)

Scenario = namedtuple("Scenario", "puffs window duration")

SCENARIOS = {
    "two puffs": Scenario(puffs=[(0.010, 0.030), (0.050, 0.115)], window=(0.035, 0.115), duration=0.12),
    "just before the rise": Scenario(puffs=[(0.060, 0.115)], window=(0.057, 0.115), duration=0.12),
    "from the start": Scenario(puffs=[(0.040, 0.115)], window=(0.0, 0.115), duration=0.12),
}


def discharge_entry(shot, window):
    """The discharge DB's entry that gives ``shot`` this ``(t_start, t_end)``: what a record's window comes from."""
    return {
        "shot_number": shot,
        "plasma_current": 0.55,
        "line_averaged_density": 1.4,
        "greenwald_fraction": 0.7,
        "t_start": window[0],
        "t_end": window[1],
        "mlp_mode": "",
        "comment": "",
    }


def before_the_puff_window(result):
    """``result`` as the dead-pixel view stored it before it looked for the puff: no smoothed array mean and none
    of the window attributes (the 111 results cached on the server until ``precompute dead_pixels --force``)."""
    old = result.drop_vars(["puff_signal", "puff_time"])
    for name in list(old.attrs):
        if name.startswith("puff_") or name.endswith("_window"):
            del old.attrs[name]
    return old


def record(scenario, seed=1, first=0.0):
    """The raw record of ``scenario`` as a Dataset: ``frames`` on ``(y, x, time)`` and a ``time`` coordinate."""
    rng = np.random.default_rng(seed)
    n = int(round(scenario.duration / DT))
    time = first + np.arange(n) * DT
    on = np.zeros(n)
    for start, stop in scenario.puffs:
        on[int(round(start / DT)) : int(round(stop / DT))] = 1.0
    phi = np.exp(-DT / 20e-6)
    shared = signal.lfilter([1], [1, -phi], rng.normal(size=n) * np.sqrt(1 - phi**2))
    light = on * (LIGHT * (1 + 0.5 * shared)[None, None, :] + 0.01 * rng.normal(size=(*SHAPE, n)))
    frames = DARK + light + 0.002 * rng.normal(size=(*SHAPE, n))
    frames[DEAD] = DARK + 0.01 * rng.normal(size=n)
    frames = np.round(frames / STEP) * STEP
    frames[EMPTY] = np.nan
    return xr.Dataset({"frames": (("y", "x", "time"), frames.astype("float32"))}, coords={"time": time})
