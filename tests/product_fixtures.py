"""A small C-Mod-like record for the three decorrelation products, and what the physics API gives on it.

The record is what a preprocessed APD file holds, in miniature: ``frames`` (y, x, time) in float64, normalised
per pixel; ``R`` and ``Z`` in **centimetres**, float32, as the files have them; a stored ``dead`` mask with its
``dead_mask_source``; a ``shot_number`` attribute. It is 3 x 3 pixels at the 1160616027 pitch, sampled every
3 us, of Gaussian pulses moving at (500, 150) m/s, so that a velocity read off it in m/s can be told from one
in cm/s or in m/s over a hundred. The recipe is ``decorrelation/tests/make_pipeline_fixture.py``'s, made
smaller: the 2DCA and the TDE cost seconds on a 3 x 3 record, and the blob parameters 0.7 s a live pixel.

Two masks over the same data:

``full``
    one dead pixel and one live pixel that never reaches the threshold (its average is the empty one):
    eight live pixels, which is what the TDE and the velocity fields need to have estimates to compare.
``sparse``
    three live pixels (an edge pixel, the interior one, and the flat one), for the blob parameters, whose
    duration-time fit costs about 0.7 s a live pixel however small the record.

The expected products are the API called directly, the way the paper's code calls it, on the record in metres
(``direct_*``). They are cached: a test must not modify one.
"""

import functools
from dataclasses import replace

import netCDF4  # noqa: F401 - before h5py, or every netCDF read fails
import numpy as np
import xarray as xr

from fusion_ui.plots._pipeline import Averages, Blobs, Tde, Tracking, pipeline

NY, NX, NT = 3, 3, 4000
SHOT = 1990101001
DT = 3e-6  # [s]
DR, DZ = 3.77e-3, 3.86e-3  # [m]
R0, Z0 = 0.8803, -0.0428  # [m]
VR, VZ = 500.0, 150.0  # [m/s]
ELL, TAU = 3.5e-3, 12e-6  # pulse width [m] and lifetime [s]
PULSES, NOISE, SEED = 6000, 0.05, 20261007
FLAT = (2, 0)  # (x, y): live, uniform noise
LAGS = 11  # a window of 10, widened to the next odd number

CASES = {
    "full": {
        "live": tuple((x, y) for y in range(NY) for x in range(NX) if (x, y) != (0, 1))
    },
    "sparse": {"live": ((0, 0), (1, 1), FLAT)},
}

#: The plan's names, as an independent statement of the three schemas (docs/PHASE_06_DECORRELATION.md).
TRACKS = ("max", "com", "2dcc")
FIELD_VARIABLES = (
    tuple(f"{k}_{t}" for t in TRACKS for k in ("vr", "vz", "nlags", "level"))
    + ("nevents",)
    + tuple(f"{k}_tde" for k in ("vr3", "vz3", "vr2", "vz2", "cc"))
    + tuple(f"{k}_catde" for k in ("vr3", "vz3", "vr2", "vz2"))
)
BLOB_VARIABLES = (
    "nevents level area lx_c ly_c theta_c lr lz lx_f ly_f theta_f taud lam".split()
)


def dead_pixels(case):
    live = set(CASES[case]["live"])
    return tuple((x, y) for y in range(NY) for x in range(NX) if (x, y) not in live)


def make_record(case="full", seed=SEED):
    """The record in centimetres, with the ``case``'s mask stored in it."""
    rng = np.random.RandomState(seed)
    r = R0 + DR * np.arange(NX)
    z = Z0 + DZ * np.arange(NY)
    R, Z = np.meshgrid(r, z)  # (y, x)
    t = DT * np.arange(NT)
    reach = 5 * TAU
    pad = 3 * ELL + reach * max(abs(VR), abs(VZ))
    t0 = rng.uniform(-reach, t[-1] + reach, PULSES)
    r0 = rng.uniform(r[0] - pad, r[-1] + pad, PULSES)
    z0 = rng.uniform(z[0] - pad, z[-1] + pad, PULSES)
    amplitude = rng.exponential(1.0, PULSES)
    half = int(round(reach / DT))
    frames = np.zeros((NY, NX, NT))
    for tk, rk, zk, ak in zip(t0, r0, z0, amplitude):
        i = int(round(tk / DT))
        lo, hi = max(i - half, 0), min(i + half + 1, NT)
        if lo >= hi:
            continue
        s = t[lo:hi] - tk
        dr = R[..., None] - rk - VR * s
        dz = Z[..., None] - zk - VZ * s
        frames[:, :, lo:hi] += ak * np.exp(-(dr**2 + dz**2) / ELL**2 - (s / TAU) ** 2)
    frames += NOISE * rng.standard_normal(frames.shape)
    frames = (frames - frames.mean(-1, keepdims=True)) / frames.std(-1, keepdims=True)
    x, y = FLAT
    flat = rng.uniform(-1.0, 1.0, NT)
    frames[y, x] = (flat - flat.mean()) / flat.std()

    dead = np.zeros((NY, NX), bool)
    for x, y in dead_pixels(case):
        dead[y, x] = True
    return xr.Dataset(
        {"frames": (("y", "x", "time"), frames), "dead": (("y", "x"), dead)},
        coords={
            "R": (("y", "x"), (R * 100).astype(np.float32)),
            "Z": (("y", "x"), (Z * 100).astype(np.float32)),
            "time": t,
        },
        attrs={"shot_number": SHOT, "dead_mask_source": f"synthetic, the {case} mask"},
    )


def in_metres(ds):
    """The record as the API takes it: the test's own statement of the conversion the adapters make."""
    return ds.assign_coords(R=ds.R / 100, Z=ds.Z / 100).load()


# ---------------------------------------------------------------------------
# Settings: the deck's, with one knob moved in every group, so that a setting an adapter
# fails to pass on shows up as a mismatch where the deck's own values would hide it.
# ---------------------------------------------------------------------------

AVERAGES = Averages(threshold=2.0, window=10, check_max=1, single_counting=True)
_DECK = Tracking()
TRACKING = Tracking(
    position_filter=replace(_DECK.position_filter, mask_signal_factor=0.6),
    cross_corr_mask_signal_factor=0.7,
    velocity=_DECK.velocity,
    neighbour_step=1,
)
TDE = Tde(min_cc=0.4, running_mean=True, interpolate=True)
_BLOBS = Blobs()
BLOBS = Blobs(
    gauss_fit=replace(_BLOBS.gauss_fit, aspect_penalty=0.3),
    taud_estimation=replace(_BLOBS.taud_estimation, nperseg=1000.0),
)


def pixel_averages_params():
    from fusion_ui.plots.pixel_averages import PixelAveragesParams

    return PixelAveragesParams(averages=AVERAGES)


def method_fields_params():
    from fusion_ui.plots.method_fields import MethodFieldsParams

    return MethodFieldsParams(averages=AVERAGES, tracking=TRACKING, tde=TDE)


def blob_parameters_params():
    from fusion_ui.plots.blob_parameters import BlobParametersParams

    return BlobParametersParams(
        averages=AVERAGES, neighbour_step=TRACKING.neighbour_step, blobs=BLOBS
    )


# ---------------------------------------------------------------------------
# What the API gives, called directly on the record in metres
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def direct_bank(case):
    ds = in_metres(make_record(case))
    dead = pipeline.dead_mask(ds)
    found = {
        (x, y): pipeline.average(ds, AVERAGES, x, y)
        for x, y in pipeline.references(dead.values)
    }
    return pipeline.stack(found, ds, AVERAGES, dead)


@functools.lru_cache(maxsize=None)
def direct_fields(case):
    ds = in_metres(make_record(case))
    bank = direct_bank(case)
    return pipeline.fields(
        ds,
        lambda x, y: pipeline.at(bank, x, y),
        AVERAGES,
        TRACKING,
        TDE,
        pipeline.dead_mask(ds),
    )


@functools.lru_cache(maxsize=None)
def direct_blobs(case):
    ds = in_metres(make_record(case))
    bank = direct_bank(case)
    return pipeline.blobs(
        ds,
        lambda x, y: pipeline.at(bank, x, y),
        AVERAGES,
        TRACKING.neighbour_step,
        BLOBS,
        pipeline.dead_mask(ds),
    )


def _own_attrs(ds):
    """The attributes the analysis wrote: the store's own ``fusion_ui_*`` ones are not the API's."""
    return {k: v for k, v in ds.attrs.items() if not k.startswith("fusion_ui_")}


def assert_bit_equal(got, expected, what=""):
    """Same variables, dims, dtypes and attributes, and every value equal to the bit (NaN equals NaN)."""
    assert set(got.variables) == set(expected.variables), what
    assert _own_attrs(got) == _own_attrs(expected), what
    for name in expected.variables:
        a, b = got[name], expected[name]
        assert a.dims == b.dims, f"{what} {name}: {a.dims} != {b.dims}"
        assert a.dtype == b.dtype, f"{what} {name}: {a.dtype} != {b.dtype}"
        assert np.array_equal(
            a.values, b.values, equal_nan=a.dtype.kind == "f"
        ), f"{what} {name} differs"
