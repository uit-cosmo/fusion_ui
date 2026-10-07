"""Synthetic ``pixel_averages``, ``method_fields`` and ``blob_parameters`` blobs, and stand-in specs that serve them.

The three product specs (J3) are written in parallel with the Fields page, so this builds the blobs
from the frozen schemas in docs/PHASE_06_DECORRELATION.md ("Products") and registers cached stand-in
specs under the permanent keys: ``pixel_averages`` batch only, the other two chained to it. Each
stand-in's ``compute`` ignores the record and returns a synthetic dataset, so the page, the store
and the ledger are exercised end to end -- ``seed`` fills them through ``store.result(batch=True)``,
as ``fusion-ui precompute`` does -- without an analysis. Parameter dataclasses mirror the plan's
(``averages``, ``tracking``, ``tde``, ``neighbour_step``, ``blobs``) with the deck's C-Mod values.

The physics is a blob moving at a known velocity across a 10 x 9 array with C-Mod's geometry and
the 22 dead pixels of the hand-made 1160616 mask, so that a test can say what a pixel's velocity,
track and fitted lags are and read them back off the figure. Every number is deterministic.

``World.install`` replaces whatever is registered under the three keys for the length of a test and
puts it back, so these tests keep working once the real specs are registered.
"""

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store

NY, NX = 10, 9
DT = 0.5e-6  # the lag step: the record's sampling interval
KEYS = ("pixel_averages", "method_fields", "blob_parameters")

#: The hand-made 1160616 mask, y = 0 first. X is dead: 22 of the 90.
DEAD_ART = """
..X......
XX.......
.X.X....X
.......X.
........X
X.......X
...X.....
X...X...X
.......XX
XX.X..XXX
"""
DEAD = np.array([[c == "X" for c in row] for row in DEAD_ART.split()], dtype=bool)

#: What ``average_cache.stamp`` writes into a bank, and what ``blob_parameters`` records.
AVERAGES_STAMP = "threshold=2.5; window=60; check_max=1; single_counting=True"
BLOB_NAMES = (
    "nevents",
    "level",
    "area",
    "lx_c",
    "ly_c",
    "theta_c",
    "lr",
    "lz",
    "lx_f",
    "ly_f",
    "theta_f",
    "taud",
    "lam",
)


# ---------------------------------------------------------------------------
# Parameters: the plan's structure, the deck's values
# ---------------------------------------------------------------------------


@dataclass
class Averages:
    threshold: float = 2.5
    window: int = 60
    check_max: int = 1
    single_counting: bool = True


@dataclass
class PixelAveragesParams:
    averages: Averages = field(default_factory=Averages)


@dataclass
class Tracking:
    mask_signal_factor: float = 0.65
    cross_corr_mask_signal_factor: float = 0.75
    estimator: str = "lsq"
    neighbour_step: int = 1


@dataclass
class Tde:
    min_cc: float = 0.5
    running_mean: bool = True
    interpolate: bool = True


@dataclass
class MethodFieldsParams:
    averages: Averages = field(default_factory=Averages)
    tracking: Tracking = field(default_factory=Tracking)
    tde: Tde = field(default_factory=Tde)


@dataclass
class Blobs:
    size_penalty: float = 5.0
    aspect_penalty: float = 0.0
    tilt_penalty: float = 0.0


@dataclass
class BlobParametersParams:
    averages: Averages = field(default_factory=Averages)
    neighbour_step: int = 1
    blobs: Blobs = field(default_factory=Blobs)


# ---------------------------------------------------------------------------
# The world the blobs describe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """A 10 x 9 array with C-Mod's pitch, a mask, and a blob velocity at every reference pixel."""

    R: np.ndarray
    Z: np.ndarray
    dead: np.ndarray
    vr: np.ndarray  # the velocity the blob has when the reference is (x, y), m/s
    vz: np.ndarray


def make_geometry(dead=DEAD):
    x, y = np.meshgrid(np.arange(NX), np.arange(NY))
    # Nearly rectilinear, as the real array is: a pitch of about 3.9 mm and a tilt of a tenth of a
    # millimetre across the array, so a heatmap on column and row means is a good approximation.
    R = 0.8803 + 0.0039 * x + 0.00012 * y
    Z = -0.0428 + 0.0039 * y - 0.00010 * x
    vr = 300.0 + 40.0 * x + 10.0 * y
    vz = 200.0 - 25.0 * x + 8.0 * (y - 5)
    return Geometry(R=R, Z=Z, dead=np.asarray(dead, dtype=bool), vr=vr, vz=vz)


@dataclass
class Options:
    """What differs between worlds. Every field has the value the main tests use."""

    seed: int = 7
    mask_source: str = "hand-made, 1160616"
    #: live pixels (x, y) whose reference has no events: no average, ``nevents`` 0
    no_events: tuple = ()
    #: live pixels (x, y) where every track's fit fails: one lag, NaN velocity
    failed: tuple = ()
    #: half-width in microseconds of the lags a pixel's slope rests on, by pixel; the default is 4.5
    fit_half_width_us: dict = field(default_factory=dict)
    #: the settings stamp written to ``method_fields``
    settings: str = "velocity=lsq; position_filter=0.65/0.75"


def lag_axis(window):
    """``window + 1`` lags centred on zero, in seconds."""
    return (np.arange(window + 1) - window // 2) * DT


def _noise(rng, shape, scale):
    return rng.normal(scale=scale, size=shape)


def build_bank(geo, window=60, options=None):
    """The ``pixel_averages`` dataset: three fields per reference pixel, as the schema says."""
    options = options or Options()
    rng = np.random.default_rng(options.seed)
    t = lag_axis(window)
    live = ~geo.dead
    for x, y in options.no_events:
        live = live.copy()
        live[y, x] = False
    # What the 2DCA is: a Gaussian of 4.5 mm that moves with the reference's velocity and decays.
    sigma, decay, decay_cc = 0.0045, 5e-6, 9e-6
    amplitude = 4.4 * (0.85 + 0.3 * rng.random((NY, NX)))
    centre_r = geo.R[:, :, None] + geo.vr[:, :, None] * t  # (ref_y, ref_x, t)
    centre_z = geo.Z[:, :, None] + geo.vz[:, :, None] * t
    d2 = (geo.R[None, None, :, :, None] - centre_r[:, :, None, None, :]) ** 2 + (
        geo.Z[None, None, :, :, None] - centre_z[:, :, None, None, :]
    ) ** 2
    env = np.exp(-(t**2) / (2 * decay**2))[None, None, None, None, :]
    env_cc = np.exp(-(t**2) / (2 * decay_cc**2))[None, None, None, None, :]
    cond_av = 0.06 + amplitude[:, :, None, None, None] * env * np.exp(
        -d2 / (2 * sigma**2)
    )
    cross_corr = 0.054 + 0.946 * env_cc * np.exp(-d2 / (2 * (1.3 * sigma) ** 2))
    cond_repr = 0.2 * cond_av / cond_av.max()
    mask = live[:, :, None, None, None]
    cond_av, cond_repr, cross_corr = (
        np.where(mask, a, np.nan) for a in (cond_av, cond_repr, cross_corr)
    )
    nevents = np.where(live, rng.integers(430, 940, size=(NY, NX)), 0).astype("int64")
    dims = ("ref_y", "ref_x", "y", "x", "time")
    return xr.Dataset(
        {
            "cond_av": (dims, cond_av),
            "cond_repr": (dims, cond_repr),
            "cross_corr": (dims, cross_corr),
            "nevents": (("ref_y", "ref_x"), nevents),
            "computed": (("ref_y", "ref_x"), ~geo.dead),
            "dead": (("y", "x"), geo.dead),
        },
        coords={
            "R": (("y", "x"), geo.R.astype("float32")),
            "Z": (("y", "x"), geo.Z.astype("float32")),
            "time": t,
        },
        attrs={"averages": AVERAGES_STAMP, "dead_mask_source": options.mask_source},
    )


def _track(geo, bank, key, options, rng):
    """One track's ``vr``, ``vz``, ``nlags``, ``level``, positions and fitted lags, over every pixel."""
    t = np.asarray(bank["time"].values)
    nt = t.size
    factor = {"max": 0.95, "com": 1.0, "2dcc": 0.88}[key]
    vr = np.full((NY, NX), np.nan)
    vz = np.full((NY, NX), np.nan)
    nlags = np.full((NY, NX), np.nan)
    level = np.full((NY, NX), np.nan)
    pos_r = np.full((NY, NX, nt), np.nan)
    pos_z = np.full((NY, NX, nt), np.nan)
    fit = np.zeros((NY, NX, nt), dtype=bool)
    noise = {"max": 1.5e-4, "com": 0.8e-4, "2dcc": 1.5e-4}[key]
    nevents = np.asarray(bank["nevents"].values)
    for y in range(NY):
        for x in range(NX):
            if geo.dead[y, x]:
                continue
            nlags[y, x] = 1.0 if (x, y) in options.failed else 0.0
            if nevents[y, x] == 0:
                nlags[y, x] = 0.0
                continue
            half = options.fit_half_width_us.get((x, y), 4.5) * 1e-6
            valid = np.abs(t) <= 12e-6
            pos_r[y, x] = np.where(
                valid,
                geo.R[y, x] + factor * geo.vr[y, x] * t + _noise(rng, nt, noise),
                np.nan,
            )
            pos_z[y, x] = np.where(
                valid,
                geo.Z[y, x] + factor * geo.vz[y, x] * t + _noise(rng, nt, noise),
                np.nan,
            )
            if (x, y) in options.failed:
                fit[y, x, nt // 2] = True
                continue
            window = valid & (np.abs(t) <= half + 1e-12)
            fit[y, x] = window
            nlags[y, x] = float(window.sum())
            # The stored velocity is what the "lsq" estimator gives: the slope of the fitted lags.
            vr[y, x] = np.polyfit(t[window], pos_r[y, x, window], 1)[0]
            vz[y, x] = np.polyfit(t[window], pos_z[y, x, window], 1)[0]
            if key == "com":
                level[y, x] = 0.45 + 0.1 * rng.random()
    return vr, vz, nlags, level, pos_r, pos_z, fit


def build_fields(geo, bank, options=None):
    """The ``method_fields`` dataset, over the lags of ``bank``."""
    options = options or Options()
    rng = np.random.default_rng(options.seed + 1)
    nevents = np.asarray(bank["nevents"].values).astype(float)
    data = {}
    for key in ("max", "com", "2dcc"):
        vr, vz, nlags, level, pos_r, pos_z, fit = _track(geo, bank, key, options, rng)
        data[f"vr_{key}"] = (("y", "x"), vr)
        data[f"vz_{key}"] = (("y", "x"), vz)
        data[f"nlags_{key}"] = (("y", "x"), nlags)
        data[f"level_{key}"] = (("y", "x"), level)
        data[f"pos_r_{key}"] = (("y", "x", "time"), pos_r)
        data[f"pos_z_{key}"] = (("y", "x", "time"), pos_z)
        data[f"fit_{key}"] = (("y", "x", "time"), fit)
    live = ~geo.dead
    nevents = np.where(live, nevents, np.nan)
    data["nevents"] = (("y", "x"), nevents)

    def tde(scale_r, scale_z, nan_columns=0, nan_fraction=0.0):
        vr = geo.vr * scale_r * (1 + 0.1 * rng.standard_normal((NY, NX)))
        vz = geo.vz * scale_z * (1 + 0.1 * rng.standard_normal((NY, NX)))
        bad = (np.arange(NX)[None, :] < nan_columns) | (
            rng.random((NY, NX)) < nan_fraction
        )
        vr, vz = np.where(live & ~bad, vr, np.nan), np.where(live & ~bad, vz, np.nan)
        return vr, vz, bad

    vr3, vz3, bad = tde(1.8, 1.4, nan_columns=2, nan_fraction=0.03)
    vr2, vz2, _ = tde(2.4, 2.9)
    vr2, vz2 = np.where(bad, np.nan, vr2), np.where(bad, np.nan, vz2)
    data["vr3_tde"], data["vz3_tde"] = (("y", "x"), vr3), (("y", "x"), vz3)
    data["vr2_tde"], data["vz2_tde"] = (("y", "x"), vr2), (("y", "x"), vz2)
    data["cc_tde"] = (
        ("y", "x"),
        np.where(live & ~bad, 0.7 + 0.15 * rng.random((NY, NX)), np.nan),
    )
    # The TDE on the conditional average needs the average, so a reference without events has none.
    vr3c, vz3c, bad_c = tde(1.15, 1.1, nan_fraction=0.04)
    bad_c = bad_c | (np.asarray(bank["nevents"].values) == 0)
    vr3c, vz3c = np.where(bad_c, np.nan, vr3c), np.where(bad_c, np.nan, vz3c)
    vr2c, vz2c, _ = tde(1.3, 1.5)
    data["vr3_catde"], data["vz3_catde"] = (("y", "x"), vr3c), (("y", "x"), vz3c)
    data["vr2_catde"], data["vz2_catde"] = (
        ("y", "x"),
        np.where(bad_c, np.nan, vr2c),
    ), (
        ("y", "x"),
        np.where(bad_c, np.nan, vz2c),
    )
    data["dead"] = (("y", "x"), geo.dead)
    return xr.Dataset(
        data,
        coords={
            "R": (("y", "x"), geo.R.astype("float32")),
            "Z": (("y", "x"), geo.Z.astype("float32")),
            "time": np.asarray(bank["time"].values),
        },
        attrs={
            "min_cc": 0.5,
            "settings": options.settings,
            "dead_mask_source": options.mask_source,
        },
    )


def build_blobs(geo, bank, options=None):
    """The ``blob_parameters`` dataset: 13 numbers at every live pixel, in metres, m^2, seconds and radians."""
    options = options or Options()
    rng = np.random.default_rng(options.seed + 2)
    live = ~geo.dead
    nevents = np.where(live, np.asarray(bank["nevents"].values).astype(float), np.nan)
    values = {
        "nevents": nevents,
        "level": 0.45 + 0.1 * rng.random((NY, NX)),
        "area": 4e-5 * (1 + 0.2 * rng.random((NY, NX))),
        "lx_c": 0.005 * (1 + 0.2 * rng.random((NY, NX))),
        "ly_c": 0.007 * (1 + 0.2 * rng.random((NY, NX))),
        "theta_c": rng.random((NY, NX)) * np.pi,
        "lr": 0.0065 * (1 + 0.2 * rng.random((NY, NX))),
        "lz": 0.0075 * (1 + 0.2 * rng.random((NY, NX))),
        "lx_f": 0.0045 * (1 + 0.2 * rng.random((NY, NX))),
        "ly_f": 0.0060 * (1 + 0.2 * rng.random((NY, NX))),
        "theta_f": rng.random((NY, NX)) * np.pi,
        "taud": 1.0e-5 * (1 + 0.3 * rng.random((NY, NX))),
        "lam": 0.3 * (1 + rng.random((NY, NX))),
    }
    data = {
        name: (("y", "x"), np.where(live, values[name], np.nan)) for name in BLOB_NAMES
    }
    return xr.Dataset(
        data,
        coords={
            "R": (("y", "x"), geo.R.astype("float32")),
            "Z": (("y", "x"), geo.Z.astype("float32")),
        },
        attrs={"units": "m, m^2, s, rad", "averages": AVERAGES_STAMP},
    )


@dataclass(frozen=True)
class Products:
    """The three datasets of one world, built directly, without the store."""

    geometry: Geometry
    bank: xr.Dataset
    fields: xr.Dataset
    blobs: xr.Dataset


def make_products(window=60, **options):
    """``Products`` straight from the builders; ``options`` are :class:`Options` fields."""
    geo = make_geometry()
    opts = Options(**options)
    bank = build_bank(geo, window, opts)
    return Products(
        geo, bank, build_fields(geo, bank, opts), build_blobs(geo, bank, opts)
    )


# ---------------------------------------------------------------------------
# Stand-in specs
# ---------------------------------------------------------------------------


def _render(result, params, target):
    return None


class World:
    """Stand-in specs under the three product keys, plus the synthetic data they serve.

    ``seed`` runs them through the store the way a batch job does. The parameters decide the data
    where the plan says they do: ``averages.window`` sets the number of lags of the bank.
    """

    def __init__(self, **options):
        self.options = Options(**options)
        self.geometry = make_geometry()
        self.calls = []
        self._saved = {}

    # -- computes ----------------------------------------------------------

    def _bank(self, ds, params):
        self.calls.append("pixel_averages")
        return build_bank(self.geometry, params.averages.window, self.options)

    def _fields(self, ds, params, upstream):
        self.calls.append("method_fields")
        return build_fields(self.geometry, upstream, self.options)

    def _blobs(self, ds, params, upstream):
        self.calls.append("blob_parameters")
        return build_blobs(self.geometry, upstream, self.options)

    def specs(self):
        bank = registry.PlotSpec(
            key="pixel_averages",
            label="Conditional average at every pixel",
            diagnostics=("apd",),
            params=PixelAveragesParams,
            render=_render,
            compute=self._bank,
            batch_only=True,
        )
        fields = registry.PlotSpec(
            key="method_fields",
            label="Velocity fields of every method",
            diagnostics=("apd",),
            params=MethodFieldsParams,
            render=_render,
            compute=self._fields,
            requires="pixel_averages",
            upstream_params=lambda p: PixelAveragesParams(averages=p.averages),
        )
        blobs = registry.PlotSpec(
            key="blob_parameters",
            label="Blob parameters at every pixel",
            diagnostics=("apd",),
            params=BlobParametersParams,
            render=_render,
            compute=self._blobs,
            requires="pixel_averages",
            upstream_params=lambda p: PixelAveragesParams(averages=p.averages),
        )
        return {
            "pixel_averages": bank,
            "method_fields": fields,
            "blob_parameters": blobs,
        }

    # -- registry ----------------------------------------------------------

    def install(self):
        """Register the stand-ins, remembering whatever was registered under those keys."""
        for key, spec in self.specs().items():
            self._saved[key] = registry.REGISTRY.get(key)
            registry.REGISTRY[key] = spec
        return self

    def uninstall(self):
        for key, spec in self._saved.items():
            if spec is None:
                registry.REGISTRY.pop(key, None)
            else:
                registry.REGISTRY[key] = spec
        self._saved = {}

    # -- seeding -----------------------------------------------------------

    def target(self, shot, machine="cmod", path="unused"):
        return registry.Target(
            machine, shot, "apd", True, path, float("nan"), float("nan"), "none"
        )

    def seed(self, conn, shot, *keys, params=None, machine="cmod"):
        """What ``fusion-ui precompute <keys> --shot N`` leaves behind; ``{key: run row}``.

        ``params`` is ``{key: params}`` for any key that is not at its defaults. A chained key
        computes its bank first, as the store does, unless the bank is already cached.
        """
        params = params or {}
        target = self.target(shot, machine)
        runs = {}
        for key in keys or KEYS:
            spec = registry.get(key)
            _, run = store.result(
                conn, spec, target, params.get(key, spec.params()), None, batch=True
            )
            runs[key] = run
        self.calls.clear()
        return runs


def default_params(key):
    return registry.get(key).params()


def with_window(window):
    """``method_fields`` parameters whose 2DCA window is ``window`` samples: a shot with faster dynamics."""
    spec = registry.REGISTRY.get("method_fields")
    params = spec.params() if spec is not None else MethodFieldsParams()
    return dataclasses.replace(
        params, averages=dataclasses.replace(params.averages, window=window)
    )


@pytest.fixture
def fields_world():
    """The stand-in product specs, registered for one test."""
    world = World().install()
    yield world
    world.uninstall()
