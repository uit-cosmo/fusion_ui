"""The dead-pixel view: the mask's evidence, every pixel's PDF and spectrum, raw files only."""

import numpy as np
import plotly.graph_objects as go
import pytest
import xarray as xr
from scipy import signal

import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.core import registry
from fusion_ui.plots import dead_pixels

STEP = 1e-3  # digitizer step the synthetic record is quantised to, as the raw files are
DEAD = (0, 3)
EMPTY = (2, 0)


@pytest.fixture(scope="module")
def record():
    """3 x 4 pixels sharing one red signal, but for one white-noise pixel and one with no samples."""
    rng = np.random.default_rng(1)
    n, dt = 60_000, 5e-7
    phi = np.exp(-dt / 20e-6)
    shared = signal.lfilter([1], [1, -phi], rng.normal(size=n) * np.sqrt(1 - phi**2))
    frames = 0.9 * shared + 0.02 * rng.normal(size=(3, 4, n))
    frames[DEAD] = 0.01 * rng.normal(size=n)
    frames = np.round(frames / STEP) * STEP
    frames[EMPTY] = np.nan
    return xr.Dataset(
        {"frames": (("y", "x", "time"), frames.astype("float32"))},
        coords={
            "time": np.arange(n) * dt,
            "R": (("y", "x"), np.tile(88.0 + 0.4 * np.arange(4), (3, 1))),
            "Z": (("y", "x"), np.tile(0.4 * np.arange(3), (4, 1)).T),
        },
    )


@pytest.fixture(scope="module")
def result(record):
    spec = registry.get("dead_pixels")
    return spec.compute(record, spec.params())


def test_the_planted_dead_pixels_are_found(result):
    expected = np.zeros((3, 4), bool)
    expected[DEAD] = expected[EMPTY] = True
    np.testing.assert_array_equal(result.dead.values, expected)


def test_a_pdf_has_one_bin_per_digitizer_level(result):
    """A dead pixel spans a few tens of levels: binned finer it would draw as a comb."""
    centres = result.pdf_volts.values[DEAD]
    steps = np.diff(centres[np.isfinite(centres)])
    np.testing.assert_allclose(steps, STEP, rtol=1e-3)


@pytest.mark.parametrize("view", dead_pixels.VIEWS)
def test_every_view_draws_one_panel_per_pixel(result, view):
    fig = dead_pixels.figure(result, view)
    assert isinstance(fig, go.Figure)
    panels = [t for t in fig.data if t.showlegend is False]
    assert len(panels) == 12
    assert len(fig.layout.shapes) == (2 * 12 if view == "Spectrum" else 0)


def test_scalars_mark_every_pixel(result):
    out = registry.get("dead_pixels").scalars(result)
    assert out["number_dead"] == 2
    assert out[(DEAD[1], DEAD[0], "dead")] == 1.0
    assert out[(1, 1, "dead")] == 0.0
    assert sum(1 for k in out if isinstance(k, tuple) and k[2] == "psd_ratio") == 12


def test_the_summary_names_the_dead_pixels(result):
    text = dead_pixels.summary(result)
    assert text.startswith("2 of 12 pixels dead")
    assert "(0, 3)" in text


def test_it_is_offered_on_raw_files_only():
    spec = registry.get("dead_pixels")
    assert spec.preprocessed is False
    assert spec in registry.for_diagnostic("apd", preprocessed=False)
    assert spec not in registry.for_diagnostic("apd", preprocessed=True)
