"""Min/max envelope downsampling -- the point is that spikes survive."""

import numpy as np

from fusion_ui.core import decimate


def test_short_traces_pass_through_unchanged():
    x = np.arange(10)
    y = np.arange(10).astype(float)
    x_out, y_out = decimate.envelope(x, y, max_points=4000)
    assert list(x_out) == list(x)
    assert list(y_out) == list(y)


def test_reduces_to_at_most_max_points():
    n = 100_000
    x = np.arange(n)
    y = np.sin(x / 100.0)
    x_out, y_out = decimate.envelope(x, y, max_points=1000)
    assert len(x_out) <= 1000
    assert len(x_out) == len(y_out)


def test_preserves_a_single_sample_spike():
    """Naive striding (``y[::n]``) would very likely step over this."""
    n = 10_000
    y = np.zeros(n)
    y[5000] = 100.0
    x = np.arange(n)
    _, y_out = decimate.envelope(x, y, max_points=200)
    assert y_out.max() == 100.0


def test_output_is_sorted_by_x():
    x = np.arange(5000)
    y = np.random.default_rng(0).normal(size=5000)
    x_out, _ = decimate.envelope(x, y, max_points=500)
    assert list(x_out) == sorted(x_out)


def test_ignores_all_nan_buckets():
    n = 4000
    y = np.full(n, np.nan)
    y[:100] = 1.0
    x = np.arange(n)
    _, y_out = decimate.envelope(x, y, max_points=100)
    assert not np.any(np.isnan(y_out))


def test_a_flat_trace_survives():
    x = np.arange(20_000)
    y = np.ones(20_000)
    _, y_out = decimate.envelope(x, y, max_points=100)
    assert np.all(y_out == 1.0)
