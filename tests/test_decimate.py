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


def test_selected_x_range_needs_two_distinct_points():
    assert decimate.selected_x_range(None) is None
    assert decimate.selected_x_range({"selection": {"points": []}}) is None
    one = {"selection": {"points": [{"x": 1.0, "y": 0.0}]}}
    assert decimate.selected_x_range(one) is None
    same = {"selection": {"points": [{"x": 1.0}, {"x": 1.0}]}}
    assert decimate.selected_x_range(same) is None


def test_selected_x_range_spans_the_box_points():
    event = {"selection": {"points": [{"x": 2.0}, {"x": 0.5}, {"x": 1.0}]}}
    assert decimate.selected_x_range(event) == (0.5, 2.0)


def test_selected_x_range_skips_unmappable_points():
    event = {"selection": {"points": [{"z": 5.0}, {"x": "nope"}, {"x": 1.0}]}}
    assert decimate.selected_x_range(event) is None
    event = {
        "selection": {"points": [{"z": 5.0}, {"x": 1.0}, {"x": 3.0}]}
    }
    assert decimate.selected_x_range(event) == (1.0, 3.0)


def test_slice_to_window_restricts_to_the_range():
    x = np.arange(10).astype(float)
    y = (np.arange(10) ** 2).astype(float)
    x_out, y_out = decimate.slice_to_window(x, y, 3.0, 5.0)
    assert list(x_out) == [3.0, 4.0, 5.0]
    assert list(y_out) == [9.0, 16.0, 25.0]
    # Reversed bounds still work; an empty window stays empty.
    x_out, _ = decimate.slice_to_window(x, y, 5.0, 3.0)
    assert list(x_out) == [3.0, 4.0, 5.0]
    x_out, y_out = decimate.slice_to_window(x, y, 20.0, 30.0)
    assert x_out.size == 0 and y_out.size == 0
