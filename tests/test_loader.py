"""Time-window slicing and lazy frame/pixel access over the APD/phantom shape.

Uses the tiny real ``apd_dataset_path`` fixture rather than the byte-stuffed
files ``test_catalog.py`` uses -- this module actually opens the file.
"""

import math
from types import SimpleNamespace

import pytest

from fusion_ui.core import loader


def test_dataset_path_follows_the_diagnostic_convention(monkeypatch, apd_dataset_path):
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(apd_dataset_path.parent.parent))
    assert loader.dataset_path("cmod", 1234, "apd", False) == str(apd_dataset_path)


def test_time_window_uses_discharge_metadata_when_present(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    discharge = SimpleNamespace(t_start=1.005, t_end=1.01)
    assert loader.time_window(ds, discharge) == (1.005, 1.01, "metadata")


@pytest.mark.parametrize(
    "discharge", [None, SimpleNamespace(t_start=float("nan"), t_end=1.0)]
)
def test_time_window_falls_back_to_a_centred_default(apd_dataset_path, discharge):
    ds = loader.open_dataset(str(apd_dataset_path))
    t_start, t_end, source = loader.time_window(ds, discharge, default_span=0.01)
    t_min, t_max = float(ds.time.min()), float(ds.time.max())
    center = (t_min + t_max) / 2
    assert source == "default"
    assert math.isclose(t_start, center - 0.005)
    assert math.isclose(t_end, center + 0.005)


def test_sliced_restricts_the_time_dimension(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    windowed = loader.sliced(ds, 1.0, 1.005)
    assert 0 < windowed.sizes["time"] < ds.sizes["time"]
    assert float(windowed.time.max()) <= 1.005


def test_frame_drops_time_and_keeps_y_x_order(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    frame = loader.frame(ds, 0)
    assert frame.dims == ("y", "x")
    assert frame.shape == (ds.sizes["y"], ds.sizes["x"])


def test_pixel_series_returns_matching_time_and_value_arrays(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    time, values = loader.pixel_series(ds, 0, 0)
    assert time.shape == values.shape == (ds.sizes["time"],)


def test_pixel_grid_reads_the_r_z_coordinates(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    r, z = loader.pixel_grid(ds)
    assert r.shape == z.shape == (ds.sizes["y"], ds.sizes["x"])


def test_pixel_grid_is_none_without_r_z_coordinates(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path)).drop_vars(["R", "Z"])
    assert loader.pixel_grid(ds) == (None, None)


def test_cached_frame_times_matches_an_uncached_read(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    direct = loader.frame_times(loader.sliced(ds, 1.0, 1.01))
    cached = loader.cached_frame_times(str(apd_dataset_path), 1.0, 1.01)
    assert list(direct) == list(cached)


def test_nearest_index_finds_the_closest_sample():
    times = [0.0, 0.1, 0.2, 0.3]
    assert loader.nearest_index(times, 0.24) == 2
    assert loader.nearest_index(times, -1.0) == 0
    assert loader.nearest_index(times, 10.0) == 3


# ---------------------------------------------------------------------------
# What an analysis computes on: the record cut to the discharge window, or, for
# a spec that declares it needs the whole record, all of it with the window.
# ---------------------------------------------------------------------------


def test_input_for_cuts_the_record_to_the_window_by_default(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    cut = loader.input_for(ds, 1.005, 1.01)
    assert 0 < cut.sizes["time"] < ds.sizes["time"]
    assert float(cut.time.min()) >= 1.005 and float(cut.time.max()) <= 1.01
    assert loader.WINDOW_ATTR not in cut.attrs  # nothing to tell it: it is the window


def test_input_for_hands_over_the_whole_record_and_the_window(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    whole = loader.input_for(ds, 1.005, 1.01, whole=True)
    assert whole.sizes["time"] == ds.sizes["time"]
    assert loader.discharge_window(whole) == (1.005, 1.01)
    # Lazy still: the spec that wants the record loads it, never the shared dataset.
    assert not whole["frames"].variable._in_memory


def test_the_whole_record_is_a_copy_the_shared_dataset_is_not_touched(apd_dataset_path):
    """``open_dataset`` is shared by every session: the window must not leak in."""
    ds = loader.open_dataset(str(apd_dataset_path))
    loader.input_for(ds, 1.005, 1.01, whole=True)
    assert loader.WINDOW_ATTR not in ds.attrs
    again = loader.input_for(ds, 1.0, 1.02, whole=True)
    assert loader.discharge_window(again) == (1.0, 1.02)
    first = loader.input_for(ds, 1.005, 1.01, whole=True)
    assert loader.discharge_window(first) == (1.005, 1.01)


def test_a_record_with_no_time_axis_comes_back_as_it_is(asp_dataset_path):
    """A probe file: each quantity has its own time axis, nothing to cut or keep."""
    import xarray as xr

    with xr.open_dataset(asp_dataset_path) as probe:
        assert loader.TIME_DIM not in probe.dims
        assert loader.input_for(probe, math.nan, math.nan) is probe
        assert loader.input_for(probe, math.nan, math.nan, whole=True) is probe


def test_a_record_with_no_discharge_window_is_refused_not_guessed(apd_dataset_path):
    """The record's own span would pass for the discharge window, and the window
    found from it would be wrong without a word."""
    ds = loader.open_dataset(str(apd_dataset_path))
    with pytest.raises(ValueError, match="carries no discharge window"):
        loader.discharge_window(ds)
    with pytest.raises(ValueError, match="carries no discharge window"):
        loader.discharge_window(ds.assign_attrs({loader.WINDOW_ATTR: "not a window"}))


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1.0734, 1.4], (1.0734, 1.4)),
        ("[0.95, 1.4]", (0.95, 1.4)),
        ((1, 2), (1.0, 2.0)),
        ("1.0734 to 1.4", None),
        ([1.0, 2.0, 3.0], None),
        ([math.nan, 1.4], None),
        (None, None),
    ],
)
def test_a_window_attribute_is_a_start_and_an_end_or_nothing(value, expected):
    assert loader.as_window(value) == expected


def test_a_cropped_file_says_its_window_by_its_analysis_window(apd_dataset_path):
    ds = loader.open_dataset(str(apd_dataset_path))
    # A raw file is cut to the discharge window, and says nothing of it.
    assert loader.stored_window(ds) is None
    as_list = ds.assign_attrs(analysis_window=[1.004, 1.012])
    assert loader.stored_window(as_list) == (1.004, 1.012)
    as_json = ds.assign_attrs(analysis_window="[1.004, 1.012]")
    assert loader.stored_window(as_json) == (1.004, 1.012)
    assert loader.stored_window(ds.assign_attrs(analysis_window="the puff")) is None
    # The discharge window is not what the file was cropped to.
    assert loader.stored_window(ds.assign_attrs(discharge_window=[0.9, 1.4])) is None
