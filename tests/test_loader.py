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
