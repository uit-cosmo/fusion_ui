"""The ragged ASP/FSP adapter -- every (quantity, position) on its own axis."""

import pytest

from fusion_ui.core import loader, probes


def test_quantities_and_positions_reads_only_var_names(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    assert probes.quantities_and_positions(ds) == {"Vf": [0, 1], "ne": [0, 1]}


def test_rho_variables_are_not_mistaken_for_quantities(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    found = probes.quantities_and_positions(ds)
    assert "rho" not in found and "rho_ne" not in found


def test_load_trace_returns_matching_time_and_value_arrays(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    trace = probes.load_trace(ds, "ne", 0)
    assert trace.time.shape == trace.value.shape
    assert trace.rho.shape == trace.rho_time.shape


def test_load_trace_without_rho(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    trace = probes.load_trace(ds, "ne", 0, with_rho=False)
    assert trace.rho is None and trace.rho_time is None


def test_positions_are_genuinely_ragged(asp_dataset_path):
    """The whole point of this adapter: no shared time axis across positions."""
    ds = loader.open_dataset(str(asp_dataset_path))
    trace0 = probes.load_trace(ds, "ne", 0)
    trace1 = probes.load_trace(ds, "ne", 1)
    assert trace0.time.shape != trace1.time.shape


def test_load_trace_raises_for_an_unknown_combination(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    with pytest.raises(KeyError):
        probes.load_trace(ds, "ne", 99)


def test_probe_geometry_reads_global_attrs(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path))
    geometry = probes.probe_geometry(ds)
    assert geometry["probe_type"] == "Mirror Langmuir Probe (MLP)"
    assert len(geometry["probe_origin"]) == 3


def test_probe_geometry_survives_a_missing_origin(asp_dataset_path):
    ds = loader.open_dataset(str(asp_dataset_path)).copy()
    del ds.attrs["probe_origin"]
    geometry = probes.probe_geometry(ds)
    assert geometry["probe_origin"] is None
