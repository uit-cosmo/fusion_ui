"""Every cached single-shot spec, through netCDF and back.

``store.compute_and_store`` returns the in-memory result on a cache miss, not
the blob it just wrote. So a spec that stores something netCDF cannot hold (a
bool, an object array) renders on the first click and breaks on every one
after it, and a test that calls ``compute`` directly does not see it.
Parametrised over the registry, so a new spec gets the check too.
"""

import dataclasses

import numpy as np
import pytest
import xarray as xr

import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.core import loader, registry, store

CENTRE = 4  # the reference pixel on the 9x9 blob fixture

#: Its render draws into Streamlit, so only the round trip of its result is
#: checked here.
DRAWS_INTO_STREAMLIT = {"dead_pixels"}

#: The three products want a preprocessed record with its dead-pixel mask
#: stored, and their round trip is ``tests/test_products.py``'s.
PRODUCTS = {"pixel_averages", "method_fields", "blob_parameters"}


def cached_imaging_specs():
    return [
        spec
        for key, spec in sorted(registry.REGISTRY.items())
        if spec.cached and key not in PRODUCTS and "apd" in spec.diagnostics
    ]


def at_centre(params):
    """Point the reference pixel, where the spec has one, at the blob."""
    for axis in ("refx", "refy"):
        if hasattr(params, axis):
            setattr(params, axis, CENTRE)
    return params


@pytest.fixture
def blobs(blob_dataset_path):
    with xr.open_dataset(blob_dataset_path) as ds:
        yield ds.load()


@pytest.fixture
def target():
    return registry.Target(
        machine="cmod",
        shot=4321,
        diagnostic="apd",
        preprocessed=True,
        path="unused",
        t_start=0.0,
        t_end=1.6e-3,
    )


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec", cached_imaging_specs(), ids=lambda s: s.key
)
def test_a_stored_result_survives_the_cache_and_still_renders(
    spec, conn, cache, blobs, target
):
    """Compute it, then ask the store for it again: the second call reads the
    blob back off disk, as the page does on every visit after the first."""
    params = at_centre(spec.params())
    if not spec.accepts(target.preprocessed):
        target = dataclasses.replace(target, preprocessed=not target.preprocessed)

    # A whole-record spec is handed the record with its discharge window, as
    # ``loader.input_for`` does. The fixture is the record, all of it.
    record = (
        loader.whole_record(blobs, target.t_start, target.t_end)
        if spec.whole_record
        else blobs
    )

    first, run = store.result(conn, spec, target, params, record)
    assert run["status"] == "ok", f"{spec.key}: {run['error']}"

    second, reloaded_run = store.result(conn, spec, target, params, record)
    assert reloaded_run["id"] == run["id"], "expected a cache hit, not a recompute"
    assert second is not first, "the second call must come off disk"

    # Same numbers, not merely a dataset of the same shape.
    for name, values in first.data_vars.items():
        assert name in second, f"{spec.key}: {name} did not survive the round trip"
        np.testing.assert_allclose(
            np.asarray(second[name].values, dtype=float),
            np.asarray(values.values, dtype=float),
            equal_nan=True,
            err_msg=f"{spec.key}: {name} changed through netCDF",
        )
        assert second[name].dtype == values.dtype, (
            f"{spec.key}: {name} came back as {second[name].dtype}, "
            f"stored as {values.dtype}"
        )

    if spec.scalars is not None:
        assert spec.scalars(second) == pytest.approx(spec.scalars(first), nan_ok=True)

    if spec.key not in DRAWS_INTO_STREAMLIT:
        assert spec.render(second, params, target) is not None
