"""Every cached spec, through netCDF and back.

The gap this closes: ``store.compute_and_store`` returns the *in-memory*
result on a cache miss, never the blob it just wrote. So a spec that stores
something netCDF cannot represent -- a bool, an object array, a dimension the
reload cannot reconstruct -- renders perfectly on the first click and breaks on
every one after it, and no per-spec test catches that, because they all call
``compute`` directly.

This is deliberately parametrised over the registry rather than written once
per spec: the next spec added gets the check for free, which is the only way a
guard like this stays true.
"""

import numpy as np
import pytest
import xarray as xr

import fusion_ui.plots  # noqa: F401 - registers every spec
from fusion_ui.core import registry, store

CENTRE = 4  # the reference pixel on the 9x9 blob fixture

#: Specs whose render needs a Streamlit runtime (a slider, a click target) and
#: so cannot be called from a plain test. The round trip of their *result* is
#: still checked; only the drawing is skipped.
DRAWS_INTO_STREAMLIT = {"two_dca", "velocity_field"}

#: Live specs have no blob to round trip, and the probe specs want an ASP file
#: rather than the imaging fixture.
SKIP = {"raw_frames", "probe_trace"}


def cached_imaging_specs():
    return [
        spec
        for key, spec in sorted(registry.REGISTRY.items())
        if spec.cached and key not in SKIP and "apd" in spec.diagnostics
    ]


def at_centre(params):
    """Point whatever reference-pixel fields this spec has at the blob."""
    for holder in (params, getattr(params, "two_dca", None)):
        if holder is None:
            continue
        for axis, value in (("refx", CENTRE), ("refy", CENTRE)):
            if hasattr(holder, axis):
                setattr(holder, axis, value)
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
    """Compute it, then ask the store for it *again*.

    The second call is the one that matters: it finds the ledger row, reads the
    blob back off disk, and hands ``render`` a dataset that has been through
    netCDF encoding. That is what the page does on every visit after the first.
    """
    params = at_centre(spec.params())

    first, run = store.result(conn, spec, target, params, blobs)
    assert run["status"] == "ok", f"{spec.key}: {run['error']}"

    second, reloaded_run = store.result(conn, spec, target, params, blobs)
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
