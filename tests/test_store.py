"""The result store, driven by a synthetic spec.

No real analysis is involved: the point is the ledger, the blob and the scalar
rows, and a fake compute makes "was this recomputed?" directly observable.
"""

import dataclasses
import os

import numpy as np
import pytest
import xarray as xr

from fusion_ui.core import registry, store


@dataclasses.dataclass
class Params:
    gain: float = 1.0


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def target():
    return registry.Target(
        machine="cmod",
        shot=1160616027,
        diagnostic="apd",
        preprocessed=True,
        path="/nowhere/apd.nc",
        t_start=1.15,
        t_end=1.45,
    )


@pytest.fixture
def calls():
    return []


@pytest.fixture
def spec(calls):
    def compute(ds, params):
        calls.append(params.gain)
        return xr.Dataset({"y": ("t", np.arange(4.0) * params.gain)})

    return registry.PlotSpec(
        key="synthetic",
        label="Synthetic",
        diagnostics=("apd",),
        params=Params,
        render=lambda result, params, target: None,
        compute=compute,
        scalars=lambda result: {
            "total": float(result["y"].sum()),
            (3, 4, "corner"): float(result["y"][-1]),
        },
    )


def test_a_first_call_computes_writes_the_blob_and_the_rows(
    conn, cache, spec, target, calls
):
    result, run = store.result(conn, spec, target, Params(), ds=None)

    assert calls == [1.0]
    assert run["status"] == "ok"
    assert os.path.exists(run["blob_path"])
    assert str(cache) in run["blob_path"]
    assert run["seconds"] is not None
    assert result["y"].values.tolist() == [0.0, 1.0, 2.0, 3.0]

    rows = {
        (r["x"], r["y"], r["name"]): r["value"]
        for r in conn.execute("SELECT * FROM scalars")
    }
    assert rows == {(-1, -1, "total"): 6.0, (3, 4, "corner"): 3.0}
    assert conn.execute("SELECT COUNT(*) FROM param_sets").fetchone()[0] == 1


def test_a_second_call_loads_the_blob_instead_of_recomputing(
    conn, cache, spec, target, calls
):
    store.result(conn, spec, target, Params(), ds=None)
    result, run = store.result(conn, spec, target, Params(), ds=None)

    assert calls == [1.0], "compute ran twice for one parameter set"
    assert result["y"].values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert run["status"] == "ok"


def test_different_parameters_get_their_own_run_and_their_own_blob(
    conn, cache, spec, target, calls
):
    _, first = store.result(conn, spec, target, Params(gain=1.0), ds=None)
    _, second = store.result(conn, spec, target, Params(gain=2.0), ds=None)

    assert calls == [1.0, 2.0]
    assert first["params_hash"] != second["params_hash"]
    assert first["blob_path"] != second["blob_path"]
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2

    # And going back to the first parameter set reads from disk.
    store.result(conn, spec, target, Params(gain=1.0), ds=None)
    assert calls == [1.0, 2.0]


def test_the_preprocessed_variant_does_not_overwrite_the_raw_one(
    conn, cache, spec, target
):
    _, one = store.result(conn, spec, target, Params(), ds=None)
    _, two = store.result(
        conn, spec, dataclasses.replace(target, preprocessed=False), Params(), ds=None
    )
    assert one["blob_path"] != two["blob_path"]


def test_a_failure_is_recorded_and_not_re_raised_on_reload(conn, cache, target):
    def boom(ds, params):
        raise ValueError("no events found")

    failing = registry.PlotSpec(
        key="failing",
        label="Failing",
        diagnostics=("apd",),
        params=Params,
        render=lambda result, params, target: None,
        compute=boom,
    )

    result, run = store.result(conn, failing, target, Params(), ds=None)
    assert result is None
    assert run["status"] == "failed"
    assert "no events found" in run["error"]
    assert run["blob_path"] is None

    # The page must be able to render the error rather than crash on it.
    again, run = store.result(conn, failing, target, Params(), ds=None)
    assert again is None and run["status"] == "failed"


def test_a_recompute_after_a_failure_replaces_the_row(conn, cache, target, spec):
    def boom(ds, params):
        raise ValueError("transient")

    failing = dataclasses.replace(spec, compute=boom)
    _, run = store.result(conn, failing, target, Params(), ds=None)
    store.delete_run(conn, run)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0

    _, run = store.result(conn, spec, target, Params(), ds=None)
    assert run["status"] == "ok"


def test_deleting_a_run_removes_its_blob_and_its_scalars(conn, cache, spec, target):
    _, run = store.result(conn, spec, target, Params(), ds=None)
    path = run["blob_path"]

    store.delete_run(conn, run)
    assert not os.path.exists(path)
    assert conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0] == 0
    # The parameter set survives: it is what the hash means, not a result.
    assert conn.execute("SELECT COUNT(*) FROM param_sets").fetchone()[0] == 1


def test_a_missing_blob_is_recomputed_rather_than_reported_as_nothing(
    conn, cache, spec, target, calls
):
    """Someone clearing CACHE_DIR by hand must not brick the run."""
    _, run = store.result(conn, spec, target, Params(), ds=None)
    os.remove(run["blob_path"])

    result, run = store.result(conn, spec, target, Params(), ds=None)
    assert calls == [1.0, 1.0]
    assert result is not None and run["status"] == "ok"


def test_a_live_spec_gets_its_input_back_and_leaves_no_ledger_row(conn, cache, target):
    live = registry.PlotSpec(
        key="live",
        label="Live",
        diagnostics=("apd",),
        params=Params,
        render=lambda result, params, target: None,
    )
    ds = xr.Dataset({"y": ("t", [1.0])})
    result, run = store.result(conn, live, target, Params(), ds=ds)

    assert result is ds
    assert run is None
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_a_nan_scalar_is_stored_as_null(conn, cache, target, spec):
    nan_spec = dataclasses.replace(
        spec, key="nanny", scalars=lambda result: {"taud": float("nan")}
    )
    _, run = store.result(conn, nan_spec, target, Params(), ds=None)
    value = conn.execute("SELECT value FROM scalars WHERE name = 'taud'").fetchone()[0]
    assert value is None


def test_the_blob_carries_its_own_provenance(conn, cache, spec, target):
    _, run = store.result(conn, spec, target, Params(gain=3.0), ds=None)
    with xr.open_dataset(run["blob_path"]) as stored:
        assert stored.attrs["fusion_ui_plot"] == "synthetic"
        assert stored.attrs["fusion_ui_params_hash"] == run["params_hash"]
        assert '"gain": 3.0' in stored.attrs["fusion_ui_params_json"]


def test_scalar_frame_returns_the_multi_shot_columns(conn, cache, spec, target):
    store.result(conn, spec, target, Params(), ds=None)
    store.result(
        conn, spec, dataclasses.replace(target, shot=1110201007), Params(), ds=None
    )

    frame = store.scalar_frame(conn, names=["total"])
    assert list(frame.columns) == [
        "machine",
        "shot",
        "diagnostic",
        "preprocessed",
        "plot",
        "params_hash",
        "x",
        "y",
        "name",
        "value",
    ]
    assert sorted(frame["shot"]) == [1110201007, 1160616027]
    assert set(frame["name"]) == {"total"}


def test_scalar_frame_hides_failed_runs(conn, cache, target, spec):
    def boom(ds, params):
        raise ValueError("nope")

    store.result(
        conn, dataclasses.replace(spec, compute=boom), target, Params(), ds=None
    )
    assert store.scalar_frame(conn).empty


def test_scalar_frame_is_empty_but_shaped_when_nothing_is_stored(conn, cache):
    frame = store.scalar_frame(conn)
    assert frame.empty
    assert "value" in frame.columns


# ---------------------------------------------------------------------------
# Chained specs
#
# The property that matters is that a cache hit on the derived quantity does
# not pay for its upstream: 2DCA is half a minute on a real shot, and four of
# phase 03's plots are built on the same average.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DerivedParams:
    base: Params = dataclasses.field(default_factory=Params)
    scale: float = 2.0


@pytest.fixture
def chain(spec, calls, registered):
    """A ``derived`` spec whose upstream is the synthetic one above."""
    registered(spec)

    def compute(ds, params, upstream):
        calls.append(f"derived {params.scale}")
        return xr.Dataset({"y": upstream["y"] * params.scale})

    return registered(
        registry.PlotSpec(
            key="derived",
            label="Derived",
            diagnostics=("apd",),
            params=DerivedParams,
            render=lambda result, params, target: None,
            compute=compute,
            scalars=lambda result: {"total": float(result["y"].sum())},
            requires="synthetic",
            upstream_params=lambda params: params.base,
        )
    )


@pytest.fixture
def registered():
    """Put a spec in the global registry for the duration of one test."""
    added = []

    def add(spec):
        registry.REGISTRY[spec.key] = spec
        added.append(spec.key)
        return spec

    yield add
    for key in added:
        registry.REGISTRY.pop(key, None)


def test_an_upstream_is_computed_once_and_reused(conn, cache, chain, target, calls):
    result, run = store.result(conn, chain, target, DerivedParams(), ds=None)
    assert run["status"] == "ok"
    assert list(result["y"].values) == [0.0, 2.0, 4.0, 6.0]
    assert calls == [1.0, "derived 2.0"]

    # A second derived parameter set: the upstream parameters are unchanged, so
    # the average is read from its blob rather than recomputed.
    store.result(conn, chain, target, DerivedParams(scale=3.0), ds=None)
    assert calls == [1.0, "derived 2.0", "derived 3.0"]

    plots = [r["plot"] for r in conn.execute("SELECT plot FROM runs ORDER BY id")]
    assert plots == ["synthetic", "derived", "derived"]


def test_changing_an_upstream_parameter_recomputes_both(
    conn, cache, chain, target, calls
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    store.result(conn, chain, target, DerivedParams(base=Params(gain=5.0)), ds=None)
    assert calls == [1.0, "derived 2.0", 5.0, "derived 2.0"]
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 4


def test_a_cache_hit_on_the_derived_result_does_not_touch_the_upstream(
    conn, cache, chain, target, calls
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    calls.clear()
    result, run = store.result(conn, chain, target, DerivedParams(), ds=None)
    assert calls == []
    assert list(result["y"].values) == [0.0, 2.0, 4.0, 6.0]


def test_a_failing_upstream_is_reported_on_the_derived_run(
    conn, cache, chain, spec, target, registered
):
    """The person is looking at the derived plot; the error has to name what
    actually broke rather than appear as an empty figure."""

    def boom(ds, params):
        raise ValueError("no events survived")

    registered(dataclasses.replace(spec, compute=boom))
    result, run = store.result(conn, chain, target, DerivedParams(), ds=None)

    assert result is None
    assert run["plot"] == "derived" and run["status"] == "failed"
    assert "upstream 'synthetic'" in run["error"]
    assert "no events survived" in run["error"]
    upstream = conn.execute("SELECT * FROM runs WHERE plot = 'synthetic'").fetchone()
    assert upstream["status"] == "failed"
