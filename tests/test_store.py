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


def test_writing_a_blob_replaces_one_this_user_cannot_write(conn, cache, target):
    """A blob left behind by the other writer carries no write bit for this
    user, so saving straight onto it would raise ``PermissionError`` -- the
    store writes aside and renames, which needs directory write only."""
    import glob
    import stat

    from fusion_ui.core import shared

    path = store.blob_path("synthetic", "abc123", target)
    store._write_blob(
        xr.Dataset({"y": 1.0}), path, "synthetic", "abc123", "{}", None, "now"
    )
    os.chmod(path, 0o444)  # what the other writer's blob looks like

    store._write_blob(
        xr.Dataset({"y": 2.0}), path, "synthetic", "abc123", "{}", None, "now"
    )
    with xr.open_dataset(path) as stored:
        assert float(stored["y"]) == 2.0
    assert glob.glob(path + ".tmp.*") == []
    assert stat.S_IMODE(os.stat(path).st_mode) == shared.FILE_MODE


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


# ---------------------------------------------------------------------------
# Reading without computing
# ---------------------------------------------------------------------------


def _raises(*args):
    raise AssertionError("compute ran where only a lookup was allowed")


def test_lookup_never_computes_and_writes_nothing(conn, cache, spec, target):
    result, run = store.lookup(
        conn, dataclasses.replace(spec, compute=_raises), target, Params()
    )
    assert (result, run) == (None, None)
    for table in ("runs", "param_sets"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_lookup_returns_what_is_stored_without_computing(
    conn, cache, spec, target, calls
):
    store.result(conn, spec, target, Params(gain=2.0), ds=None)
    result, run = store.lookup(
        conn, dataclasses.replace(spec, compute=_raises), target, Params(gain=2.0)
    )
    assert run["status"] == "ok"
    assert result["y"].values.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert calls == [2.0]


def test_lookup_hands_back_a_recorded_failure_as_its_run(conn, cache, spec, target):
    def boom(ds, params):
        raise ValueError("no events found")

    store.result(conn, dataclasses.replace(spec, compute=boom), target, Params(), None)
    result, run = store.lookup(conn, spec, target, Params())
    assert result is None
    assert run["status"] == "failed" and "no events found" in run["error"]


def test_lookup_reports_an_ok_row_whose_blob_is_gone(conn, cache, spec, target):
    _, stored = store.result(conn, spec, target, Params(), ds=None)
    os.remove(stored["blob_path"])
    result, run = store.lookup(conn, spec, target, Params())
    assert result is None
    assert run["status"] == "ok" and run["id"] == stored["id"]


def test_lookup_never_resolves_an_upstream(
    conn, cache, chain, spec, target, registered
):
    registered(dataclasses.replace(spec, compute=_raises))
    assert store.lookup(conn, chain, target, DerivedParams()) == (None, None)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_lookup_refuses_a_live_spec(conn, cache, spec, target):
    with pytest.raises(ValueError, match="live spec"):
        store.lookup(conn, dataclasses.replace(spec, compute=None), target, Params())


# ---------------------------------------------------------------------------
# Batch-only specs: computed by a batch job, never on a cache miss in a page
# ---------------------------------------------------------------------------


@pytest.fixture
def bank(spec, calls, registered):
    """A batch-only spec, standing in for the 2DCA at every pixel."""

    def compute(ds, params):
        calls.append(f"bank {params.gain}")
        return xr.Dataset({"y": ("t", np.arange(4.0) * params.gain)})

    return registered(
        dataclasses.replace(spec, key="bank", compute=compute, batch_only=True)
    )


@pytest.fixture
def fields(bank, calls, registered):
    """An ordinary cached spec built on the batch-only one."""

    def compute(ds, params, upstream):
        calls.append(f"fields {params.scale}")
        return xr.Dataset({"y": upstream["y"] * params.scale})

    return registered(
        registry.PlotSpec(
            key="fields",
            label="Fields",
            diagnostics=("apd",),
            params=DerivedParams,
            render=lambda result, params, target: None,
            compute=compute,
            requires="bank",
            upstream_params=lambda params: params.base,
        )
    )


def test_a_batch_only_spec_is_computed_only_by_a_batch_job(
    conn, cache, bank, target, calls
):
    with pytest.raises(store.BatchOnlyError, match="batch only") as raised:
        store.result(conn, bank, target, Params(), ds=None)
    assert raised.value.spec is bank and raised.value.params == Params()
    assert calls == []
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0

    _, run = store.result(conn, bank, target, Params(), ds=None, batch=True)
    assert run["status"] == "ok" and calls == ["bank 1.0"]

    # Once it is cached, anyone may read it.
    result, again = store.result(conn, bank, target, Params(), ds=None)
    assert again["id"] == run["id"] and calls == ["bank 1.0"]
    assert result["y"].values.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_chain_never_starts_a_missing_batch_only_upstream(
    conn, cache, fields, target, calls
):
    with pytest.raises(store.BatchOnlyError) as raised:
        store.result(conn, fields, target, DerivedParams(), ds=None)
    assert raised.value.spec.key == "bank"
    assert calls == []
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0

    store.result(conn, fields, target, DerivedParams(), ds=None, batch=True)
    assert calls == ["bank 1.0", "fields 2.0"]

    # With the bank cached, a new derived parameter set computes on demand,
    # and only the derived part.
    _, run = store.result(conn, fields, target, DerivedParams(scale=3.0), ds=None)
    assert run["status"] == "ok"
    assert calls == ["bank 1.0", "fields 2.0", "fields 3.0"]


def test_an_unreadable_batch_only_blob_is_refused_not_recomputed(
    conn, cache, fields, bank, target, calls
):
    """The race a page's check cannot close: the bank's blob broke between
    the check and the compute. The store itself must still refuse."""
    _, run = store.result(conn, bank, target, Params(), ds=None, batch=True)
    with open(run["blob_path"], "wb") as broken:
        broken.write(b"not netCDF")

    with pytest.raises(store.BatchOnlyError):
        store.result(conn, fields, target, DerivedParams(), ds=None)
    assert calls == ["bank 1.0"]


def test_nothing_is_missing_without_a_batch_only_link(conn, cache, chain, target):
    assert store.missing_batch_upstreams(conn, chain, target, DerivedParams()) == []


def test_a_missing_batch_only_spec_is_listed_with_its_parameters(
    conn, cache, bank, target
):
    assert store.missing_batch_upstreams(conn, bank, target, Params(gain=2.0)) == [
        (bank, Params(gain=2.0), None)
    ]
    store.result(conn, bank, target, Params(gain=2.0), ds=None, batch=True)
    assert store.missing_batch_upstreams(conn, bank, target, Params(gain=2.0)) == []


def test_a_derived_spec_lists_the_batch_only_link_it_would_need(
    conn, cache, fields, bank, target
):
    params = DerivedParams(base=Params(gain=5.0))
    assert store.missing_batch_upstreams(conn, fields, target, params) == [
        (bank, Params(gain=5.0), None)
    ]
    store.result(conn, bank, target, Params(gain=5.0), ds=None, batch=True)
    assert store.missing_batch_upstreams(conn, fields, target, params) == []


def test_a_cached_derived_result_needs_nothing_beneath_it(
    conn, cache, fields, bank, target
):
    """A cache hit is not resolved further down, so a bank deleted since does
    not stop the derived result being shown."""
    store.result(conn, fields, target, DerivedParams(), ds=None, batch=True)
    _, bank_run = store.result(conn, bank, target, Params(), ds=None)
    store.delete_run(conn, bank_run)
    assert store.missing_batch_upstreams(conn, fields, target, DerivedParams()) == []


def test_a_failed_or_vanished_bank_is_listed_with_its_run(
    conn, cache, fields, bank, target
):
    def boom(ds, params):
        raise ValueError("no events survived")

    failing = dataclasses.replace(bank, compute=boom)
    store.result(conn, failing, target, Params(), ds=None, batch=True)
    [(link, _, run)] = store.missing_batch_upstreams(
        conn, fields, target, DerivedParams()
    )
    assert link is bank and run["status"] == "failed"

    store.delete_run(conn, run)
    _, ok = store.result(conn, bank, target, Params(), ds=None, batch=True)
    os.remove(ok["blob_path"])
    [(_, _, run)] = store.missing_batch_upstreams(conn, fields, target, DerivedParams())
    assert run["status"] == "ok" and run["id"] == ok["id"]


# ---------------------------------------------------------------------------
# What a run was computed from (schema v4), and staleness
# ---------------------------------------------------------------------------


@pytest.fixture
def indexed(conn, tmp_path):
    """Put the target's input file in the index, as rescan finds it."""
    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    path = folder / "apd_1160616027_preprocessed.nc"
    path.write_bytes(b"x")
    rewrite(conn, path, 1_700_000_000)
    return path


def rewrite(conn, path, seconds):
    """The file changed on disk, and a rescan saw it."""
    from fusion_ui.core import catalog

    os.utime(path, (seconds, seconds))
    catalog.rescan(conn, str(path.parent.parent), "cmod", None)


def index_mtime(conn):
    return conn.execute("SELECT mtime FROM shots").fetchone()[0]


def runs_by_plot(conn):
    return {r["plot"]: r for r in conn.execute("SELECT * FROM runs ORDER BY id")}


def stale(conn, plot=None):
    return {r["plot"]: r["stale"] for r in store.stale_runs(conn, plot)}


def test_a_run_records_its_input_and_the_run_it_was_built_on(
    conn, cache, indexed, chain, target
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    runs = runs_by_plot(conn)
    assert runs["derived"]["upstream_run_id"] == runs["synthetic"]["id"]
    assert runs["synthetic"]["upstream_run_id"] is None
    # The index's own string, so that the two compare directly.
    assert runs["derived"]["input_mtime"] == index_mtime(conn)
    assert runs["synthetic"]["input_mtime"] == index_mtime(conn)
    assert stale(conn) == {}


def test_a_failed_downstream_records_its_upstream_too(
    conn, cache, indexed, chain, spec, target, registered
):
    def boom(ds, params):
        raise ValueError("no events survived")

    registered(dataclasses.replace(spec, compute=boom))
    store.result(conn, chain, target, DerivedParams(), ds=None)
    runs = runs_by_plot(conn)
    assert runs["derived"]["status"] == "failed"
    assert runs["derived"]["upstream_run_id"] == runs["synthetic"]["id"]


def test_recomputing_the_upstream_in_place_makes_the_downstream_stale(
    conn, cache, indexed, chain, spec, target
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    before = runs_by_plot(conn)

    store.compute_and_store(conn, spec, target, Params(), ds=None)
    after = runs_by_plot(conn)
    assert after["synthetic"]["id"] == before["synthetic"]["id"], "not in place"
    assert after["derived"]["upstream_run_id"] == after["synthetic"]["id"]
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_RECOMPUTED}

    # Recomputing the downstream in place makes it current again.
    store.compute_and_store(conn, chain, target, DerivedParams(), ds=None)
    assert stale(conn) == {}


def test_deleting_the_upstream_makes_the_downstream_stale(
    conn, cache, indexed, chain, spec, target
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    store.delete_run(conn, runs_by_plot(conn)["synthetic"])
    assert runs_by_plot(conn)["derived"]["upstream_run_id"] is None
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_DELETED}

    # Computing the upstream again gives a new row, not the one this result
    # was built on: still stale.
    store.result(conn, spec, target, Params(), ds=None)
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_DELETED}


def test_a_rewritten_input_makes_every_run_on_it_stale(
    conn, cache, indexed, chain, spec, target
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    elsewhere = dataclasses.replace(target, shot=1110201007)
    store.result(conn, spec, elsewhere, Params(), ds=None)

    rewrite(conn, indexed, 1_700_000_600)
    assert stale(conn) == {
        "synthetic": store.STALE_INPUT,
        "derived": store.STALE_INPUT,
    }
    # The other shot is not indexed: unknown, so never listed.
    assert {r["shot"] for r in store.stale_runs(conn)} == {target.shot}


@dataclasses.dataclass
class TopParams:
    derived: DerivedParams = dataclasses.field(default_factory=DerivedParams)
    offset: float = 1.0


def test_staleness_follows_the_chain_all_the_way_up(
    conn, cache, indexed, chain, spec, target, registered
):
    top = registered(
        registry.PlotSpec(
            key="top",
            label="Top",
            diagnostics=("apd",),
            params=TopParams,
            render=lambda result, params, target: None,
            compute=lambda ds, params, upstream: upstream + params.offset,
            requires="derived",
            upstream_params=lambda params: params.derived,
        )
    )
    store.result(conn, top, target, TopParams(), ds=None)
    assert stale(conn) == {}

    store.compute_and_store(conn, spec, target, Params(), ds=None)
    assert stale(conn) == {
        "derived": store.STALE_UPSTREAM_RECOMPUTED,
        "top": store.STALE_UPSTREAM_STALE,
    }
    assert stale(conn, plot="top") == {"top": store.STALE_UPSTREAM_STALE}


def test_a_failed_downstream_is_stale_once_its_upstream_is_retried(
    conn, cache, indexed, chain, spec, target, registered
):
    """So `--stale` retries a failure that was only its upstream's."""

    def boom(ds, params):
        raise ValueError("transient")

    registered(dataclasses.replace(spec, compute=boom))
    store.result(conn, chain, target, DerivedParams(), ds=None)
    assert stale(conn) == {}

    registered(spec)
    store.delete_run(conn, runs_by_plot(conn)["synthetic"])
    store.result(conn, spec, target, Params(), ds=None)
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_DELETED}
    assert store.stale_runs(conn)[0]["status"] == "failed"


def test_runs_without_provenance_are_unknown_never_stale(
    conn, cache, indexed, chain, target
):
    """What every row written before schema v4 looks like: legacy results
    must not all be recomputed at once."""
    store.result(conn, chain, target, DerivedParams(), ds=None)
    with conn:
        conn.execute("UPDATE runs SET input_mtime = NULL, upstream_run_id = NULL")

    rewrite(conn, indexed, 1_700_000_600)
    store.delete_run(conn, runs_by_plot(conn)["synthetic"])
    assert store.stale_runs(conn) == []


def test_a_target_outside_the_index_records_no_input_and_is_unknown(
    conn, cache, chain, target
):
    store.result(conn, chain, target, DerivedParams(), ds=None)
    assert all(r["input_mtime"] is None for r in runs_by_plot(conn).values())
    store.delete_run(conn, runs_by_plot(conn)["synthetic"])
    assert store.stale_runs(conn) == []


def test_a_run_whose_input_left_the_index_is_never_listed(
    conn, cache, indexed, chain, target, tmp_path
):
    """Nothing could recompute it, so `precompute --stale` must not aim at it."""
    from fusion_ui.core import catalog

    store.result(conn, chain, target, DerivedParams(), ds=None)
    store.delete_run(conn, runs_by_plot(conn)["synthetic"])
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_DELETED}
    mtime = index_mtime(conn)

    # The file leaves the data tree and a rescan drops its shots row.
    away = tmp_path / "away.nc"
    os.rename(indexed, away)
    catalog.rescan(conn, str(indexed.parent.parent), "cmod", None)
    assert conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0] == 0
    assert store.stale_runs(conn) == []

    # Back with the same mtime (a rename keeps it): judged again.
    os.rename(away, indexed)
    catalog.rescan(conn, str(indexed.parent.parent), "cmod", None)
    assert index_mtime(conn) == mtime
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_DELETED}


def test_an_unknown_run_is_not_listed_through_its_upstream_either(
    conn, cache, indexed, chain, spec, target, registered
):
    """A run with no input_mtime but a link -- written before its target was
    indexed -- stays unknown even when the run beneath it goes stale."""
    top = registered(
        registry.PlotSpec(
            key="top",
            label="Top",
            diagnostics=("apd",),
            params=TopParams,
            render=lambda result, params, target: None,
            compute=lambda ds, params, upstream: upstream + params.offset,
            requires="derived",
            upstream_params=lambda params: params.derived,
        )
    )
    store.result(conn, top, target, TopParams(), ds=None)
    with conn:
        conn.execute("UPDATE runs SET input_mtime = NULL WHERE plot = 'top'")

    store.compute_and_store(conn, spec, target, Params(), ds=None)
    assert stale(conn) == {"derived": store.STALE_UPSTREAM_RECOMPUTED}


def test_record_run_never_keeps_the_previous_provenance(conn, cache, target):
    """An update in place that is not given the columns clears them, rather
    than keeping the last attempt's on a row that no longer has them."""
    digest, _ = store.record_params(conn, "synthetic", Params())
    store.record_run(conn, target, "synthetic", digest, status="ok", input_mtime="t")
    run = store.record_run(conn, target, "synthetic", digest, status="failed")
    assert run["input_mtime"] is None and run["upstream_run_id"] is None
