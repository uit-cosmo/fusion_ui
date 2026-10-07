"""Which products exist for a shot, under which settings, and what fills the ones that do not.

Seeded through the store with stand-in specs (``tests/fields_fixtures``). The point of most of these
is what the page may rely on: reading the ledger writes nothing, and a missing product comes with
the exact command that fills it.
"""

import dataclasses
import json
import os

import pytest

from fusion_ui.core import catalog, params_ui, registry, store
from fusion_ui.views import products as prod
from tests import fields_fixtures as ff
from tests.fields_fixtures import fields_world  # noqa: F401 - the fixture

SHOT = 1160616027


class Stored:
    """A database with the shot indexed and an empty cache, and the stand-in world that fills it."""

    def __init__(self, world, root, path):
        self.world, self.root, self.path = world, root, path

    def seed(self, conn, shot, *keys, **kwargs):
        return self.world.seed(conn, shot, *keys, **kwargs)

    @property
    def calls(self):
        return self.world.calls


@pytest.fixture
def stored(conn, tmp_path, monkeypatch, fields_world):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    path = folder / f"apd_{SHOT}_preprocessed.nc"
    path.write_bytes(b"x")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    (folder / "apd_1160616026_preprocessed.nc").write_bytes(b"x")
    (folder / "apd_1160616026.nc").write_bytes(b"x")
    catalog.rescan(conn, str(tmp_path / "alcator"), "cmod", None)
    return Stored(fields_world, tmp_path / "alcator", path)


def target():
    return registry.Target(
        "cmod", SHOT, "apd", True, "unused", float("nan"), float("nan"), "none"
    )


def collect(conn, found=None, params=None, **kwargs):
    found = found or prod.specs(registry)[0]
    params = params or found["method_fields"].params()
    return prod.collect(
        conn,
        target(),
        found,
        params,
        lambda run: store.load_result(conn, run),
        **kwargs,
    )


def counts(conn):
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("runs", "param_sets", "scalars", "presets", "shots")
    }


# -- settings ---------------------------------------------------------------------------------


def test_a_settings_difference_is_a_list_of_the_leaves_that_differ():
    spec = ff.World().specs()["method_fields"]
    _, default = params_ui.hash_params("method_fields", spec.params())
    short = ff.with_window(40)
    _, text = params_ui.hash_params("method_fields", short)
    assert prod.params_diff(default, text) == (("averages.window", 60, 40),)
    assert prod.params_diff(default, default) == ()
    two = dataclasses.replace(
        short, tracking=dataclasses.replace(short.tracking, neighbour_step=2)
    )
    diff = prod.params_diff(default, params_ui.hash_params("method_fields", two)[1])
    assert diff == (("averages.window", 60, 40), ("tracking.neighbour_step", 1, 2))
    assert prod.describe(()) == "default"
    assert prod.describe(diff) == "averages.window = 40; tracking.neighbour_step = 2"
    many = tuple((f"p{i}", 0, 1) for i in range(5))
    assert prod.describe(many).endswith("+2 more")


def test_the_settings_picker_offers_the_default_then_every_other_set_with_a_run(
    stored, conn
):
    spec = prod.specs(registry)[0]["method_fields"]
    default_hash = params_ui.hash_params("method_fields", spec.params())[0]
    options = prod.settings(conn, spec, "cmod")
    assert [o.label for o in options] == ["default"] and options[0].hash == default_hash
    assert options[0].is_default and options[0].shots == 0

    stored.seed(conn, SHOT)
    stored.seed(conn, 1160616026, "method_fields")
    stored.seed(
        conn, SHOT, "method_fields", params={"method_fields": ff.with_window(40)}
    )
    options = prod.settings(conn, spec, "cmod")
    assert [o.label for o in options] == ["default", "averages.window = 40"]
    assert [o.shots for o in options] == [2, 1]
    assert (
        options[1].diff == (("averages.window", 60, 40),) and not options[1].is_default
    )

    # A parameter set nothing ran with is not offered, and nor is another plot's.
    store.record_params(conn, "method_fields", ff.with_window(30))
    assert [o.label for o in prod.settings(conn, spec, "cmod")] == [
        "default",
        "averages.window = 40",
    ]
    # Nor another machine's.
    assert [o.label for o in prod.settings(conn, spec, "aug")] == ["default"]


def test_a_parameter_set_this_version_cannot_reproduce_is_not_offered(stored, conn):
    """One stored with a leaf the parameters no longer have (or without one they now have) would
    be rebuilt to another hash, and the page would look for its products under a key they are not
    under. It is left out of the picker rather than offered broken."""
    spec = prod.specs(registry)[0]["method_fields"]
    stored.seed(
        conn,
        SHOT,
        "pixel_averages",
        "method_fields",
        params={"method_fields": ff.with_window(40)},
    )
    assert [o.label for o in prod.settings(conn, spec, "cmod")] == [
        "default",
        "averages.window = 40",
    ]
    short_hash = prod.settings(conn, spec, "cmod")[1].hash
    row = conn.execute(
        "SELECT params_json FROM param_sets WHERE hash = ?", (short_hash,)
    ).fetchone()
    body = json.loads(row["params_json"])
    del body["params"]["values"]["averages"][
        "window"
    ]  # a leaf the stored set has and this one reads
    drifted = json.dumps(body, indent=2)
    with conn:
        conn.execute(
            "UPDATE param_sets SET params_json = ? WHERE hash = ?",
            (drifted, short_hash),
        )
    assert [o.label for o in prod.settings(conn, spec, "cmod")] == ["default"]
    with conn:
        conn.execute(
            "UPDATE param_sets SET params_json = ? WHERE hash = ?",
            ("not json", short_hash),
        )
    assert [o.label for o in prod.settings(conn, spec, "cmod")] == ["default"]


def test_the_other_products_follow_from_the_settings(stored, conn):
    found = prod.specs(registry)[0]
    default = found["method_fields"].params()
    related = prod.related_params(found, default)
    for key in prod.KEYS:  # at the defaults every product is at its own default
        assert (
            params_ui.hash_params(key, related[key])[0]
            == params_ui.hash_params(key, found[key].params())[0]
        )

    short = ff.with_window(40)
    related = prod.related_params(found, short)
    assert (
        related["pixel_averages"].averages.window == 40
    )  # exactly what method_fields is chained to
    assert related["pixel_averages"] == found["method_fields"].upstream_params(short)
    assert related["blob_parameters"].averages.window == 40
    # The contour's neighbour step is carried over, and the rest stays the blob parameters' own.
    stepped = dataclasses.replace(
        short, tracking=dataclasses.replace(short.tracking, neighbour_step=2)
    )
    blobs = prod.related_params(found, stepped)["blob_parameters"]
    assert (
        blobs.neighbour_step == 2
        and blobs.blobs == found["blob_parameters"].params().blobs
    )
    assert prod.neighbour_step(stepped) == 2 and prod.neighbour_step(object()) == 1


def test_the_shots_offered_are_the_preprocessed_ones_and_the_computed_ones_are_known(
    stored, conn
):
    assert prod.preprocessed_shots(conn, "cmod") == [
        (1160616026, str(stored.root / "apd" / "apd_1160616026_preprocessed.nc")),
        (SHOT, str(stored.path)),
    ]
    assert prod.preprocessed_shots(conn, "aug") == []
    spec = prod.specs(registry)[0]["method_fields"]
    digest = params_ui.hash_params("method_fields", spec.params())[0]
    assert prod.good_shots(conn, "cmod", digest) == set()
    stored.seed(conn, 1160616026, "method_fields")
    assert prod.good_shots(conn, "cmod", digest) == {1160616026}


# -- what exists ------------------------------------------------------------------------------


def test_nothing_computed_means_three_missing_products_and_the_command_for_each(
    stored, conn
):
    products = collect(conn)
    assert [p.state for p in products.values()] == [prod.MISSING] * 3
    assert [p.command for p in products.values()] == [
        f"fusion-ui precompute pixel_averages --shot {SHOT}",
        f"fusion-ui precompute method_fields --shot {SHOT}",
        f"fusion-ui precompute blob_parameters --shot {SHOT}",
    ]
    assert not any(p.ok or p.dataset is not None for p in products.values())
    assert all(p.params_hash for p in products.values())


def test_other_settings_name_the_parameters_to_save(stored, conn):
    products = collect(conn, params=ff.with_window(40))
    assert all(
        p.command.endswith("--params-json params.json") for p in products.values()
    )
    assert products["pixel_averages"].params.averages.window == 40
    assert products["blob_parameters"].params.averages.window == 40
    assert (
        products["method_fields"].params_hash
        != collect(conn)["method_fields"].params_hash
    )


def test_a_computed_shot_has_its_three_products_loaded_and_nothing_to_say(stored, conn):
    runs = stored.seed(conn, SHOT)
    products = collect(conn)
    assert [p.state for p in products.values()] == [prod.OK] * 3
    assert all(
        p.ok and p.stale is None and p.code_note is None and p.command is None
        for p in products.values()
    )
    assert products["method_fields"].run["id"] == runs["method_fields"]["id"]
    assert "vr_com" in products["method_fields"].dataset
    assert products["pixel_averages"].dataset["cond_av"].dims == (
        "ref_y",
        "ref_x",
        "y",
        "x",
        "time",
    )
    assert products["blob_parameters"].dataset["taud"].dims == ("y", "x")
    # The method_fields run is the bank's child, and the page can tell.
    assert (
        products["method_fields"].run["upstream_run_id"]
        == products["pixel_averages"].run["id"]
    )


def test_reading_the_ledger_writes_nothing_and_computes_nothing(
    stored, conn, monkeypatch
):
    stored.seed(conn, SHOT, "pixel_averages")
    before = counts(conn)

    def refused(*args, **kwargs):
        raise AssertionError("the page must never compute or record")

    for name in (
        "result",
        "compute_and_store",
        "record_params",
        "record_run",
        "write_scalars",
        "delete_run",
    ):
        monkeypatch.setattr(store, name, refused)
    for params in (None, ff.with_window(40)):
        products = collect(conn, params=params)
        assert counts(conn) == before
    assert stored.calls == []
    assert (
        products["pixel_averages"].state == prod.MISSING
    )  # (window 40 was never computed)


def test_a_failure_is_shown_with_the_command_that_retries_it(stored, conn):
    spec = prod.specs(registry)[0]["method_fields"]

    def boom(ds, params, upstream):
        raise ValueError("no events survived")

    failing = dataclasses.replace(spec, compute=boom)
    _, run = store.result(conn, failing, target(), spec.params(), None, batch=True)
    assert run["status"] == "failed"
    products = collect(conn)
    assert products["method_fields"].state == prod.FAILED
    assert "no events survived" in products["method_fields"].run["error"]
    assert (
        products["method_fields"].command
        == f"fusion-ui precompute method_fields --shot {SHOT} --retry-failed"
    )
    assert products[
        "pixel_averages"
    ].ok  # its bank was computed on the way, and is still there
    assert products["blob_parameters"].state == prod.MISSING


def test_a_blob_that_has_gone_is_unreadable_and_gets_the_command_that_replaces_it(
    stored, conn
):
    runs = stored.seed(conn, SHOT)
    os.remove(runs["method_fields"]["blob_path"])
    products = collect(conn)
    assert products["method_fields"].state == prod.UNREADABLE
    assert products["method_fields"].command.endswith("--force")
    with open(runs["blob_parameters"]["blob_path"], "wb") as broken:
        broken.write(b"not netCDF")
    assert collect(conn)["blob_parameters"].state == prod.UNREADABLE


def test_a_product_whose_input_changed_is_stale_and_says_how_to_refresh_it(
    stored, conn
):
    stored.seed(conn, SHOT)
    assert prod.stale_reasons(conn) == {}
    # The file was rewritten (a corrected mask) and a rescan saw it.
    os.utime(stored.path, (1_700_000_500, 1_700_000_500))
    catalog.rescan(conn, str(stored.root), "cmod", None)
    products = collect(conn, stale=prod.stale_reasons(conn))
    assert [p.stale for p in products.values()] == ["input changed"] * 3
    assert (
        products["method_fields"].refresh
        == f"fusion-ui precompute method_fields --shot {SHOT} --force"
    )
    assert all(
        p.ok for p in products.values()
    )  # still shown: stale is information, not an error


def test_a_recomputed_bank_makes_what_was_built_on_it_stale(stored, conn):
    runs = stored.seed(conn, SHOT)
    found = prod.specs(registry)[0]
    store.delete_run(conn, runs["pixel_averages"])
    stored.seed(conn, SHOT, "pixel_averages")
    products = collect(conn, stale=prod.stale_reasons(conn))
    assert products["pixel_averages"].stale is None
    assert products["method_fields"].stale == "upstream deleted"
    assert found["method_fields"].requires == "pixel_averages"


def test_a_product_computed_under_another_fusion_scripts_commit_is_marked(
    stored, conn, monkeypatch
):
    monkeypatch.setattr(
        store,
        "_code_version",
        lambda: "fusion_scripts=1111111 fusion_ui=aaaaaaa velocity_estimation=bbbbbbb",
    )
    stored.seed(conn, SHOT)
    moved = {"fusion_scripts": "2222222", "fusion_ui": "ccccccc"}
    notes = [p.code_note for p in collect(conn, current=moved).values()]
    assert (
        notes
        == ["computed under fusion_scripts 1111111, the checkout is now at 2222222"] * 3
    )
    # The same commit, dirty or not, is the same code.
    assert not any(
        p.code_note
        for p in collect(conn, current={"fusion_scripts": "1111111-dirty"}).values()
    )
    # Another repository moving on is not the page's business, and an unreadable checkout says nothing.
    assert not any(
        p.code_note for p in collect(conn, current={"fusion_ui": "ccccccc"}).values()
    )
    assert not any(
        p.code_note
        for p in collect(conn, current={"fusion_scripts": "unknown"}).values()
    )
    assert not any(p.code_note for p in collect(conn, current=None).values())


def test_runs_from_before_the_versions_were_recorded_are_not_marked(
    stored, conn, monkeypatch
):
    monkeypatch.setattr(
        store, "_code_version", lambda: "fusion_ui=aaaaaaa imaging_methods=bbbbbbb"
    )
    stored.seed(conn, SHOT)
    assert not any(
        p.code_note
        for p in collect(conn, current={"fusion_scripts": "2222222"}).values()
    )


def test_a_spec_that_is_not_registered_is_said_so_and_nothing_is_offered_for_it(
    stored, conn
):
    found, missing = prod.specs(registry)
    assert missing == []
    partial = {k: v for k, v in found.items() if k != "blob_parameters"}
    products = collect(conn, found=partial)
    assert (
        products["blob_parameters"].state == prod.UNREGISTERED
        and products["blob_parameters"].command is None
    )
    assert products["method_fields"].state == prod.MISSING
    found, missing = prod.specs(
        types_namespace({"method_fields": found["method_fields"]})
    )
    assert missing == ["pixel_averages", "blob_parameters"]


def types_namespace(entries):
    import types

    return types.SimpleNamespace(REGISTRY=entries)
