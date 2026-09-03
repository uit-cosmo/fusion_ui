"""The precompute engine: targets from the index, params, and the run loop."""

import dataclasses
import os

import pytest

import fusion_ui.plots  # noqa: F401 - registers taud_psd etc.
from fusion_ui.core import catalog, db, precompute, registry, store


@pytest.fixture
def indexed(monkeypatch, tmp_path, apd_dataset_path):
    """One indexed, real (tiny) APD file, no discharge metadata."""
    data_folder = apd_dataset_path.parent.parent  # .../alcator
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(data_folder))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(tmp_path / "no_discharges.json"))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")

    conn = db.open_db(database)
    catalog.rescan(conn, str(data_folder), "cmod", None)
    yield conn
    conn.close()


def test_targets_for_picks_only_shots_the_spec_accepts(indexed):
    spec = registry.get("taud_psd")
    targets = precompute.targets_for(indexed, spec, "cmod")
    assert [(t.shot, t.diagnostic, t.preprocessed) for t in targets] == [
        (1234, "apd", False)
    ]
    # An ASP-only spec matches nothing in this tree.
    assert precompute.targets_for(indexed, registry.get("probe_trace"), "cmod") == []


def test_targets_can_be_restricted_to_a_shot(indexed):
    spec = registry.get("taud_psd")
    assert precompute.targets_for(indexed, spec, "cmod", shots={999}) == []


def test_default_params_sets_the_reference_pixel(indexed):
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    assert (params.refx, params.refy) == (2, 3)
    # A nested refx/refy (the 2DCA-derived specs) is set too.
    derived = precompute.default_params(registry.get("velocity_contour"), pixel=(1, 2))
    assert (derived.two_dca.refx, derived.two_dca.refy) == (1, 2)


def test_a_full_fill_computes_then_reads_the_cache(indexed):
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")

    first = precompute.run(indexed, spec, targets, params)
    assert (first.computed, first.cached, first.failed) == (1, 0, 0)

    # The scalar was written at the pixel it was computed for.
    value = indexed.execute(
        "SELECT s.value FROM scalars s WHERE s.x = 2 AND s.y = 3"
        " AND s.name = 'taud_psd'"
    ).fetchone()[0]
    assert value is not None

    second = precompute.run(indexed, spec, targets, params)
    assert (second.computed, second.cached, second.failed) == (0, 1, 0)


def test_force_recomputes_an_existing_result(indexed):
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")

    precompute.run(indexed, spec, targets, params)
    forced = precompute.run(indexed, spec, targets, params, force=True)
    assert (forced.computed, forced.cached) == (1, 0)


def test_a_missing_blob_is_recomputed_not_skipped(indexed):
    """An 'ok' run whose blob was cleared must be re-warmed, not skipped: the
    ledger alone is not enough to count a cache hit."""
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")

    precompute.run(indexed, spec, targets, params)
    params_hash, _ = store.record_params(indexed, spec.key, params)
    run = store.find_run(indexed, targets[0], spec.key, params_hash)
    assert run["status"] == "ok"
    os.remove(run["blob_path"])

    stats = precompute.run(indexed, spec, targets, params)
    assert (stats.computed, stats.cached, stats.failed) == (1, 0, 0)


def test_a_failed_compute_is_counted_not_raised(indexed):
    spec = registry.get("taud_psd")
    # The tiny fixture is 4x5, so the default reference pixel (6, 6) is out of
    # range and compute raises -- which must land as a failed run, not a traceback.
    targets = precompute.targets_for(indexed, spec, "cmod")
    stats = precompute.run(indexed, spec, targets, precompute.default_params(spec))
    assert (stats.failed, stats.computed) == (1, 0)

    run = indexed.execute("SELECT status FROM runs").fetchone()
    assert run["status"] == "failed"
