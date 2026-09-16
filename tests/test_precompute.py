"""The precompute engine: targets from the index, params, and the run loop."""

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


def test_targets_come_back_in_a_fixed_order(indexed):
    """The query orders the fill; the query plan must not get a say in it."""
    spec = registry.get("taud_psd")
    keys = [
        (t.shot, t.diagnostic, t.preprocessed)
        for t in precompute.targets_for(indexed, spec, "cmod")
    ]
    assert keys == sorted(keys)


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


def test_a_failed_run_is_counted_without_reopening_its_file(indexed, monkeypatch):
    """``store.result`` would only hand a failed run straight back, so the fill
    must not reopen a ~500 MB file to rediscover it."""
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec)  # out-of-range pixel -> failed
    targets = precompute.targets_for(indexed, spec, "cmod")

    precompute.run(indexed, spec, targets, params)

    def boom(*args, **kwargs):
        raise AssertionError("a failed run must not reopen its dataset")

    monkeypatch.setattr(precompute.xr, "open_dataset", boom)
    stats = precompute.run(indexed, spec, targets, params)
    assert (stats.failed, stats.computed, stats.cached) == (1, 0, 0)


def test_an_unreadable_file_records_a_failure_and_continues(indexed):
    """A file removed after rescan must not abort the whole overnight fill."""
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")
    os.remove(targets[0].path)

    stats = precompute.run(indexed, spec, targets, params)
    assert (stats.failed, stats.computed, stats.cached) == (1, 0, 0)

    run = indexed.execute("SELECT status, error FROM runs").fetchone()
    assert run["status"] == "failed"
    assert "FileNotFoundError" in run["error"]


def test_a_corrupt_file_is_recorded_and_does_not_abort_the_fill(
    indexed, apd_dataset_path
):
    """``xr.open_dataset`` raises ValueError, not OSError, on a non-dataset
    file -- which still must land as a failed run while the fill continues."""
    (apd_dataset_path.parent / "apd_1.nc").write_bytes(b"not a netCDF file")
    catalog.rescan(indexed, str(apd_dataset_path.parent.parent), "cmod", None)

    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")
    assert [t.shot for t in targets] == [1, 1234]  # the bad file sorts first

    lines = []
    stats = precompute.run(indexed, spec, targets, params, log=lines.append)
    assert (stats.computed, stats.failed) == (1, 1)

    text = "\n".join(lines)
    assert "[1/2]" in text and "[2/2]" in text
    assert "failed" in text and "ok in" in text

    run = indexed.execute("SELECT status, error FROM runs WHERE shot = 1").fetchone()
    assert run["status"] == "failed"
    assert "ValueError" in run["error"]


def test_run_logs_each_targets_fate(indexed):
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")

    lines = []
    precompute.run(indexed, spec, targets, params, log=lines.append)
    assert "1 targets" in "\n".join(lines)
    assert any("computing" in line for line in lines)
    assert any("ok in" in line for line in lines)

    cached_lines = []
    precompute.run(indexed, spec, targets, params, log=cached_lines.append)
    assert any("cached, skipping" in line for line in cached_lines)


def test_a_skipped_failure_points_at_force(indexed):
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec)  # out-of-range pixel -> failed
    targets = precompute.targets_for(indexed, spec, "cmod")
    precompute.run(indexed, spec, targets, params)

    lines = []
    stats = precompute.run(indexed, spec, targets, params, log=lines.append)
    assert stats.failed == 1
    assert any("--force" in line for line in lines)


def test_an_unwritable_cache_dir_skips_before_opening_the_file(
    indexed, monkeypatch
):
    """The two-writer failure: the blob directory belongs to the other account,
    so the fill must report it without paying for the file open and compute."""
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")

    def boom(*args, **kwargs):
        raise AssertionError("an unwritable cache must not open its dataset")

    monkeypatch.setattr(precompute.xr, "open_dataset", boom)
    monkeypatch.setattr(
        precompute, "_output_writable", lambda *args: (False, "/nowhere/cache")
    )

    lines = []
    stats = precompute.run(indexed, spec, targets, params, log=lines.append)
    assert (stats.failed, stats.computed, stats.cached) == (1, 0, 0)
    assert indexed.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert any("not writable" in line for line in lines)


def test_a_permission_error_is_reported_not_recorded(indexed, monkeypatch):
    """Infrastructure is not analysis: it leaves no `failed` row behind to
    poison later fills, so fixing the setup and rerunning is enough."""
    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", targets[0].path)

    monkeypatch.setattr(precompute.xr, "open_dataset", denied)

    lines = []
    stats = precompute.run(indexed, spec, targets, params, log=lines.append)
    assert (stats.failed, stats.computed) == (1, 0)
    assert indexed.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert any("without recording" in line for line in lines)


def test_output_writable_is_false_when_the_directory_cannot_be_made(
    indexed, monkeypatch
):
    import errno

    spec = registry.get("taud_psd")
    params = precompute.default_params(spec, pixel=(2, 3))
    targets = precompute.targets_for(indexed, spec, "cmod")
    params_hash, _ = store.record_params(indexed, spec.key, params)

    def denied(path, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", path)

    monkeypatch.setattr(precompute.shared, "makedirs", denied)
    ok, parent = precompute._output_writable(spec, targets[0], params_hash)
    assert ok is False and parent
