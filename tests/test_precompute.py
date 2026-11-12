"""The precompute engine in one process: targets from the index, params, and the run loop."""

import dataclasses
import errno
import os
import shutil

import pytest
import xarray as xr

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


def fill(conn, pixel=(2, 3), **options):
    """``taud_psd`` over the index: ``(stats, lines)``.

    The fixture is 4x5, so ``pixel=None``, the spec's default (6, 6), is out of
    range and the compute raises.
    """
    spec = registry.get("taud_psd")
    lines = []
    stats = precompute.run(
        conn,
        spec,
        precompute.targets_for(conn, spec, "cmod"),
        precompute.default_params(spec, pixel),
        log=lines.append,
        **options,
    )
    return stats, lines


def the_run(conn):
    (run,) = conn.execute("SELECT * FROM runs").fetchall()
    return run


def count_opens(monkeypatch):
    """The paths ``precompute`` opens from here on."""
    opened = []
    real_open = xr.open_dataset

    def counting(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(precompute.xr, "open_dataset", counting)
    return opened


def test_targets_for_picks_only_shots_the_spec_accepts(indexed):
    spec = registry.get("taud_psd")
    targets = precompute.targets_for(indexed, spec, "cmod")
    assert [(t.shot, t.diagnostic, t.preprocessed) for t in targets] == [
        (1234, "apd", False)
    ]
    assert precompute.targets_for(indexed, spec, "cmod", shots={999}) == []
    # An ASP-only spec matches nothing in this tree.
    assert precompute.targets_for(indexed, registry.get("probe_trace"), "cmod") == []


def test_targets_skip_the_file_version_a_spec_does_not_accept(
    indexed, apd_dataset_path
):
    """dead_pixels reads raw files only: the preprocessed file has them interpolated away."""
    shutil.copy(
        apd_dataset_path, apd_dataset_path.with_name("apd_1234_preprocessed.nc")
    )
    catalog.rescan(indexed, str(apd_dataset_path.parent.parent), "cmod", None)

    def versions(key):
        targets = precompute.targets_for(indexed, registry.get(key), "cmod")
        return [t.preprocessed for t in targets]

    assert versions("taud_psd") == [False, True]
    assert versions("dead_pixels") == [False]


def test_default_params_sets_the_reference_pixel():
    params = precompute.default_params(registry.get("taud_psd"), pixel=(2, 3))
    assert (params.refx, params.refy) == (2, 3)


def test_a_fill_computes_then_reads_the_cache_unless_forced(indexed, monkeypatch):
    first, _ = fill(indexed)
    assert (first.computed, first.cached, first.failed) == (1, 0, 0)
    # The scalar was written at the pixel it was computed for.
    value = indexed.execute(
        "SELECT value FROM scalars WHERE x = 2 AND y = 3 AND name = 'taud_psd'"
    ).fetchone()[0]
    assert value is not None

    opened = count_opens(monkeypatch)
    second, _ = fill(indexed)
    assert (second.computed, second.cached, second.failed) == (0, 1, 0)
    assert opened == [], "a cache hit must not open its file"

    forced, _ = fill(indexed, force=True)
    assert (forced.computed, forced.cached) == (1, 0)


def test_a_missing_blob_is_recomputed_not_skipped(indexed):
    """The ledger alone is not a cache hit: a cleared cache must be filled again."""
    fill(indexed)
    os.remove(the_run(indexed)["blob_path"])

    stats, _ = fill(indexed)
    assert (stats.computed, stats.cached, stats.failed) == (1, 0, 0)


def test_a_failed_compute_is_recorded_then_skipped_until_asked_to_retry(
    indexed, monkeypatch
):
    stats, _ = fill(indexed, pixel=None)
    assert (stats.failed, stats.computed) == (1, 0)
    assert the_run(indexed)["status"] == "failed"

    # ``store.result`` would only hand the failure back, so the file stays shut.
    opened = count_opens(monkeypatch)
    stats, lines = fill(indexed, pixel=None)
    assert (stats.failed, stats.computed, stats.cached) == (1, 0, 0)
    assert opened == []
    assert any("--force or --retry-failed" in line for line in lines)

    # The failed row is dropped first, or the store would return it as it is.
    stats, _ = fill(indexed, pixel=None, retry_failed=True)
    assert opened, "retry_failed must reopen the file instead of skipping"
    assert (stats.failed, stats.computed) == (1, 0)


def test_delete_run_removes_its_rows_even_when_the_blob_is_locked(indexed, monkeypatch):
    """Otherwise ``--force`` would keep returning the old cached result."""
    fill(indexed)
    run = the_run(indexed)

    def locked(*args, **kwargs):
        raise PermissionError("locked by another writer")

    monkeypatch.setattr(store.os, "remove", locked)
    store.delete_run(indexed, run)
    assert indexed.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_a_file_that_cannot_be_read_is_recorded_and_the_fill_goes_on(
    indexed, apd_dataset_path
):
    """One corrupt (``xr.open_dataset`` raises ValueError, not OSError) and one
    removed since the rescan: each a failed run, and the good file is computed."""
    folder = apd_dataset_path.parent
    (folder / "apd_1.nc").write_bytes(b"not a netCDF file")
    shutil.copy(apd_dataset_path, folder / "apd_2.nc")
    catalog.rescan(indexed, str(folder.parent), "cmod", None)
    os.remove(folder / "apd_2.nc")

    stats, lines = fill(indexed)
    assert (stats.computed, stats.failed) == (1, 2)
    text = "\n".join(lines)
    assert "[1/3]" in text and "[3/3]" in text and "ok in" in text

    errors = dict(
        indexed.execute("SELECT shot, error FROM runs WHERE status = 'failed'")
    )
    assert "ValueError" in errors[1]
    assert "FileNotFoundError" in errors[2]


def test_an_unwritable_cache_dir_is_reported_before_the_file_is_opened(
    indexed, monkeypatch
):
    """The two-writer failure: the blob directory belongs to the other account,
    so the fill reports it without paying for the open and the compute."""

    def denied(path, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied", path)

    monkeypatch.setattr(precompute.shared, "makedirs", denied)
    opened = count_opens(monkeypatch)

    stats, lines = fill(indexed)
    assert (stats.failed, stats.computed, stats.cached) == (1, 0, 0)
    assert opened == []
    assert indexed.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert any("not writable" in line for line in lines)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_a_permission_error_is_reported_not_recorded(indexed, apd_dataset_path):
    """Infrastructure is not analysis: it leaves no ``failed`` row for later
    fills to skip, so fixing the setup and rerunning is enough."""
    apd_dataset_path.chmod(0)
    try:
        stats, lines = fill(indexed)
    finally:
        apd_dataset_path.chmod(0o644)
    assert (stats.failed, stats.computed) == (1, 0)
    assert indexed.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert any(
        "failed without recording a run: PermissionError" in line for line in lines
    )

    stats, _ = fill(indexed)
    assert (stats.computed, stats.failed) == (1, 0)


@pytest.fixture
def batch_only_spec():
    """A batch-only spec, registered for one test."""

    @dataclasses.dataclass
    class BankParams:
        window: int = 60

    spec = registry.register(
        registry.PlotSpec(
            key="test_bank",
            label="Test bank",
            diagnostics=("apd",),
            params=BankParams,
            render=lambda result, params, target: None,
            compute=lambda ds, params: xr.Dataset({"y": ("t", [1.0, 2.0])}),
            batch_only=True,
        )
    )
    yield spec
    registry.REGISTRY.pop(spec.key, None)


def test_precompute_is_the_batch_job_that_fills_a_batch_only_spec(
    indexed, batch_only_spec
):
    targets = precompute.targets_for(indexed, batch_only_spec, "cmod")
    stats = precompute.run(indexed, batch_only_spec, targets, batch_only_spec.params())
    assert (stats.computed, stats.failed) == (1, 0)

    run = the_run(indexed)
    assert run["status"] == "ok"
    # What it was computed from: the file's mtime as the index has it.
    (mtime,) = indexed.execute("SELECT mtime FROM shots").fetchone()
    assert run["input_mtime"] == mtime


def test_command_names_the_shot_and_only_what_differs(indexed, batch_only_spec):
    (target,) = precompute.targets_for(indexed, batch_only_spec, "cmod")
    params = batch_only_spec.params()
    assert (
        precompute.command(batch_only_spec, target, params)
        == "fusion-ui precompute test_bank --shot 1234"
    )
    assert precompute.command(batch_only_spec, target, params, "--retry-failed") == (
        "fusion-ui precompute test_bank --shot 1234 --retry-failed"
    )
    assert precompute.command(
        batch_only_spec, target, dataclasses.replace(params, window=30)
    ) == ("fusion-ui precompute test_bank --shot 1234 --params-json params.json")
    elsewhere = dataclasses.replace(target, machine="w7x")
    assert precompute.command(batch_only_spec, elsewhere, params) == (
        "fusion-ui precompute test_bank --shot 1234 --machine w7x"
    )
