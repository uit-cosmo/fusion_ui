"""Phase 06's precompute: several plots, a pool of workers, --stale, --params-json.

Toy specs (``tests/precompute_toys``) compute instantly on four tiny APD files
over two run days, so what is tested is the engine: what runs where, what is
written, and what a Ctrl-C leaves behind. A pool worker is a spawned process,
so the toys reach it by name (``modules``), never through a fixture; in this
process they are registered for this module's tests only.
"""

import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import fusion_ui.plots  # noqa: F401 - registers the real specs
from fusion_ui import cli
from fusion_ui.core import catalog, db, params_ui, precompute, registry, store
from tests import precompute_toys as toys

REPO = Path(__file__).resolve().parent.parent
TOYS = (toys.REGISTERING_MODULE,)


@pytest.fixture(autouse=True, scope="module")
def registered_toys():
    """The toys, for these tests only: a page under test elsewhere lists every
    registered spec."""
    toys.register()
    yield
    toys.unregister()


#: Shot -> samples. Different lengths make different file sizes, so the order
#: a pool takes them in (largest first) is known: 1140827001 first.
SHOTS = {1160616001: 300, 1160616002: 200, 1140827001: 400, 1140827002: 100}
#: The descriptor's window on 1160616001; the other shots have none, so their
#: whole (0.02 s) record is used.
WINDOW = (1.005, 1.015)


def write_record(path, shot, n_time):
    rng = np.random.default_rng(shot % 97)
    xr.Dataset(
        {"frames": (["y", "x", "time"], rng.normal(size=(4, 5, n_time)))},
        coords={
            "R": (["y", "x"], np.tile(np.linspace(80.0, 90.0, 5), (4, 1))),
            "Z": (["y", "x"], np.tile(np.linspace(-4.0, 4.0, 4), (5, 1)).T),
            "time": ("time", np.linspace(1.0, 1.02, n_time)),
        },
        attrs={"shot_number": shot},
    ).to_netcdf(path)


@pytest.fixture
def tree(monkeypatch, tmp_path):
    """Four indexed APD files, a descriptor with one window, and the env."""
    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    for shot, n_time in SHOTS.items():
        write_record(folder / f"apd_{shot}.nc", shot, n_time)
    discharges = tmp_path / "plasma_discharges.json"
    discharges.write_text(
        json.dumps(
            [
                {
                    "shot_number": 1160616001,
                    "plasma_current": 0.55,
                    "line_averaged_density": 1.4,
                    "greenwald_fraction": 0.7,
                    "t_start": WINDOW[0],
                    "t_end": WINDOW[1],
                    "mlp_mode": "",
                    "comment": "",
                }
            ]
        )
    )
    database = tmp_path / "state" / "shot_explorer.sqlite"
    monkeypatch.setenv("FUSION_DATA_FOLDER", str(tmp_path / "alcator"))
    monkeypatch.setenv("FUSION_DISCHARGE_DB", str(discharges))
    monkeypatch.setenv("FUSION_UI_DB", str(database))
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FUSION_MACHINE", "cmod")
    conn = db.open_db(database)
    catalog.rescan(conn, str(tmp_path / "alcator"), "cmod", str(discharges))
    yield SimpleNamespace(
        conn=conn,
        root=tmp_path,
        folder=folder,
        cache=tmp_path / "cache",
        discharges=discharges,
    )
    conn.close()


def fill(conn, plots, workers=1, force=False, retry_failed=False, **selection):
    """The CLI's fill, through the API, with the toys known to the workers."""
    specs = precompute.in_dependency_order([registry.get(key) for key in plots])
    targets = precompute.select_targets(conn, specs, "cmod", **selection)
    plan = precompute.plan_fill(
        conn,
        [(spec, spec.params()) for spec in specs],
        targets,
        force=force,
        retry_failed=retry_failed,
    )
    lines = []
    report = precompute.execute(
        conn, plan, log=lines.append, workers=workers, modules=TOYS
    )
    return report, lines


def stale(conn, plots, workers=1, **kwargs):
    specs = precompute.in_dependency_order([registry.get(key) for key in plots])
    plan = precompute.plan_stale(conn, specs, "cmod", **kwargs)
    lines = []
    report = precompute.execute(
        conn, plan, log=lines.append, workers=workers, modules=TOYS
    )
    return plan, report, lines


def runs(conn, plot=None):
    query, args = "SELECT * FROM runs", ()
    if plot is not None:
        query, args = query + " WHERE plot = ?", (plot,)
    return {
        (r["plot"], r["shot"]): dict(r)
        for r in conn.execute(query + " ORDER BY id", args)
    }


def counts(stats):
    return (stats.computed, stats.cached, stats.failed, stats.skipped, stats.cancelled)


def leftovers(cache):
    """Temporary files ``store._write_blob`` writes beside a blob."""
    return glob.glob(str(cache / "**" / "*.tmp.*"), recursive=True)


def rewrite(tree, shot, seconds=1_700_000_000):
    """The file changed on disk, and a rescan saw it."""
    os.utime(tree.folder / f"apd_{shot}.nc", (seconds, seconds))
    catalog.rescan(tree.conn, str(tree.root / "alcator"), "cmod", str(tree.discharges))


def blobs(cache, plot):
    return glob.glob(str(cache / "runs" / plot / "**" / "*.nc*"), recursive=True)


# ---------------------------------------------------------------------------
# A pool of workers
# ---------------------------------------------------------------------------


def test_two_workers_fill_four_targets_then_skip_them_all_as_cached(
    tree, capfd, monkeypatch
):
    contexts = []
    real_context = precompute.multiprocessing.get_context

    def spy(method=None):
        contexts.append(method)
        return real_context(method)

    monkeypatch.setattr(precompute.multiprocessing, "get_context", spy)
    report, lines = fill(tree.conn, ["toy_mean"], workers=2)
    # Spawned, not forked: no SQLite handle or thread is inherited.
    assert contexts == ["spawn"]
    (stats,) = report.stats
    assert counts(stats) == (4, 0, 0, 0, 0)

    done = runs(tree.conn)
    assert len(done) == 4 and all(r["status"] == "ok" for r in done.values())
    for row in done.values():
        assert os.path.exists(row["blob_path"])
        with xr.open_dataset(row["blob_path"]) as blob:
            attrs = dict(blob.attrs)
        # Computed in a worker, on the windowed record loaded into memory.
        assert attrs["pid"] != os.getpid()
        assert attrs["in_memory"] == 1
        if row["shot"] == 1160616001:  # the descriptor's window
            assert 0 < attrs["samples"] < SHOTS[row["shot"]]
        else:
            assert attrs["samples"] == SHOTS[row["shot"]]
    scalars = tree.conn.execute("SELECT COUNT(*) FROM scalars").fetchone()[0]
    assert scalars == 4 * 2

    out = capfd.readouterr().out
    worked = [line for line in out.splitlines() if ": ok in " in line]
    assert len(worked) == 4
    # Each worker prints its own lines, tagged, and stamped with the time.
    assert all(" w1 [" in line or " w2 [" in line for line in worked)
    assert all(line[:4].isdigit() for line in worked)
    assert any("4 of 4 targets to compute on 2 workers" in line for line in lines)

    # A second run skips them all from the ledger: no file opened, no pool.
    def no_open(*args, **kwargs):
        raise AssertionError("a cache hit must not open its file")

    def no_pool(*args, **kwargs):
        raise AssertionError("nothing to compute must start no workers")

    monkeypatch.setattr(precompute.xr, "open_dataset", no_open)
    monkeypatch.setattr(precompute.concurrent.futures, "ProcessPoolExecutor", no_pool)
    report, lines = fill(tree.conn, ["toy_mean"], workers=2)
    assert counts(report.stats[0]) == (0, 4, 0, 0, 0)
    assert sum("cached, skipping" in line for line in lines) == 4


def test_a_pool_takes_the_largest_files_first():
    def job(shot, size):
        target = registry.Target("cmod", shot, "apd", False, "", 0.0, 0.0)
        return precompute.Job(target=target, index=0, steps=(), size=size)

    jobs = [job(1, 10), job(2, 40), job(3, 10), job(4, 30)]
    assert [j.target.shot for j in precompute.largest_first(jobs)] == [2, 4, 1, 3]


def test_a_chain_runs_in_dependency_order_one_worker_per_target(tree, capfd):
    """Named in any order, the bank runs first in each target, once; both
    products are built on that one run. The bank is batch only, so this also
    checks that the workers compute as a batch job."""
    report, _ = fill(tree.conn, ["toy_blobs", "toy_bank", "toy_fields"], workers=2)
    assert [s.plot for s in report.stats] == ["toy_bank", "toy_blobs", "toy_fields"]
    assert all(counts(s) == (4, 0, 0, 0, 0) for s in report.stats)

    done = runs(tree.conn)
    assert len(done) == 12 and all(r["status"] == "ok" for r in done.values())
    for shot in SHOTS:
        bank = done[("toy_bank", shot)]
        assert done[("toy_fields", shot)]["upstream_run_id"] == bank["id"]
        assert done[("toy_blobs", shot)]["upstream_run_id"] == bank["id"]

    out = capfd.readouterr().out.splitlines()
    for shot in SHOTS:
        mine = [line for line in out if f" {shot} · " in line]
        workers = {line.split()[2] for line in mine}
        assert len(workers) == 1, f"one worker handles {shot} end to end"
        order = [line.split(" · ")[2].split(":")[0] for line in mine if "ok in" in line]
        assert order == ["toy_bank", "toy_blobs", "toy_fields"]


def test_a_worker_connection_waits_out_the_other_writers(tmp_path):
    conn = precompute._worker_connection(tmp_path / "state" / "db.sqlite")
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30_000
    finally:
        conn.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_an_infrastructure_error_in_a_worker_stays_unrecorded(tree, capfd):
    unreadable = tree.folder / "apd_1140827002.nc"
    unreadable.chmod(0)
    try:
        report, _ = fill(tree.conn, ["toy_mean"], workers=2)
    finally:
        unreadable.chmod(0o644)
    assert counts(report.stats[0]) == (3, 0, 1, 0, 0)
    assert ("toy_mean", 1140827002) not in runs(tree.conn)
    assert "failed without recording a run: PermissionError" in capfd.readouterr().out

    # Fixed: the next fill computes it, rather than skipping a failure.
    report, _ = fill(tree.conn, ["toy_mean"])
    assert counts(report.stats[0]) == (1, 3, 0, 0, 0)


def test_an_unwritable_upstream_directory_stops_the_chain_before_it_computes(
    tree, monkeypatch
):
    """A bank computed on the way must not end in a PermissionError, and a
    `failed` row, on its save: its directory is checked too."""
    bank = registry.get("toy_bank")
    digest, _ = params_ui.hash_params("toy_bank", bank.params())
    real = precompute._output_writable

    def bank_unwritable(spec, target, params_hash):
        if spec.key == "toy_bank" and params_hash == digest:
            return False, "/nowhere/toy_bank"
        return real(spec, target, params_hash)

    monkeypatch.setattr(precompute, "_output_writable", bank_unwritable)
    report, lines = fill(tree.conn, ["toy_fields"], shots={1160616001})
    assert counts(report.stats[0]) == (0, 0, 1, 0, 0)
    assert runs(tree.conn) == {}
    assert any("/nowhere/toy_bank is not writable" in line for line in lines)


def test_a_dead_worker_stops_the_fill_without_recording_anything(tree):
    report, lines = fill(tree.conn, ["toy_crash"], workers=2)
    assert report.broken
    assert report.stats[0].failed == 0 and report.stats[0].computed == 0
    assert runs(tree.conn) == {}
    assert "a worker died" in report.summary_lines()[-1]


# ---------------------------------------------------------------------------
# Ctrl-C
# ---------------------------------------------------------------------------


def test_ctrl_c_in_a_worker_leaves_no_failed_row_no_blob_and_no_temporary_file(
    tree, capfd
):
    report, _ = fill(tree.conn, ["toy_interrupt"], workers=2)
    assert report.interrupted
    assert counts(report.stats[0])[:3] == (0, 0, 0)
    assert report.stats[0].cancelled == 4
    assert runs(tree.conn) == {}
    assert blobs(tree.cache, "toy_interrupt") == []
    assert leftovers(tree.cache) == []
    assert "interrupted, nothing recorded" in capfd.readouterr().out


@pytest.mark.parametrize("workers", [1, 2])
def test_ctrl_c_while_the_store_writes_waits_until_the_result_is_whole(tree, workers):
    """SIGINT between the run row's commit and its scalars' must not leave an
    `ok` row without them: the write finishes, then the fill stops."""
    report, _ = fill(tree.conn, ["toy_interrupt_on_write"], workers=workers)
    assert report.interrupted
    done = runs(tree.conn)
    assert done, "the result being written when Ctrl-C came is kept"
    for row in done.values():
        assert row["status"] == "ok" and os.path.exists(row["blob_path"])
        names = {
            r["name"]
            for r in tree.conn.execute(
                "SELECT name FROM scalars WHERE run_id = ?", (row["id"],)
            )
        }
        assert names == {"mean", "pixel_mean"}
    assert report.stats[0].computed == len(done)
    assert report.stats[0].cancelled == 4 - len(done)
    assert leftovers(tree.cache) == []


def test_ctrl_c_in_one_process_stops_the_fill_where_it_is(tree, capsys):
    """One worker, in this process: the handler is put back afterwards."""
    before = signal.getsignal(signal.SIGINT)
    report, lines = fill(tree.conn, ["toy_interrupt"])
    assert signal.getsignal(signal.SIGINT) is before
    assert report.interrupted and report.stats[0].cancelled == 4
    assert runs(tree.conn) == {}
    assert sum("interrupted, nothing recorded" in line for line in lines) == 1
    assert leftovers(tree.cache) == []


def test_an_ignored_sigint_stays_ignored():
    """A fill started in the background of a script inherits SIGINT ignored,
    and must stay deaf to the terminal's Ctrl-C."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        interrupts = precompute._Interrupts().install()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        interrupts.restore()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def drive(tree, *argv, preexec_fn=None, start_new_session=False, **env):
    """Start the CLI, with the toys registered, in a process of its own."""
    return subprocess.Popen(
        [sys.executable, "-c", toys.DRIVER, *argv],
        cwd=REPO,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=preexec_fn,
        start_new_session=start_new_session,
    )


def wait_for(process, markers, count, seconds=120):
    """Until ``count`` marker files exist, failing if the process ended first."""
    deadline = time.monotonic() + seconds
    while len(os.listdir(markers)) < count and time.monotonic() < deadline:
        assert process.poll() is None, process.communicate()[0]
        time.sleep(0.05)
    assert len(os.listdir(markers)) >= count, sorted(os.listdir(markers))


def test_ctrl_c_from_the_terminal_stops_every_worker_cleanly(tree, tmp_path):
    """SIGINT to the whole process group, as a terminal sends it."""
    markers = tmp_path / "markers"
    markers.mkdir()
    process = drive(
        tree,
        "precompute",
        "toy_block",
        "--workers",
        "2",
        start_new_session=True,
        **{toys.MARKERS: str(markers)},
    )
    wait_for(process, markers, 2)  # both workers are computing

    os.killpg(process.pid, signal.SIGINT)
    out, _ = process.communicate(timeout=120)
    assert process.returncode == 130, out
    assert "interrupted" in out
    # The two targets in flight stopped, and the two pending never started.
    assert len(os.listdir(markers)) == 2
    assert runs(tree.conn) == {}
    assert blobs(tree.cache, "toy_block") == []
    assert leftovers(tree.cache) == []


def test_a_second_ctrl_c_kills_workers_that_do_not_stop(tree, tmp_path):
    """A compute deaf to the first Ctrl-C (a long call into C) is killed by
    the second, and still nothing is recorded."""
    markers = tmp_path / "markers"
    markers.mkdir()
    process = drive(
        tree,
        "precompute",
        "toy_stubborn",
        "--workers",
        "2",
        start_new_session=True,
        **{toys.MARKERS: str(markers)},
    )
    wait_for(process, markers, 2)
    os.killpg(process.pid, signal.SIGINT)
    wait_for(process, markers, 4)  # each worker noted the interrupt, and went on
    time.sleep(1)  # the parent is waiting for them now
    os.killpg(process.pid, signal.SIGINT)
    out, _ = process.communicate(timeout=120)
    assert process.returncode == 130, out
    assert "killed the workers" in out and "killed:" in out
    assert runs(tree.conn) == {}
    assert blobs(tree.cache, "toy_stubborn") == []
    assert leftovers(tree.cache) == []


def test_nice_is_set_never_added_to_what_the_workers_inherit(tree):
    """`nice -n 3 fusion-ui precompute … --workers 2 --nice 5` computes at 5,
    not 8 -- on every thread, BLAS's included. (The default of 10 with
    several workers is `test_the_nice_default_follows_the_worker_count`.)"""
    process = drive(
        tree,
        "precompute",
        "toy_mean",
        "--workers",
        "2",
        "--nice",
        "5",
        preexec_fn=lambda: os.nice(3),
    )
    out, _ = process.communicate(timeout=180)
    assert process.returncode == 0, out
    assert "at nice 5" in out
    seen = set()
    for row in runs(tree.conn).values():
        with xr.open_dataset(row["blob_path"]) as blob:
            assert blob.attrs["pid"] != process.pid  # a worker computed it
            seen.add((blob.attrs["niceness"], blob.attrs["least_thread_niceness"]))
    assert seen == {(5, 5)}


# ---------------------------------------------------------------------------
# --stale
# ---------------------------------------------------------------------------


def test_stale_picks_only_the_targets_whose_input_changed(tree):
    fill(tree.conn, ["toy_mean"])
    before = runs(tree.conn)
    rewrite(tree, 1160616002)
    rewrite(tree, 1140827001, seconds=1_700_000_600)

    plan, report, lines = stale(tree.conn, ["toy_mean"], workers=2)
    assert sorted(job.target.shot for job in plan.jobs) == [1140827001, 1160616002]
    assert counts(report.stats[0]) == (2, 0, 0, 0, 0)

    after = runs(tree.conn)
    index = {
        row["shot"]: row["mtime"]
        for row in tree.conn.execute("SELECT shot, mtime FROM shots")
    }
    for key, row in after.items():
        assert row["id"] == before[key]["id"], "recomputed in place"
        changed = key[1] in (1140827001, 1160616002)
        assert (row["created_at"] != before[key]["created_at"]) == changed
        assert row["input_mtime"] == index[key[1]]
    assert store.stale_runs(tree.conn) == []

    _, report, _ = stale(tree.conn, ["toy_mean"])
    assert report.stats[0].considered == 0


def test_stale_leaves_a_result_whose_upstream_is_stale_and_says_what_fixes_it(
    tree, capsys
):
    fill(tree.conn, ["toy_fields"])
    before = runs(tree.conn)
    rewrite(tree, 1160616002)

    plan, report, lines = stale(tree.conn, ["toy_fields"])
    assert counts(report.stats[0]) == (0, 0, 0, 1, 0)
    assert plan.unnamed_upstreams == ["toy_bank"]
    (line,) = [line for line in lines if "not recomputed" in line]
    assert "upstream toy_bank (input changed)" in line
    assert (
        "`fusion-ui precompute toy_bank toy_fields --stale --shot 1160616002`" in line
    )
    assert runs(tree.conn) == before

    # The CLI ends with the one command for every shot it left.
    assert cli.main(["precompute", "toy_fields", "--stale"]) == 0
    out = capsys.readouterr().out
    assert "run `fusion-ui precompute toy_bank toy_fields --stale`" in out

    # Named together, the chain is recomputed upstream first, and is current.
    plan, report, _ = stale(tree.conn, ["toy_fields", "toy_bank"], workers=2)
    assert [s.plot for s in report.stats] == ["toy_bank", "toy_fields"]
    assert all(counts(s) == (1, 0, 0, 0, 0) for s in report.stats)
    assert store.stale_runs(tree.conn) == []


def test_stale_after_a_forced_upstream_recomputes_the_products_on_it(tree):
    """The plan's recompute for a changed 2DCA: `precompute pixel_averages
    --force`, then `precompute method_fields blob_parameters --stale`."""
    fill(tree.conn, ["toy_fields", "toy_blobs"])
    fill(tree.conn, ["toy_bank"], force=True)
    reasons = {(r["plot"], r["stale"]) for r in store.stale_runs(tree.conn)}
    assert reasons == {
        ("toy_fields", store.STALE_UPSTREAM_DELETED),
        ("toy_blobs", store.STALE_UPSTREAM_DELETED),
    }
    banks = runs(tree.conn, "toy_bank")

    plan, report, _ = stale(tree.conn, ["toy_fields", "toy_blobs"])
    assert all(counts(s) == (4, 0, 0, 0, 0) for s in report.stats)
    assert store.stale_runs(tree.conn) == []
    # The banks were reused, not recomputed.
    assert runs(tree.conn, "toy_bank") == banks


def test_stale_recomputes_every_parameter_set_with_its_own_parameters(tree, tmp_path):
    fill(tree.conn, ["toy_bank"], shots={1160616001})
    changes = tmp_path / "short.json"
    changes.write_text(json.dumps({"averages": {"window": 2}}))
    argv = ["toy_bank", "--shot", "1160616001", "--params-json", str(changes)]
    assert cli.main(["precompute", *argv]) == 0
    before = {
        r["params_hash"]: dict(r)
        for r in tree.conn.execute("SELECT * FROM runs WHERE plot = 'toy_bank'")
    }
    assert len(before) == 2
    rewrite(tree, 1160616001)

    plan, report, lines = stale(tree.conn, ["toy_bank"])
    assert counts(report.stats[0]) == (2, 0, 0, 0, 0)
    assert plan.options.hashed == frozenset({"toy_bank"})  # lines name the set
    after = {
        r["params_hash"]: dict(r)
        for r in tree.conn.execute("SELECT * FROM runs WHERE plot = 'toy_bank'")
    }
    assert set(after) == set(before)
    for digest, row in after.items():
        with xr.open_dataset(row["blob_path"]) as blob:
            assert blob.attrs["fusion_ui_params_hash"] == digest

    # --params-json picks one of them.
    rewrite(tree, 1160616001, seconds=1_700_000_600)
    specs = [registry.get("toy_bank")]
    only = precompute.params_from_file(changes, specs)
    plan = precompute.plan_stale(tree.conn, specs, "cmod", only=only)
    (job,) = plan.jobs
    (step,) = job.steps
    assert step.params_hash == params_ui.hash_params("toy_bank", only["toy_bank"])[0]


def test_stale_and_force_or_retry_failed_do_not_mix(tree, capsys):
    assert cli.main(["precompute", "toy_mean", "--stale", "--force"]) == 1
    assert cli.main(["precompute", "toy_mean", "--stale", "--retry-failed"]) == 1
    assert "--stale" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --params-json
# ---------------------------------------------------------------------------


def test_the_page_s_command_fills_the_result_the_page_looks_up(
    tree, tmp_path, capsys, monkeypatch
):
    """J2a's page shows `precompute.command` and the parameter set to save as
    params.json; run as written, it must store what `store.lookup` finds."""
    spec = registry.get("toy_bank")
    params = toys.BankParams(averages=toys.Averages(window=2, threshold=3.0))
    (target,) = precompute.select_targets(tree.conn, [spec], shots={1160616002})
    line = precompute.command(spec, target, params)
    assert line.endswith("--params-json params.json")
    shown = params_ui.hash_params(spec.key, params)[1]
    (tmp_path / precompute.PARAMS_FILE).write_text(shown)

    monkeypatch.chdir(tmp_path)
    assert cli.main(line.split()[1:]) == 0, capsys.readouterr()
    result, run = store.lookup(tree.conn, spec, target, params)
    assert result is not None and run["status"] == "ok"
    assert (
        result.attrs["fusion_ui_params_hash"]
        == params_ui.hash_params(spec.key, params)[0]
    )


def test_one_file_of_changes_applies_to_every_plot_named(tree, tmp_path):
    """The plan's short window for one shot: `precompute method_fields
    blob_parameters --shot N --params-json short.json`, two params classes
    sharing `averages`. One bank, built once, under the shorter window."""
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"averages": {"window": 2}}))
    code = cli.main(
        [
            "precompute",
            "toy_fields",
            "toy_blobs",
            "--shot",
            "1160616002",
            "--params-json",
            str(short),
        ]
    )
    assert code == 0

    averages = toys.Averages(window=2)
    expected = {
        "toy_fields": toys.FieldsParams(averages=averages),
        "toy_blobs": toys.BlobsParams(averages=averages),
        "toy_bank": toys.BankParams(averages=averages),
    }
    done = {r["plot"]: r for r in runs(tree.conn).values()}
    assert set(done) == set(expected)
    (target,) = precompute.select_targets(
        tree.conn, [registry.get("toy_bank")], shots={1160616002}
    )
    for plot, params in expected.items():
        assert done[plot]["params_hash"] == params_ui.hash_params(plot, params)[0]
        result, _ = store.lookup(tree.conn, registry.get(plot), target, params)
        assert result is not None
    assert done["toy_fields"]["upstream_run_id"] == done["toy_bank"]["id"]
    assert done["toy_blobs"]["upstream_run_id"] == done["toy_bank"]["id"]


def payload(plot, params):
    return params_ui.hash_params(plot, params)[1]


@pytest.mark.parametrize(
    "argv, body, message",
    [
        (
            ["toy_fields", "toy_blobs"],
            payload("toy_fields", toys.FieldsParams()),
            "says nothing about the other plots named",
        ),
        (
            ["toy_fields"],
            payload("toy_bank", toys.BankParams()),
            "is the parameter set of plot 'toy_bank'; the plot named is 'toy_fields'",
        ),
        (
            ["toy_fields", "toy_blobs"],
            json.dumps({"tracking": {"step": 2}}),
            "does not fit toy_blobs",
        ),
        (["toy_fields"], json.dumps({"averages": {"windw": 2}}), "no field 'windw'"),
        (
            ["toy_fields"],
            json.dumps({"averages": {"single_counting": "false"}}),
            "expected true or false",
        ),
        (["toy_fields"], json.dumps({"averages": {"window": 2.5}}), "whole number"),
        (
            ["toy_mean", "--pixel", "1", "2"],
            payload("toy_mean", toys.ToyParams()),
            "--pixel",
        ),
        (["toy_mean", "--pixel", "1", "2"], json.dumps({"refx": 3}), "--pixel"),
        (["toy_mean"], "{not json", "not valid JSON"),
        (["toy_mean"], "[1, 2]", "expected a JSON object"),
    ],
)
def test_a_params_file_that_does_not_fit_stops_before_anything_runs(
    tree, tmp_path, capsys, argv, body, message
):
    path = tmp_path / "params.json"
    path.write_text(body)
    assert cli.main(["precompute", *argv, "--params-json", str(path)]) == 1
    assert message in capsys.readouterr().err
    assert tree.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert tree.conn.execute("SELECT COUNT(*) FROM param_sets").fetchone()[0] == 0


def test_changes_combine_with_pixel(tree, tmp_path):
    path = tmp_path / "scale.json"
    path.write_text(json.dumps({"scale": 2.0}))
    (spec,) = [registry.get("toy_mean")]
    got = precompute.params_from_file(path, [spec], pixel=(1, 2))["toy_mean"]
    assert (got.scale, got.refx, got.refy) == (2.0, 1, 2)


# ---------------------------------------------------------------------------
# Selecting targets
# ---------------------------------------------------------------------------


def test_run_days_and_shots_add_up(tree):
    spec = registry.get("toy_mean")

    def shots(**selection):
        return [
            t.shot
            for t in precompute.select_targets(tree.conn, [spec], "cmod", **selection)
        ]

    assert shots(run_days={1160616}) == [1160616001, 1160616002]
    assert shots(run_days={1160616}, shots={1140827002}) == [
        1140827002,
        1160616001,
        1160616002,
    ]
    assert shots(run_days={1110201}) == []


@pytest.mark.parametrize("day", ["116061", "11606160", "1160616a"])
def test_a_run_day_is_seven_digits(day, capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["precompute", "toy_mean", "--run-day", day])
    assert "not a run day" in capsys.readouterr().err


def test_the_nice_default_follows_the_worker_count(tree, monkeypatch):
    seen = []
    real = precompute.execute

    def spy(conn, plan, log=None, workers=1, nice=None, modules=None):
        seen.append((workers, nice))
        return real(conn, plan, log=log, workers=1, nice=None, modules=modules)

    monkeypatch.setattr(precompute, "execute", spy)
    assert cli.main(["precompute", "toy_mean", "--shot", "1160616001"]) == 0
    assert cli.main(["precompute", "toy_mean", "--workers", "3"]) == 0
    assert cli.main(["precompute", "toy_mean", "--workers", "3", "--nice", "4"]) == 0
    assert seen == [(1, None), (3, precompute.DEFAULT_NICE), (3, 4)]


# ---------------------------------------------------------------------------
# One worker: the output a fill has always printed
# ---------------------------------------------------------------------------


def _normalised(text):
    import re

    text = re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", "<time> ", text, flags=re.M)
    text = re.sub(r"\d+\.\ds\b", "<s>", text)
    text = re.sub(r"(ValueError): .*", r"\1: <message>", text)
    return [
        line for line in text.splitlines() if "MemoryCacheStorageManager" not in line
    ]


def test_the_single_worker_output_is_unchanged(tree, capsys):
    """Line for line what `fusion-ui precompute PLOT` printed before phase 06
    (checked against main at faeaede over a longer scenario when J2b landed)."""
    digest = params_ui.hash_params("toy_mean", toys.ToyParams())[0][:12]
    labels = {shot: f"cmod {shot} · apd (raw)" for shot in sorted(SHOTS)}
    order = sorted(SHOTS)

    assert cli.main(["precompute", "toy_mean"]) == 0
    expected = [
        f"<time> toy_mean: 4 targets, params {digest}, force=False retry_failed=False"
    ]
    for i, shot in enumerate(order, start=1):
        expected += [
            f"<time> [{i}/4] {labels[shot]}: computing…",
            f"<time> [{i}/4] {labels[shot]}: ok in <s>",
        ]
    expected.append("toy_mean: 4 shots, 4 computed, 0 cached in <s>")
    assert _normalised(capsys.readouterr().out) == expected

    (tree.folder / "apd_1160616002.nc").write_bytes(b"not a netCDF file")
    assert cli.main(["precompute", "toy_mean", "--force", "--shot", "1160616002"]) == 0
    assert _normalised(capsys.readouterr().out) == [
        f"<time> toy_mean: 1 targets, params {digest}, force=True retry_failed=False",
        f"<time> [1/1] {labels[1160616002]}: computing…",
        f"<time> [1/1] {labels[1160616002]}: failed after <s>: ValueError: <message>",
        "toy_mean: 1 shots, 0 computed, 0 cached, 1 failed in <s>",
    ]

    assert cli.main(["precompute", "toy_mean"]) == 0
    expected = [
        f"<time> toy_mean: 4 targets, params {digest}, force=False retry_failed=False"
    ]
    for i, shot in enumerate(order, start=1):
        if shot == 1160616002:
            expected.append(
                f"<time> [{i}/4] {labels[shot]}: previous failure recorded, skipping"
                " (--force or --retry-failed to retry): ValueError: <message>"
            )
        else:
            expected.append(f"<time> [{i}/4] {labels[shot]}: cached, skipping")
    expected.append("toy_mean: 4 shots, 0 computed, 3 cached, 1 failed in <s>")
    assert _normalised(capsys.readouterr().out) == expected
