"""``fusion-ui prune``: one plot's results counted, then deleted, and nothing else touched.

The ledger is built the way the app builds it -- ``store.result`` on synthetic specs,
so every blob is a real netCDF under a real hash path and every run has its scalars
and its parameter set -- and the command is driven through ``cli.main`` on that file.
Nothing here reads the real ledger or the data tree.

The last section is ``--scalar``: only the scalars of some names under one plot key,
counted per name and then deleted, with the runs, the blobs, the other scalars and the
same names under other plots left as they are. Its ledger is the one the seed's three
mislabelled names were pruned from: an earlier ``import-results`` wrote all fifteen.
"""

import dataclasses
import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from fusion_ui import cli
from fusion_ui.core import catalog, params_ui, registry, seed, store

SHOTS = (1160616027, 1160616028, 1160616029)


@dataclasses.dataclass
class Params:
    gain: float = 1.0


@dataclasses.dataclass
class DerivedParams:
    base: Params = dataclasses.field(default_factory=Params)
    scale: float = 2.0


def make_spec(key):
    """A cached spec that writes a blob and two scalars, one of them at a pixel."""
    return registry.PlotSpec(
        key=key,
        label=key,
        diagnostics=("apd",),
        params=Params,
        render=lambda result, params, target: None,
        compute=lambda ds, params: xr.Dataset(
            {"y": ("t", np.arange(4.0) * params.gain)}
        ),
        scalars=lambda result: {
            "total": float(result["y"].sum()),
            (3, 4, "corner"): float(result["y"][-1]),
        },
    )


def make_derived(key, upstream):
    return registry.PlotSpec(
        key=key,
        label=key,
        diagnostics=("apd",),
        params=DerivedParams,
        render=lambda result, params, target: None,
        compute=lambda ds, params, bank: xr.Dataset({"y": bank["y"] * params.scale}),
        scalars=lambda result: {"total": float(result["y"].sum())},
        requires=upstream,
        upstream_params=lambda params: params.base,
    )


def failing(spec):
    def boom(ds, params):
        raise ValueError("no events found")

    return dataclasses.replace(spec, compute=boom)


def target(shot=SHOTS[0]):
    return registry.Target(
        machine="cmod",
        shot=shot,
        diagnostic="apd",
        preprocessed=True,
        path="/nowhere/apd.nc",
        t_start=1.15,
        t_end=1.45,
    )


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def database(tmp_path):
    """The file the ``conn`` fixture opened, for the command to open as well."""
    return tmp_path / "state" / "shot_explorer.sqlite"


@pytest.fixture
def registered():
    """Put specs in the global registry for the duration of one test."""
    added = []

    def add(spec):
        registry.REGISTRY[spec.key] = spec
        added.append(spec.key)
        return spec

    yield add
    for key in added:
        registry.REGISTRY.pop(key, None)


@pytest.fixture
def ledger(conn, cache):
    """Two plots that know nothing of each other.

    ``doomed``, the one the tests prune: shots 0 and 1 at the default parameters, shot 0
    again at another set, a run that failed (no blob), and a parameter set that was asked
    for and never run. Three blobs, six scalar rows, three parameter sets.
    ``keeper``: shots 0 and 1, to be left exactly as they are.
    """
    doomed, keeper = make_spec("doomed"), make_spec("keeper")
    for shot in SHOTS[:2]:
        store.result(conn, doomed, target(shot), Params(), ds=None)
        store.result(conn, keeper, target(shot), Params(), ds=None)
    store.result(conn, doomed, target(SHOTS[0]), Params(gain=2.0), ds=None)
    store.result(conn, failing(doomed), target(SHOTS[2]), Params(), ds=None)
    store.record_params(conn, "doomed", Params(gain=9.0))
    return SimpleNamespace(doomed=doomed, keeper=keeper)


def snapshot(conn, cache, outside=None):
    """Rows, scalars, parameter sets and blob files as plain values.

    All of them, or all but ``outside``'s: what a prune of that plot must leave as it was.
    """

    def keep(plot):
        return outside is None or plot != outside

    runs = [
        tuple(row)
        for row in conn.execute("SELECT * FROM runs ORDER BY id")
        if keep(row["plot"])
    ]
    ids = {row[0] for row in runs}
    scalars = [
        tuple(row)
        for row in conn.execute("SELECT * FROM scalars ORDER BY run_id, x, y, name")
        if row["run_id"] in ids
    ]
    params = [
        tuple(row)
        for row in conn.execute("SELECT * FROM param_sets ORDER BY hash")
        if keep(row["plot"])
    ]
    blobs = sorted(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in cache.rglob("*")
        if path.is_file() and keep(path.relative_to(cache).parts[1])
    )
    return {"runs": runs, "scalars": scalars, "params": params, "blobs": blobs}


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def blobs_of(conn, plot):
    return [
        row["blob_path"]
        for row in conn.execute(
            "SELECT blob_path FROM runs WHERE plot = ? AND blob_path IS NOT NULL"
            " ORDER BY id",
            (plot,),
        )
    ]


def prune(database, *args):
    return cli.main(["--database", str(database), "prune", *args])


def say(capsys):
    out = capsys.readouterr()
    return out.out, out.err


# ---------------------------------------------------------------------------
# What a prune takes, and what it leaves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "still_registered", [True, False], ids=["registered", "no longer registered"]
)
def test_a_pruned_plot_goes_whole_and_every_other_plot_is_untouched(
    conn, cache, database, ledger, registered, capsys, still_registered
):
    if still_registered:
        registered(ledger.doomed)
    assert (ledger.doomed.key in registry.REGISTRY) is still_registered
    blobs = blobs_of(conn, "doomed")
    assert len(blobs) == 3 and all(os.path.exists(path) for path in blobs)
    others = snapshot(conn, cache, outside="doomed")
    assert others["runs"] and others["scalars"] and others["blobs"] and others["params"]
    # The scalars go by the cascade, which only a connection with foreign keys on gives.
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    assert prune(database, "--plot", "doomed", "--yes") == 0

    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = 'doomed'") == 0
    assert count(conn, "SELECT COUNT(*) FROM param_sets WHERE plot = 'doomed'") == 0
    assert not any(os.path.exists(path) for path in blobs)
    assert (
        count(
            conn,
            "SELECT COUNT(*) FROM scalars WHERE run_id NOT IN (SELECT id FROM runs)",
        )
        == 0
    )
    assert count(conn, "SELECT COUNT(*) FROM scalars") == len(others["scalars"])
    assert snapshot(conn, cache) == snapshot(conn, cache, outside="doomed")
    assert snapshot(conn, cache) == others
    out, err = say(capsys)
    assert "deleted       4 runs, 6 scalar rows, 3 blob files, 3 param sets" in out
    assert err == ""

    # Asked again there is nothing left, and the command says so rather than succeed.
    assert prune(database, "--plot", "doomed", "--yes") == 1
    assert "nothing to prune" in say(capsys)[1]


def test_without_yes_it_counts_and_deletes_nothing(
    conn, cache, database, ledger, capsys
):
    before = snapshot(conn, cache)

    assert prune(database, "--plot", "doomed") == 1

    assert snapshot(conn, cache) == before
    out, err = say(capsys)
    assert out.splitlines() == [
        "plot          doomed",
        "runs          4  (failed 1, ok 3)",
        "scalar rows   6",
        "blob files    3 on disk, 0 listed but already missing",
        "param sets    3 that nothing will reference any more",
        "built on them no run of another plot",
    ]
    assert "Nothing was deleted" in err and "--yes" in err


def test_the_command_cannot_be_run_without_a_plot(database, capsys):
    with pytest.raises(SystemExit) as exit_:
        prune(database)
    assert exit_.value.code == 2
    assert "--plot" in say(capsys)[1]


def test_a_key_the_ledger_has_no_runs_of_says_which_it_has(
    conn, cache, database, ledger, capsys
):
    before = snapshot(conn, cache)
    assert prune(database, "--plot", "doomd", "--yes") == 1
    out, err = say(capsys)
    assert out == ""
    assert "'doomd'" in err and "doomed (4)" in err and "keeper (2)" in err
    assert snapshot(conn, cache) == before


def test_a_parameter_set_a_preset_names_is_kept(conn, cache, database, ledger, capsys):
    """A preset is a person's saved choice, not a result: it is not pruned, and the
    parameter set it names must survive it (the foreign key would refuse otherwise)."""
    named = store.record_params(conn, "doomed", Params(gain=2.0))[0]
    with conn:
        conn.execute(
            "INSERT INTO presets (name, plot, params_hash, note, created_at)"
            " VALUES ('slow', 'doomed', ?, NULL, 'now')",
            (named,),
        )

    assert prune(database, "--plot", "doomed", "--yes") == 0

    assert [
        row[0]
        for row in conn.execute("SELECT hash FROM param_sets WHERE plot = 'doomed'")
    ] == [named]
    assert count(conn, "SELECT COUNT(*) FROM presets") == 1
    assert "3 blob files, 2 param sets" in say(capsys)[0]


# ---------------------------------------------------------------------------
# Blobs: missing ones are reported, outside ones refused, unremovable ones kept
# ---------------------------------------------------------------------------


def test_a_blob_that_is_already_missing_is_counted_and_reported_not_fatal(
    conn, cache, database, ledger, capsys
):
    os.remove(blobs_of(conn, "doomed")[0])

    assert prune(database, "--plot", "doomed") == 1
    out, _ = say(capsys)
    assert "blob files    2 on disk, 1 listed but already missing" in out

    assert prune(database, "--plot", "doomed", "--yes") == 0
    out, err = say(capsys)
    assert "deleted       4 runs, 6 scalar rows, 2 blob files, 3 param sets" in out
    assert "already gone  1 blob files" in out
    assert err == ""
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = 'doomed'") == 0


@pytest.fixture
def elsewhere(tmp_path):
    """A file next to the cache, not in it: what a prune must never reach."""
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    precious = folder / "precious.nc"
    precious.write_bytes(b"not a cache blob")
    return precious


def point_a_run_at(conn, path):
    """Corrupt the ledger the way a copied database or a wrong setting would."""
    with conn:
        conn.execute(
            "UPDATE runs SET blob_path = ? WHERE id = (SELECT MIN(id) FROM runs"
            " WHERE plot = 'doomed' AND blob_path IS NOT NULL)",
            (str(path),),
        )


@pytest.mark.parametrize("how", ["absolute", "dotdot", "symlinked directory"])
@pytest.mark.parametrize("yes", [[], ["--yes"]], ids=["counting", "--yes"])
def test_a_blob_outside_the_cache_is_refused_and_nothing_is_deleted(
    conn, cache, database, ledger, elsewhere, capsys, how, yes
):
    doomed_dir = cache / "runs" / "doomed"
    if how == "absolute":
        listed = elsewhere
    elif how == "dotdot":
        # Textually under the cache, so a prefix test would let it through.
        listed = doomed_dir / os.path.relpath(elsewhere, doomed_dir)
        assert ".." in listed.parts and str(listed).startswith(str(cache))
    else:
        (doomed_dir / "link").symlink_to(elsewhere.parent)
        listed = doomed_dir / "link" / elsewhere.name
    point_a_run_at(conn, listed)
    before = snapshot(conn, cache)

    assert prune(database, "--plot", "doomed", *yes) == 1

    out, err = say(capsys)
    assert "Refusing to prune 'doomed'" in err and str(listed) in err
    assert "outside cache 1 blob paths: refused" in out
    assert elsewhere.read_bytes() == b"not a cache blob"
    # Refused whole: not one other blob, row, scalar or parameter set of the plot went.
    assert snapshot(conn, cache) == before


def test_a_symlink_in_the_cache_is_removed_and_what_it_points_at_is_not(
    conn, cache, database, ledger, elsewhere
):
    """The last component is not followed: the link is the cache's to remove."""
    first = blobs_of(conn, "doomed")[0]
    os.remove(first)
    os.symlink(elsewhere, first)

    assert prune(database, "--plot", "doomed", "--yes") == 0

    assert not os.path.lexists(first)
    assert elsewhere.read_bytes() == b"not a cache blob"


@pytest.fixture
def locked(conn, cache, ledger):
    """The directory of the gain-2 blob made unwritable, as the other writer's would be.

    Removing a file needs write permission on its directory, so that is what to take away.
    """
    slow = params_ui.hash_params("doomed", Params(gain=2.0))[0]
    (path,) = [
        row["blob_path"]
        for row in conn.execute(
            "SELECT blob_path FROM runs WHERE plot = 'doomed' AND params_hash = ?",
            (slow,),
        )
    ]
    folder = os.path.dirname(path)
    os.chmod(folder, 0o555)
    yield SimpleNamespace(path=path, folder=folder)
    os.chmod(folder, 0o755)  # so that the temporary tree can be cleaned up


def test_a_blob_that_cannot_be_removed_keeps_its_run_and_is_reported(
    conn, cache, database, ledger, locked, capsys
):
    """The blob goes before the row. Were it the other way round, this blob would be left
    on disk with nothing in the ledger to name it, and nothing would ever look for it.
    """
    if os.access(locked.folder, os.W_OK):
        pytest.skip("permissions do not bind this user (root?)")
    others = snapshot(conn, cache, outside="doomed")

    # The counts forecast it, so that whoever reads them can fix the directory first.
    assert prune(database, "--plot", "doomed") == 1
    out, _ = say(capsys)
    assert (
        "unwritable    1 blob files in 1 directories this user cannot write, so"
        f" their runs would be kept: {locked.folder}" in out
    )

    assert prune(database, "--plot", "doomed", "--yes") == 1

    out, err = say(capsys)
    assert "deleted       3 runs, 4 scalar rows, 2 blob files, 2 param sets" in out
    assert "KEPT          1 runs, because their blob file could not be removed" in out
    assert f"{locked.path}  PermissionError: Permission denied" in out
    assert err == ""
    # Exactly the run whose blob stayed, with its blob, its scalars and its parameter set.
    (kept,) = conn.execute("SELECT * FROM runs WHERE plot = 'doomed'").fetchall()
    assert kept["blob_path"] == locked.path and os.path.exists(locked.path)
    assert kept["status"] == "ok"
    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE run_id = ?", kept["id"]) == 2
    assert [
        r[0] for r in conn.execute("SELECT hash FROM param_sets WHERE plot = 'doomed'")
    ] == [kept["params_hash"]]
    assert snapshot(conn, cache, outside="doomed") == others

    # Once the permissions are fixed the same command takes up where this one stopped.
    os.chmod(locked.folder, 0o755)
    assert prune(database, "--plot", "doomed", "--yes") == 0
    assert (
        "deleted       1 runs, 2 scalar rows, 1 blob files, 1 param sets"
        in say(capsys)[0]
    )
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = 'doomed'") == 0
    assert count(conn, "SELECT COUNT(*) FROM param_sets WHERE plot = 'doomed'") == 0
    assert snapshot(conn, cache, outside="doomed") == others


# ---------------------------------------------------------------------------
# The ledger: one transaction, and foreign keys
# ---------------------------------------------------------------------------


def test_the_ledger_is_deleted_in_one_transaction_or_not_at_all(
    conn, cache, database, ledger, capsys
):
    """A failure after the runs were deleted undoes them: the rows stay, as a unit."""
    with conn:
        conn.execute(
            "CREATE TRIGGER refuse BEFORE DELETE ON param_sets"
            " BEGIN SELECT RAISE(ABORT, 'refused by the test'); END"
        )
    rows = snapshot(conn, cache)["runs"]

    assert prune(database, "--plot", "doomed", "--yes") == 1

    out, err = say(capsys)
    assert "The ledger could not be updated (refused by the test)" in err
    assert "so it is unchanged" in err and "3 blob files were removed already" in err
    assert snapshot(conn, cache)["runs"] == rows
    assert count(conn, "SELECT COUNT(*) FROM scalars") == 10

    # The blobs are gone and the rows say so: the state the store already handles. The
    # same command, once whatever refused has gone, finishes the job.
    with conn:
        conn.execute("DROP TRIGGER refuse")
    assert prune(database, "--plot", "doomed", "--yes") == 0
    out, _ = say(capsys)
    assert "deleted       4 runs, 6 scalar rows, 0 blob files, 3 param sets" in out
    assert "already gone  3 blob files" in out


def test_a_connection_without_foreign_keys_is_refused_before_anything_is_deleted(
    conn, cache, database, ledger
):
    """Without the cascade the scalars would be orphaned without a word."""
    bare = sqlite3.connect(database)
    bare.row_factory = sqlite3.Row
    try:
        assert bare.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        plan = store.plan_prune(bare, "doomed")
        before = snapshot(conn, cache)
        with pytest.raises(store.PruneError, match="foreign keys off"):
            store.prune(bare, plan)
    finally:
        bare.close()
    assert snapshot(conn, cache) == before


def test_no_cache_directory_configured_stops_the_command(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FUSION_UI_CACHE", raising=False)
    assert prune(tmp_path / "empty.sqlite", "--plot", "doomed") == 1
    assert "FUSION_UI_CACHE" in say(capsys)[1]


# ---------------------------------------------------------------------------
# A plot that other plots are built on
# ---------------------------------------------------------------------------


@pytest.fixture
def chain(conn, cache, tmp_path, registered):
    """``fields`` on shots 0 and 1, built on ``bank``, with their input in the index.

    The index is what makes a run judgeable: staleness is told only for a run that
    recorded its input.
    """
    folder = tmp_path / "alcator" / "apd"
    folder.mkdir(parents=True)
    for shot in SHOTS[:2]:
        (folder / f"apd_{shot}_preprocessed.nc").write_bytes(b"x")
    catalog.rescan(conn, str(tmp_path / "alcator"), "cmod", None)
    registered(make_spec("bank"))
    fields = registered(make_derived("fields", "bank"))
    for shot in SHOTS[:2]:
        store.result(conn, fields, target(shot), DerivedParams(), ds=None)
    return fields


def test_the_runs_built_on_a_pruned_plot_lose_their_link_and_read_as_stale(
    conn, cache, database, chain, capsys
):
    banks = [
        row["id"] for row in conn.execute("SELECT id FROM runs WHERE plot = 'bank'")
    ]
    fields = conn.execute(
        "SELECT * FROM runs WHERE plot = 'fields' ORDER BY id"
    ).fetchall()
    assert [row["upstream_run_id"] for row in fields] == banks
    assert store.stale_runs(conn) == []
    before = snapshot(conn, cache)

    assert prune(database, "--plot", "bank") == 1
    out, _ = say(capsys)
    assert (
        "built on them 2 runs of other plots (fields 2): they lose their upstream"
        " link, so the store treats them as stale" in out
    )
    assert snapshot(conn, cache) == before

    assert prune(database, "--plot", "bank", "--yes") == 0
    out, _ = say(capsys)
    assert "now stale     2 runs of other plots lost their upstream" in out

    # Not refused and not deleted: the same runs with the same results and scalars,
    # their link to the deleted upstream cleared and nothing else changed.
    after = conn.execute(
        "SELECT * FROM runs WHERE plot = 'fields' ORDER BY id"
    ).fetchall()
    assert [row["upstream_run_id"] for row in after] == [None, None]

    def unlinked(rows):
        return [
            {k: v for k, v in dict(r).items() if k != "upstream_run_id"} for r in rows
        ]

    assert unlinked(after) == unlinked(fields)
    ids = {row["id"] for row in fields}
    scalars = [row for row in before["scalars"] if row[0] in ids]
    assert scalars and snapshot(conn, cache, outside="bank")["scalars"] == scalars
    blobs = [blob for blob in before["blobs"] if "/runs/fields/" in blob[0]]
    assert blobs and snapshot(conn, cache, outside="bank")["blobs"] == blobs
    assert {(r["plot"], r["stale"]) for r in store.stale_runs(conn)} == {
        ("fields", store.STALE_UPSTREAM_DELETED)
    }
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = 'bank'") == 0
    assert count(conn, "SELECT COUNT(*) FROM param_sets WHERE plot = 'bank'") == 0
    # The derived plot's own parameter set is not the pruned plot's to take.
    assert count(conn, "SELECT COUNT(*) FROM param_sets WHERE plot = 'fields'") == 1


def test_pruning_the_derived_plot_does_not_touch_what_it_was_built_on(
    conn, cache, database, chain, capsys
):
    before = snapshot(conn, cache, outside="fields")

    assert prune(database, "--plot", "fields", "--yes") == 0

    out, _ = say(capsys)
    assert "built on them no run of another plot" in out
    assert snapshot(conn, cache) == before
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = 'bank'") == 2
    assert store.stale_runs(conn) == []


# ---------------------------------------------------------------------------
# --scalar: some names of one plot, counted per name, then deleted
# ---------------------------------------------------------------------------

#: The fifteen names of the seed's records (``density_scan.discharge.BlobParameters``),
#: all of which an earlier import wrote, and the three the importer leaves out now.
SEED_NAMES = (
    "vx_c vy_c area_c vx_2dca_tde vy_2dca_tde vx_tde vy_tde lx_f ly_f lr lz theta_f"
    " taud_psd lambda_psd number_events"
).split()
ELLIPSE = ("lx_f", "ly_f", "theta_f")
PIXELS = ((6, 6), (6, 5), (4, 6))
SEED = seed.IMPORT_PLOT


def ellipse_spec(key):
    """A cached spec with a blob, and the three names (and one more) at one pixel."""
    return registry.PlotSpec(
        key=key,
        label=key,
        diagnostics=("apd",),
        params=Params,
        render=lambda result, params, target: None,
        compute=lambda ds, params: xr.Dataset({"y": ("t", np.arange(4.0))}),
        scalars=lambda result: {
            (6, 6, "lx_f"): 3.3e-3,
            (6, 6, "ly_f"): 5.8e-3,
            (6, 6, "theta_f"): 1.1,
            (6, 6, "lr"): 4.0e-3,
        },
    )


def seed_run(conn, shot, digest, rows):
    run = store.record_run(
        conn,
        target(shot),
        SEED,
        digest,
        blob_path=None,
        status="ok",
        error=None,
        seconds=None,
        code_version="imported",
    )
    store.write_scalars(conn, run["id"], rows)
    return run


@pytest.fixture
def seeded(conn, cache):
    """The ledger as it stands before the seed's three names are pruned.

    The earlier import, with all fifteen names: two shots, three pixels each, so 45
    rows a run. ``lx_f`` and ``ly_f`` are at every pixel, one of them a failed fit
    (a NULL value, which is still a row); ``theta_f`` is at two pixels only, and once
    at shot level. The three counts therefore differ: 6, 6 and 5, in two runs each.
    Beside it ``gaussian_sizes``, with the same names at one pixel and a blob, and
    ``keeper``, which has blobs and none of these names.
    """
    digest, _ = store.record_params(
        conn, SEED, seed.ImportedResults(source="results.json", sha1="0" * 40)
    )
    for number, shot in enumerate(SHOTS[:2]):
        rows = {
            (x, y, name): float(1000 * number + 10 * x + y + index)
            for x, y in PIXELS
            for index, name in enumerate(SEED_NAMES)
        }
        rows[(4, 6, "lx_f")] = float("nan")
        del rows[(4, 6, "theta_f")]
        if number == 0:
            rows["theta_f"] = 0.7
        seed_run(conn, shot, digest, rows)
    store.result(conn, ellipse_spec("gaussian_sizes"), target(), Params(), ds=None)
    for shot in SHOTS[:2]:
        store.result(conn, make_spec("keeper"), target(shot), Params(), ds=None)
    return SimpleNamespace(digest=digest)


def seed_ids(conn):
    return {
        row["id"] for row in conn.execute("SELECT id FROM runs WHERE plot = ?", (SEED,))
    }


def prune_ellipse(database, *more):
    flags = [f"--scalar={name}" for name in ELLIPSE]
    return prune(database, "--plot", SEED, *flags, *more)


# -- the store: counted per name ----------------------------------------------------------------


def test_the_counts_per_name_are_right(conn, seeded):
    plan = store.plan_prune_scalars(conn, SEED, ELLIPSE)

    assert plan.names == ELLIPSE and not plan.empty
    assert plan.counts == {"lx_f": 6, "ly_f": 6, "theta_f": 5}
    assert plan.total == 17
    first, second = sorted(seed_ids(conn))
    assert plan.rows == {
        "lx_f": {first: 3, second: 3},
        "ly_f": {first: 3, second: 3},
        # Two pixels and the shot-level row in the first run, two pixels in the second.
        "theta_f": {first: 3, second: 2},
    }
    # 89 rows in the seed's two runs, of which these 17 go; the other plots' rows are not its own.
    assert (
        count(
            conn, "SELECT COUNT(*) FROM scalars WHERE run_id IN (?, ?)", first, second
        )
        == 89
    )
    assert (plan.runs, plan.kept) == (2, 72)
    assert plan.elsewhere == {"gaussian_sizes": 3}


def test_a_name_with_no_rows_counts_as_zero_and_a_repeated_name_once(conn, seeded):
    plan = store.plan_prune_scalars(conn, SEED, ["lx_f", "nosuch", "lx_f", "theta_x"])

    assert plan.names == ("lx_f", "nosuch", "theta_x")
    assert plan.counts == {"lx_f": 6, "nosuch": 0, "theta_x": 0}
    assert plan.total == 6
    # What the plot does have, to tell a typo from a job already done.
    assert set(plan.held) == set(SEED_NAMES)
    assert "nosuch, theta_x: nothing of that name to delete" in "\n".join(plan.lines())


def test_a_plot_the_ledger_has_no_run_of_is_empty_and_the_plan_lists_the_plots_it_has(
    conn, seeded
):
    plan = store.plan_prune_scalars(conn, "density_scan_imprt", ELLIPSE)

    assert plan.empty and plan.total == 0
    assert plan.plots == {SEED: 2, "gaussian_sizes": 1, "keeper": 2}


def test_no_name_at_all_is_a_mistake_and_not_a_prune(conn, seeded):
    with pytest.raises(ValueError, match="no scalar name"):
        store.plan_prune_scalars(conn, SEED, [])


def test_counting_deletes_nothing(conn, cache, seeded):
    before = snapshot(conn, cache)

    plan = store.plan_prune_scalars(conn, SEED, ELLIPSE)
    assert plan.lines()

    assert snapshot(conn, cache) == before


# -- the store: deleted --------------------------------------------------------------------------


def test_only_those_names_under_that_plot_go(conn, cache, seeded):
    before = snapshot(conn, cache)
    ids = seed_ids(conn)
    plan = store.plan_prune_scalars(conn, SEED, ELLIPSE)

    report = store.prune_scalars(conn, plan)

    assert report.deleted == {"lx_f": 6, "ly_f": 6, "theta_f": 5}
    assert report.scalars == 17
    after = snapshot(conn, cache)
    # Exactly the seed's rows of those three names, a NULL one and a shot-level one included.
    assert after["scalars"] == [
        row for row in before["scalars"] if not (row[0] in ids and row[3] in ELLIPSE)
    ]
    assert len(before["scalars"]) - len(after["scalars"]) == 17
    assert {row[3] for row in after["scalars"] if row[0] in ids} == (
        set(SEED_NAMES) - set(ELLIPSE)
    )
    # The same names under another plot are where they were.
    elsewhere = [
        row[3] for row in after["scalars"] if row[0] not in ids and row[3] in ELLIPSE
    ]
    assert elsewhere == ["lx_f", "ly_f", "theta_f"]
    # And the runs, the parameter sets and the blobs are untouched.
    assert after["runs"] == before["runs"]
    assert after["params"] == before["params"]
    assert after["blobs"] == before["blobs"] and before["blobs"]


def test_the_runs_and_their_blobs_stay_when_a_plots_scalars_go(conn, cache, ledger):
    """On a plot with blobs: four runs, three of them with a blob and a ``corner``."""
    before = snapshot(conn, cache)
    blobs = blobs_of(conn, "doomed")
    assert len(blobs) == 3 and all(os.path.exists(path) for path in blobs)

    plan = store.plan_prune_scalars(conn, "doomed", ["corner"])
    assert plan.counts == {"corner": 3} and (plan.runs, plan.kept) == (4, 3)
    assert plan.elsewhere == {"keeper": 2}
    store.prune_scalars(conn, plan)

    after = snapshot(conn, cache)
    assert after["runs"] == before["runs"] and after["params"] == before["params"]
    assert after["blobs"] == before["blobs"]
    assert all(os.path.exists(path) for path in blobs)
    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE name = 'corner'") == 2
    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE name = 'total'") == 5


def test_a_run_written_after_the_plan_keeps_its_rows(conn, seeded):
    plan = store.plan_prune_scalars(conn, SEED, ELLIPSE)
    late = seed_run(
        conn, SHOTS[2], seeded.digest, {(6, 6, "lx_f"): 1.0, (6, 6, "lr"): 2.0}
    )

    assert store.prune_scalars(conn, plan).scalars == 17

    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE run_id = ?", late["id"]) == 2


def test_the_rows_go_in_one_transaction_or_not_at_all(conn, cache, seeded):
    """``theta_f`` is deleted last: the refusal comes after ``lx_f`` and ``ly_f`` are gone."""
    with conn:
        conn.execute(
            "CREATE TRIGGER refuse BEFORE DELETE ON scalars WHEN old.name = 'theta_f'"
            " BEGIN SELECT RAISE(ABORT, 'refused by the test'); END"
        )
    before = snapshot(conn, cache)
    plan = store.plan_prune_scalars(conn, SEED, ELLIPSE)

    with pytest.raises(store.PruneError, match="refused by the test") as error:
        store.prune_scalars(conn, plan)

    assert "so it is unchanged and nothing was deleted" in str(error.value)
    assert snapshot(conn, cache) == before

    with conn:
        conn.execute("DROP TRIGGER refuse")
    assert store.prune_scalars(conn, plan).scalars == 17


def test_pruning_scalars_on_a_connection_without_foreign_keys_is_refused(
    conn, cache, database, seeded
):
    bare = sqlite3.connect(database)
    bare.row_factory = sqlite3.Row
    try:
        assert bare.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        plan = store.plan_prune_scalars(bare, SEED, ELLIPSE)
        before = snapshot(conn, cache)
        with pytest.raises(store.PruneError, match="foreign keys off"):
            store.prune_scalars(bare, plan)
    finally:
        bare.close()
    assert snapshot(conn, cache) == before


# -- the command ---------------------------------------------------------------------------------


def test_the_dry_run_counts_per_name_deletes_nothing_and_exits_1(
    conn, cache, database, seeded, capsys
):
    before = snapshot(conn, cache)

    assert prune_ellipse(database) == 1

    assert snapshot(conn, cache) == before
    out, err = say(capsys)
    assert out.splitlines() == [
        "plot          density_scan_import",
        "lx_f          6 scalar rows in 2 runs",
        "ly_f          6 scalar rows in 2 runs",
        "theta_f       5 scalar rows in 2 runs",
        "to delete     17 scalar rows",
        "stays         all 2 runs of the plot, with their blobs, and its 72 scalar rows"
        " of other names",
        "elsewhere     the same names under other plots, not touched: gaussian_sizes 3",
    ]
    assert "Nothing was deleted. Run it again with --yes" in err


def test_yes_deletes_those_rows_and_nothing_else_and_exits_0(
    conn, cache, database, seeded, capsys
):
    before = snapshot(conn, cache)
    ids = seed_ids(conn)

    assert prune_ellipse(database, "--yes") == 0

    out, err = say(capsys)
    assert err == ""
    assert (
        out.splitlines()[-1]
        == "deleted       17 scalar rows (lx_f 6, ly_f 6, theta_f 5)"
    )
    after = snapshot(conn, cache)
    assert after["scalars"] == [
        row for row in before["scalars"] if not (row[0] in ids and row[3] in ELLIPSE)
    ]
    for key in ("runs", "params", "blobs"):
        assert after[key] == before[key], key
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = ?", SEED) == 2


def test_asked_again_every_count_is_zero_and_that_is_not_an_error(
    conn, cache, database, seeded, capsys
):
    assert prune_ellipse(database, "--yes") == 0
    say(capsys)
    before = snapshot(conn, cache)

    assert prune_ellipse(database) == 1  # the dry run's status, whatever it found
    out, err = say(capsys)
    assert "lx_f          0 scalar rows in 0 runs" in out
    assert "to delete     0 scalar rows" in out
    assert "no rows       lx_f, ly_f, theta_f: nothing of that name to delete" in out
    assert "area_c, lambda_psd, lr" in out  # what the plot does have
    assert "None of these names has a row to delete" in err

    assert prune_ellipse(database, "--yes") == 0
    out, err = say(capsys)
    assert "deleted       0 scalar rows (lx_f 0, ly_f 0, theta_f 0)" in out
    assert err == ""
    assert snapshot(conn, cache) == before


def test_a_name_that_has_no_rows_is_reported_as_zero_beside_one_that_has(
    conn, cache, database, seeded, capsys
):
    assert (
        prune(database, "--plot", SEED, "--scalar", "lx_f", "--scalar", "lxf", "--yes")
        == 0
    )

    out, err = say(capsys)
    assert "lxf           0 scalar rows in 0 runs" in out
    assert "deleted       6 scalar rows (lx_f 6, lxf 0)" in out
    assert err == ""
    left = count(conn, "SELECT COUNT(*) FROM scalars WHERE name = 'lx_f'")
    assert left == 1  # the one row gaussian_sizes has


def test_a_name_given_twice_is_counted_once(conn, cache, database, seeded, capsys):
    assert (
        prune(database, "--plot", SEED, "--scalar", "lx_f", "--scalar", "lx_f", "--yes")
        == 0
    )
    assert "deleted       6 scalar rows (lx_f 6)" in say(capsys)[0]


def test_a_plot_with_no_runs_is_an_error_that_says_which_plots_have_some(
    conn, cache, database, seeded, capsys
):
    before = snapshot(conn, cache)

    assert (
        prune(database, "--plot", "density_scan_imprt", "--scalar", "lx_f", "--yes")
        == 1
    )

    out, err = say(capsys)
    assert out == ""
    assert "'density_scan_imprt'" in err and "density_scan_import (2)" in err
    assert "gaussian_sizes (1)" in err and "keeper (2)" in err
    assert snapshot(conn, cache) == before


def test_the_scalar_prune_needs_no_result_cache(
    conn, cache, database, seeded, monkeypatch, capsys
):
    """It looks at no blob, so a machine with no ``FUSION_UI_CACHE`` can still do it."""
    monkeypatch.delenv("FUSION_UI_CACHE")
    assert prune(database, "--plot", SEED) == 1
    assert "FUSION_UI_CACHE" in say(capsys)[1]

    assert prune_ellipse(database, "--yes") == 0

    assert "deleted       17 scalar rows" in say(capsys)[0]


def test_the_plain_prune_of_the_same_plot_is_as_it_was(
    conn, cache, database, seeded, capsys
):
    """Without ``--scalar`` the whole plot goes: every run, every name, the parameter set."""
    before = snapshot(conn, cache, outside=SEED)

    assert prune(database, "--plot", SEED) == 1
    out, err = say(capsys)
    assert out.splitlines() == [
        "plot          density_scan_import",
        "runs          2  (ok 2)",
        "scalar rows   89",
        "blob files    0 on disk, 0 listed but already missing",
        "param sets    1 that nothing will reference any more",
        "built on them no run of another plot",
    ]
    assert "Nothing was deleted" in err

    assert prune(database, "--plot", SEED, "--yes") == 0
    assert (
        "deleted       2 runs, 89 scalar rows, 0 blob files, 1 param sets"
        in say(capsys)[0]
    )
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = ?", SEED) == 0
    assert snapshot(conn, cache) == before


def test_after_the_scalars_the_plain_prune_takes_what_is_left(
    conn, cache, database, seeded, capsys
):
    assert prune_ellipse(database, "--yes") == 0
    say(capsys)

    assert prune(database, "--plot", SEED, "--yes") == 0

    assert (
        "deleted       2 runs, 72 scalar rows, 0 blob files, 1 param sets"
        in say(capsys)[0]
    )
    assert count(conn, "SELECT COUNT(*) FROM runs WHERE plot = ?", SEED) == 0
    # The same names under the other plot were never the seed's to take.
    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE name = 'lx_f'") == 1
