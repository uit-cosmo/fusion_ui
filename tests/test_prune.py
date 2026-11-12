"""``fusion-ui prune``: one plot's results counted, then deleted, and nothing else touched.

The ledger is built the way the app builds it, with ``store.result`` on synthetic specs,
so every blob is a real netCDF under a real hash path and every run has its scalars
and its parameter set, and the command is driven through ``cli.main`` on that file.

The last section is ``--scalar``: only the scalars of some names under one plot key,
with the runs, the blobs, the other scalars and the same names under other plots left
as they are.
"""

import dataclasses
import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from fusion_ui import cli
from fusion_ui.core import catalog, params_ui, registry, store

SHOTS = (1160616027, 1160616028, 1160616029)


@dataclasses.dataclass
class Params:
    gain: float = 1.0


@dataclasses.dataclass
class DerivedParams:
    base: Params = dataclasses.field(default_factory=Params)
    scale: float = 2.0


def total_and_corner(result):
    return {
        "total": float(result["y"].sum()),
        (3, 4, "corner"): float(result["y"][-1]),
    }


def make_spec(key, scalars=total_and_corner):
    """A cached spec that writes a blob and, unless told otherwise, two scalars,
    one of them at a pixel."""
    return registry.PlotSpec(
        key=key,
        label=key,
        diagnostics=("apd",),
        params=Params,
        render=lambda result, params, target: None,
        compute=lambda ds, params: xr.Dataset(
            {"y": ("t", np.arange(4.0) * params.gain)}
        ),
        scalars=scalars,
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


def test_a_pruned_plot_goes_whole_and_every_other_plot_is_untouched(
    conn, cache, database, ledger, capsys
):
    # The key is not registered, as the key of a removed spec is not: prune
    # reads the ledger, never the registry.
    assert "doomed" not in registry.REGISTRY
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
        before = snapshot(conn, cache)
        with pytest.raises(store.PruneError, match="foreign keys off"):
            store.prune(bare, store.plan_prune(bare, "doomed"))
        scalars = store.plan_prune_scalars(bare, "doomed", ["corner"])
        with pytest.raises(store.PruneError, match="foreign keys off"):
            store.prune_scalars(bare, scalars)
    finally:
        bare.close()
    assert snapshot(conn, cache) == before


def test_without_a_cache_directory_only_the_scalar_prune_runs(
    conn, cache, database, ledger, monkeypatch, capsys
):
    """The whole-plot prune must tell the cache's blobs from any others; the
    scalar prune looks at no blob."""
    monkeypatch.delenv("FUSION_UI_CACHE")
    assert prune(database, "--plot", "doomed") == 1
    assert "FUSION_UI_CACHE" in say(capsys)[1]

    assert prune(database, "--plot", "doomed", "--scalar", "corner", "--yes") == 0
    assert "deleted       3 scalar rows (corner 3)" in say(capsys)[0]


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

#: The plot of the example, and two of the scalars it stopped writing:
#: ``fusion-ui prune --plot blob_parameters --scalar level --scalar lx_c --yes``.
BLOBS = "blob_parameters"
RETIRED = ("level", "lx_c")


def blob_scalars(result):
    """Seven rows a run: ``level`` at two pixels, one of them a failed fit (a
    NaN, stored as NULL, which is still a row); ``lx_c`` at two pixels and once
    at shot level; and two names that stay."""
    return {
        (6, 6, "level"): 0.5,
        (6, 5, "level"): float("nan"),
        (6, 6, "lx_c"): 3.3e-3,
        (6, 5, "lx_c"): 3.1e-3,
        "lx_c": 3.2e-3,
        (6, 6, "area"): 1.0e-5,
        (6, 6, "lr"): 4.0e-3,
    }


@pytest.fixture
def blobs(conn, cache):
    """``blob_parameters`` on two shots; ``sizes``, another plot with one of the
    names; and ``keeper``, which has none of them."""
    spec = make_spec(BLOBS, blob_scalars)
    for shot in SHOTS[:2]:
        store.result(conn, spec, target(shot), Params(), ds=None)
        store.result(conn, make_spec("keeper"), target(shot), Params(), ds=None)
    sizes = make_spec("sizes", lambda result: {(6, 6, "lx_c"): 3.0e-3, "lr": 4.0e-3})
    store.result(conn, sizes, target(), Params(), ds=None)
    return spec


def without(snap, plot_runs, names):
    """``snap`` less the scalar rows of ``names`` in the runs ``plot_runs``."""
    kept = [r for r in snap["scalars"] if not (r[0] in plot_runs and r[3] in names)]
    return {**snap, "scalars": kept}


def prune_retired(database, *more):
    flags = [f"--scalar={name}" for name in RETIRED]
    return prune(database, "--plot", BLOBS, *flags, *more)


def test_the_dry_run_counts_per_name_deletes_nothing_and_exits_1(
    conn, cache, database, blobs, capsys
):
    before = snapshot(conn, cache)

    assert prune_retired(database) == 1

    assert snapshot(conn, cache) == before
    out, err = say(capsys)
    assert out.splitlines() == [
        "plot          blob_parameters",
        "level         4 scalar rows in 2 runs",
        "lx_c          6 scalar rows in 2 runs",
        "to delete     10 scalar rows",
        "stays         all 2 runs of the plot, with their blobs, and its 4 scalar rows"
        " of other names",
        "elsewhere     the same names under other plots, not touched: sizes 1",
    ]
    assert "Nothing was deleted. Run it again with --yes" in err


def test_yes_deletes_those_rows_and_nothing_else_and_asked_again_finds_none(
    conn, cache, database, blobs, capsys
):
    before = snapshot(conn, cache)
    assert before["blobs"]
    runs = {r[0] for r in conn.execute("SELECT id FROM runs WHERE plot = ?", (BLOBS,))}

    assert prune_retired(database, "--yes") == 0

    out, err = say(capsys)
    assert out.splitlines()[-1] == "deleted       10 scalar rows (level 4, lx_c 6)"
    assert err == ""
    # Exactly the plot's rows of those names, the NULL and the shot-level one
    # included. Its runs, blobs, parameter set and other names are as they
    # were, and so is the same name under the other plot.
    after = snapshot(conn, cache)
    assert after == without(before, runs, RETIRED)
    assert len(before["scalars"]) - len(after["scalars"]) == 10
    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE name = 'lx_c'") == 1

    # Asked again every count is 0, which is not an error.
    assert prune_retired(database) == 1  # the dry run's status, whatever it found
    out, err = say(capsys)
    assert "lx_c          0 scalar rows in 0 runs" in out
    assert "to delete     0 scalar rows" in out
    assert "no rows       level, lx_c: nothing of that name to delete" in out
    assert "The scalar names this plot has: area, lr" in out
    assert "None of these names has a row to delete" in err

    assert prune_retired(database, "--yes") == 0
    out, err = say(capsys)
    assert "deleted       0 scalar rows (level 0, lx_c 0)" in out
    assert err == ""
    assert snapshot(conn, cache) == after


def test_a_name_with_no_rows_counts_as_zero_and_a_repeated_name_once(
    conn, cache, database, blobs, capsys
):
    flags = ["--scalar=lx_c", "--scalar=lxc", "--scalar=lx_c"]
    assert prune(database, "--plot", BLOBS, *flags, "--yes") == 0

    out, err = say(capsys)
    assert "lxc           0 scalar rows in 0 runs" in out
    # What the plot does have, to tell a typo from a job already done.
    assert "lxc: nothing of that name to delete" in out
    assert "The scalar names this plot has: area, level, lr, lx_c" in out
    assert "deleted       6 scalar rows (lx_c 6, lxc 0)" in out
    assert err == ""


def test_a_plot_with_no_runs_is_an_error_that_says_which_plots_have_some(
    conn, cache, database, blobs, capsys
):
    before = snapshot(conn, cache)

    assert prune(database, "--plot", "blob_parameter", "--scalar", "lx_c", "--yes") == 1

    out, err = say(capsys)
    assert out == ""
    assert "'blob_parameter'" in err and "blob_parameters (2)" in err
    assert "sizes (1)" in err and "keeper (2)" in err
    assert snapshot(conn, cache) == before


def test_no_name_at_all_is_a_mistake_and_not_a_prune(conn, blobs):
    with pytest.raises(ValueError, match="no scalar name"):
        store.plan_prune_scalars(conn, BLOBS, [])


def test_a_run_written_after_the_plan_keeps_its_rows(conn, blobs):
    plan = store.plan_prune_scalars(conn, BLOBS, RETIRED)
    _, late = store.result(conn, blobs, target(SHOTS[2]), Params(), ds=None)

    assert store.prune_scalars(conn, plan).scalars == 10

    assert count(conn, "SELECT COUNT(*) FROM scalars WHERE run_id = ?", late["id"]) == 7


def test_the_rows_go_in_one_transaction_or_not_at_all(conn, cache, blobs):
    """``lx_c`` is deleted last: the refusal comes after ``level`` is gone."""
    with conn:
        conn.execute(
            "CREATE TRIGGER refuse BEFORE DELETE ON scalars WHEN old.name = 'lx_c'"
            " BEGIN SELECT RAISE(ABORT, 'refused by the test'); END"
        )
    before = snapshot(conn, cache)
    plan = store.plan_prune_scalars(conn, BLOBS, RETIRED)

    with pytest.raises(store.PruneError, match="refused by the test") as error:
        store.prune_scalars(conn, plan)

    assert "so it is unchanged and nothing was deleted" in str(error.value)
    assert snapshot(conn, cache) == before

    with conn:
        conn.execute("DROP TRIGGER refuse")
    assert store.prune_scalars(conn, plan).scalars == 10
