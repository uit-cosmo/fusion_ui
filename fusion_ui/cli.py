"""``fusion-ui`` -- the commands that run outside the browser.

``rescan`` is the one that goes on cron; ``status`` is the one to run after
deploying; ``import-results`` is run once, to seed the scalar store from
``density_scan``; ``precompute`` (phase 04) warms the cache overnight, for
several plots and on several workers since phase 06. ``prune`` deletes every
stored result of one plot, or with ``--scalar`` only the scalars of some names
under it, and counts them first.
"""

import argparse
import os
import sys

from fusion_ui import config
from fusion_ui.core import catalog, db, seed, shared, store


def _resolve(attribute):
    try:
        return getattr(config, attribute), None
    except RuntimeError as error:
        return None, str(error)


def cmd_init_db(args):
    path = args.database or config.UI_DB_PATH
    conn = db.connect(path)
    version = db.init_db(conn)
    conn.close()
    # The other half of the state this app owns. Empty until phase 02, but
    # created here so a permissions problem shows up at deploy time.
    cache, error = _resolve("CACHE_DIR")
    if not error:
        shared.makedirs(cache)
    print(f"{path}: schema v{version}")
    return 0


def cmd_rescan(args):
    data_folder = args.data_folder or config.DATA_FOLDER
    if not os.path.isdir(data_folder):
        print(f"Data folder {data_folder!r} does not exist.", file=sys.stderr)
        return 1
    discharge_db, error = _resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(discharge_db):
        print(
            "Discharge DB unavailable — indexing anyway, every shot will be "
            "flagged as missing metadata.",
            file=sys.stderr,
        )
        discharge_db = None

    conn = db.open_db(args.database)
    stats = catalog.rescan(conn, data_folder, args.machine, discharge_db)
    print(stats.summary())
    return 0


def cmd_import_results(args):
    conn = db.open_db(args.database)
    try:
        stats = seed.import_results(conn, args.results, args.machine)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(stats.summary())
    return 0


def _print_flushed(message):
    # Flush every line: a fill can run for days, and a buffered "computing…"
    # line is no use to someone tailing the log overnight. A closed pipe (tee
    # killed by the same Ctrl-C) must not turn into a traceback.
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


def _selection_flags(args):
    """The flags that chose these targets, to repeat in a suggested command."""
    flags = [f" --shot {shot}" for shot in args.shot]
    flags += [f" --run-day {day}" for day in args.run_day]
    if args.machine != config.MACHINE:
        flags.append(f" --machine {args.machine}")
    if args.workers > 1:
        flags.append(f" --workers {args.workers}")
    return "".join(flags)


def cmd_precompute(args):
    """Run the plots' compute over every matching shot, warming the cache.

    Several plots run in dependency order within each shot; ``--workers``
    spreads the shots over a pool of processes. ``fusion_ui.plots`` is
    imported here, not at module top, so the plain maintenance commands stay
    importable on a machine without the analysis packages installed.
    """
    import fusion_ui.plots  # noqa: F401 - importing the package registers specs

    from fusion_ui.core import precompute, registry

    keys = list(dict.fromkeys(args.plots))
    for key in keys:
        if key not in registry.REGISTRY:
            known = ", ".join(sorted(registry.REGISTRY))
            print(f"Unknown plot {key!r}. Registered: {known}", file=sys.stderr)
            return 1
        if not registry.get(key).cached:
            print(
                f"{key!r} is a live view; it has nothing to precompute.",
                file=sys.stderr,
            )
            return 1
    if args.stale and (args.force or args.retry_failed):
        print(
            "--stale recomputes exactly the stale results, whatever their status;"
            " it does not combine with --force or --retry-failed.",
            file=sys.stderr,
        )
        return 1
    specs = precompute.in_dependency_order([registry.get(key) for key in keys])

    # Every parameter set is settled before the database is opened: a file
    # that does not fit must stop the command before anything is computed.
    try:
        if args.params_json:
            params = precompute.params_from_file(args.params_json, specs, args.pixel)
        else:
            params = {
                spec.key: precompute.default_params(spec, args.pixel) for spec in specs
            }
    except precompute.ParamsError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.nice is not None:
        nice = args.nice
    else:
        nice = precompute.DEFAULT_NICE if args.workers > 1 else None
    conn = db.open_db(args.database)
    try:
        try:
            plan = _precompute_plan(args, conn, keys, specs, params)
        except KeyboardInterrupt:
            print("interrupted before anything was computed", file=sys.stderr)
            return 130
        if isinstance(plan, int):  # nothing to do: the exit status
            return plan
        report = precompute.execute(
            conn, plan, log=_print_flushed, workers=args.workers, nice=nice
        )
        for line in report.summary_lines():
            _print_flushed(line)
        if plan.unnamed_upstreams:
            plots = precompute.in_dependency_order(
                [registry.get(key) for key in (*plan.unnamed_upstreams, *keys)]
            )
            _print_flushed(
                "Some results were left stale because an upstream of theirs is"
                " stale too and was not named. To recompute them as well, run"
                f" `fusion-ui precompute {' '.join(spec.key for spec in plots)}"
                f" --stale{_selection_flags(args)}`"
            )
        if report.broken:
            return 1
        return 130 if report.interrupted else 0
    finally:
        conn.close()


def _precompute_plan(args, conn, keys, specs, params):
    """The plan for ``precompute``, or the exit status when there is nothing to do.

    Reads the ledger and the index, and records the parameter sets; opens no
    data file.
    """
    from fusion_ui.core import precompute

    shots = set(args.shot) or None
    run_days = set(args.run_day) or None
    if args.stale:
        # --params-json or --pixel picks one parameter set per plot; without
        # them every stale set is recomputed, each with its own parameters.
        chosen = params if (args.params_json or args.pixel) else None
        plan = precompute.plan_stale(
            conn, specs, args.machine, shots, run_days, only=chosen
        )
        if not plan.jobs:
            print(
                f"Nothing is stale for {', '.join(keys)} on the shots selected"
                f" ({args.machine})."
            )
            return 0
        return plan
    targets = precompute.select_targets(conn, specs, args.machine, shots, run_days)
    if not targets:
        named = (
            f"plot {keys[0]!r}"
            if len(keys) == 1
            else "plots " + ", ".join(repr(key) for key in keys)
        )
        print(
            f"No indexed shots match {named} for machine "
            f"{args.machine!r}. Run `fusion-ui rescan` first.",
            file=sys.stderr,
        )
        return 1
    return precompute.plan_fill(
        conn,
        [(spec, params[spec.key]) for spec in specs],
        targets,
        force=args.force,
        retry_failed=args.retry_failed,
    )


def cmd_prune(args):
    """Delete every stored result of one plot: blobs, runs, scalars, parameter sets.

    Counts first and prints, and deletes only with ``--yes``; without it the
    exit status is 1, so a script cannot mistake a count for a deletion. The key
    need not be registered: this is how the results of a removed spec go. The
    deletion is :func:`fusion_ui.core.store.prune`, whose docstring says in what
    order, and why. Imports nothing of the analysis packages.

    With ``--scalar NAME`` (repeatable) it deletes only the scalar rows of those
    names under ``--plot``'s runs, in the same two steps (:func:`_prune_scalars`).
    """
    conn = db.open_db(args.database)
    try:
        if args.scalar:
            return _prune_scalars(args, conn)
        plan = store.plan_prune(conn, args.plot)
        if plan.empty:
            known = ", ".join(f"{plot} ({n})" for plot, n in plan.plots.items())
            print(
                f"The ledger holds nothing to prune for plot {args.plot!r}. Plots"
                f" with runs: {known or 'none'}.",
                file=sys.stderr,
            )
            return 1
        # Flushed, so that stdout stays ahead of stderr in a log (`2>&1 | tee`).
        for line in plan.lines():
            _print_flushed(line)
        if plan.refusal:  # also without --yes: it is not a matter of confirming
            print(plan.refusal, file=sys.stderr)
            return 1
        if not args.yes:
            print(
                "Nothing was deleted. Run it again with --yes to delete these.",
                file=sys.stderr,
            )
            return 1
        report = store.prune(conn, plan)
        for line in report.lines():
            _print_flushed(line)
        return 1 if report.kept else 0
    finally:
        conn.close()


def _prune_scalars(args, conn):
    """``prune --plot KEY --scalar NAME ...``: the scalars of those names, counted first.

    The runs, their blobs, their other scalars and the same names under any other
    plot key stay (:func:`fusion_ui.core.store.prune_scalars`). Without ``--yes``
    it prints the counts, deletes nothing and exits 1, as the whole-plot prune
    does. A name with no rows is counted as 0 and is not an error; a plot with no
    runs at all is, since that is a mistyped key.
    """
    plan = store.plan_prune_scalars(conn, args.plot, args.scalar)
    if plan.empty:
        known = ", ".join(f"{plot} ({n})" for plot, n in plan.plots.items())
        print(
            f"The ledger holds no runs of plot {args.plot!r}, so no scalars to"
            f" prune. Plots with runs: {known or 'none'}.",
            file=sys.stderr,
        )
        return 1
    for line in plan.lines():
        _print_flushed(line)
    if not args.yes:
        print(
            "Nothing was deleted. "
            + (
                "Run it again with --yes to delete these."
                if plan.total
                else "None of these names has a row to delete."
            ),
            file=sys.stderr,
        )
        return 1
    report = store.prune_scalars(conn, plan)
    for line in report.lines():
        _print_flushed(line)
    return 0


def cmd_backfill_dt(args):
    """Measure each phantom file's frame interval into ``shots.dt``.

    Reads only the 1-D time axis per file, so a whole machine's phantom
    collection takes seconds, not hours. Run on the machine that holds the
    data; the browser's phantom-dt column reads what this leaves behind.
    """
    conn = db.open_db(args.database)
    try:
        shots = set(args.shot) if args.shot else None
        stats = catalog.backfill_dt(conn, args.machine, shots=shots, force=args.force)
    finally:
        conn.close()
    print(stats.summary())
    return 0


def cmd_status(args):
    print(f"machine          {config.MACHINE}")
    for label, attribute in (
        ("discharge DB", "DISCHARGE_DB_PATH"),
        ("data folder", "DATA_FOLDER"),
        ("app database", "UI_DB_PATH"),
        ("result cache", "CACHE_DIR"),
    ):
        value, error = _resolve(attribute)
        if error:
            print(f"{label:<16} (unset)")
        else:
            print(
                f"{label:<16} {value} {'✓' if os.path.exists(value) else '✗ missing'}"
            )

    conn = db.open_db(args.database)
    print(f"schema           v{db.schema_version(conn)}")
    rows = conn.execute(
        "SELECT machine, diagnostic, preprocessed, COUNT(*) AS n,"
        "       SUM(has_metadata) AS curated"
        "  FROM shots GROUP BY machine, diagnostic, preprocessed"
        "  ORDER BY machine, diagnostic, preprocessed"
    ).fetchall()
    if not rows:
        print("index            empty — run `fusion-ui rescan`")
    else:
        print("index")
        for row in rows:
            kind = "preprocessed" if row["preprocessed"] else "raw"
            print(
                f"  {row['machine']} {row['diagnostic']:<8} {kind:<12} "
                f"{row['n']:>6} files, {row['curated']:>6} curated"
            )
        shots = conn.execute("SELECT COUNT(DISTINCT shot) FROM shots").fetchone()[0]
        print(f"  {shots} distinct shots")

    runs = conn.execute(
        "SELECT r.plot, r.status, COUNT(*) AS n,"
        "       COUNT(DISTINCT r.shot) AS shots,"
        "       (SELECT COUNT(*) FROM scalars s WHERE s.run_id IN"
        "          (SELECT id FROM runs q WHERE q.plot = r.plot"
        "                                AND q.status = r.status)) AS scalars"
        "  FROM runs r GROUP BY r.plot, r.status ORDER BY r.plot, r.status"
    ).fetchall()
    if not runs:
        print("results          none yet")
        return 0
    print("results")
    for row in runs:
        print(
            f"  {row['plot']:<24} {row['status']:<7} {row['n']:>5} runs, "
            f"{row['shots']:>4} shots, {row['scalars']:>7} scalars"
        )
    return 0


def _run_day(text):
    """A run day: the first seven digits of a C-Mod shot number ``1YYMMDDnnn``."""
    if len(text) != 7 or not text.isdigit():
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a run day: give its 7 digits, e.g. 1160616 for"
            " shots 1160616001 to 1160616999"
        )
    return int(text)


def _positive(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected a whole number of at least 1, got {text!r}"
        )
    return value


def _niceness(text):
    try:
        value = int(text)
    except ValueError:
        value = -1
    if not 0 <= value <= 19:
        raise argparse.ArgumentTypeError(
            f"expected a niceness from 0 to 19, got {text!r} (lowering it needs"
            " root, and this never asks for that)"
        )
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fusion-ui", description="Shot Explorer maintenance commands."
    )
    parser.add_argument(
        "--database",
        default=None,
        metavar="PATH",
        help="app SQLite file (default: $FUSION_UI_DB)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-db", help="create or migrate the app database")
    init.set_defaults(func=cmd_init_db)

    rescan = subparsers.add_parser("rescan", help="index the data tree into `shots`")
    rescan.add_argument(
        "--machine",
        default=None,
        help="machine the data belongs to (default: $FUSION_MACHINE, else cmod)",
    )
    rescan.add_argument(
        "--data-folder",
        default=None,
        metavar="PATH",
        help="data tree to walk (default: $FUSION_DATA_FOLDER)",
    )
    rescan.set_defaults(func=cmd_rescan)

    precompute = subparsers.add_parser(
        "precompute",
        help="run plots' compute over every matching shot to warm the cache",
    )
    precompute.add_argument(
        "plots",
        nargs="+",
        metavar="PLOT",
        help="plot key, e.g. velocity_contour; several run in dependency order"
        " within each shot, so an upstream named with them is computed once",
    )
    precompute.add_argument(
        "--machine",
        default=None,
        help="machine the data belongs to (default: $FUSION_MACHINE, else cmod)",
    )
    precompute.add_argument(
        "--shot",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help="restrict to this shot number (repeatable, and adds to --run-day;"
        " default: every shot)",
    )
    precompute.add_argument(
        "--run-day",
        action="append",
        type=_run_day,
        default=[],
        metavar="D",
        help="restrict to the shots of this run day, the first 7 digits of a"
        " shot number, e.g. 1160616 (repeatable, and adds to --shot)",
    )
    precompute.add_argument(
        "--pixel",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="set the reference pixel (refx/refy) instead of the spec default",
    )
    precompute.add_argument(
        "--params-json",
        default=None,
        metavar="PATH",
        help="parameters instead of the defaults: one plot's complete set as"
        " param_sets.params_json stores it (what the single-shot page shows),"
        ' or only the fields to change, e.g. {"averages": {"window": 30}},'
        " applied to every plot named",
    )
    precompute.add_argument(
        "--force",
        action="store_true",
        help="recompute even when a cached result already exists",
    )
    precompute.add_argument(
        "--retry-failed",
        action="store_true",
        help="recompute rows previously recorded as failed (e.g. after fixing"
        " a full disk); without it, failed rows are skipped without reopening",
    )
    precompute.add_argument(
        "--stale",
        action="store_true",
        help="recompute in place only the results whose input file or upstream"
        " changed since, each with its own parameters; one built on a stale"
        " upstream that is not named is left, with the command that fixes it",
    )
    precompute.add_argument(
        "--workers",
        type=_positive,
        default=1,
        metavar="N",
        help="processes computing at once, largest file first (default: 1, in"
        " this process)",
    )
    precompute.add_argument(
        "--nice",
        type=_niceness,
        default=None,
        metavar="N",
        help="run the computes at this niceness, set rather than added to what"
        " they inherit (default: 10 with --workers above 1, else unchanged)",
    )
    precompute.set_defaults(func=cmd_precompute)

    prune = subparsers.add_parser(
        "prune",
        help="delete every stored result of one plot, or only some of its scalars"
        " (counts first; --yes deletes)",
    )
    prune.add_argument(
        "--plot",
        required=True,
        metavar="KEY",
        help="plot key whose blobs, runs, scalars and parameter sets go. It need"
        " not be registered any more, which is how a removed spec's results are"
        " cleared; the other plots are untouched. With --scalar, the plot key"
        " whose scalars of those names go, and nothing else of it",
    )
    prune.add_argument(
        "--scalar",
        action="append",
        default=None,
        metavar="NAME",
        help="delete only the scalar rows of this exact name under --plot's runs"
        " (repeatable). The runs, their blobs, their other scalars and the same"
        " name under any other plot stay. A name with no rows counts as 0; it is"
        " not an error",
    )
    prune.add_argument(
        "--yes",
        action="store_true",
        help="delete. Without it the counts are printed, nothing is deleted and"
        " the exit status is 1",
    )
    prune.set_defaults(func=cmd_prune)

    backfill = subparsers.add_parser(
        "backfill-dt",
        help="measure each phantom file's frame interval into `shots.dt`",
    )
    backfill.add_argument(
        "--machine",
        default=None,
        help="machine the data belongs to (default: $FUSION_MACHINE, else cmod)",
    )
    backfill.add_argument(
        "--shot",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help="restrict to this shot number (repeatable; default: every phantom file)",
    )
    backfill.add_argument(
        "--force",
        action="store_true",
        help="re-measure even where a value is already stored",
    )
    backfill.set_defaults(func=cmd_backfill_dt)

    seed_results = subparsers.add_parser(
        "import-results",
        help="seed `scalars` from a density_scan results.json",
    )
    seed_results.add_argument(
        "--results",
        default=None,
        metavar="PATH",
        help="results.json to import (default: fusion_scripts' DENSITY_SCAN_RESULTS)",
    )
    seed_results.set_defaults(func=cmd_import_results)

    status = subparsers.add_parser("status", help="resolved paths and index counts")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "machine", None) is None:
        args.machine = config.MACHINE
    try:
        return args.func(args)
    except RuntimeError as error:  # unset config variable, wrong schema version
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
