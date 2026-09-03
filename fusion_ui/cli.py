"""``fusion-ui`` -- the commands that run outside the browser.

``rescan`` is the one that goes on cron; ``status`` is the one to run after
deploying; ``import-results`` is run once, to seed the scalar store from
``density_scan``; ``precompute`` (phase 04) warms the cache overnight.
``prune`` arrives with phase 05.
"""

import argparse
import os
import sys

from fusion_ui import config
from fusion_ui.core import catalog, db, seed


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
        os.makedirs(cache, exist_ok=True)
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


def cmd_precompute(args):
    """Run one plot's compute over every matching shot, warming the cache.

    ``fusion_ui.plots`` is imported here, not at module top, so the plain
    maintenance commands stay importable on a machine without the analysis
    packages installed.
    """
    import fusion_ui.plots  # noqa: F401 - importing the package registers specs

    from fusion_ui.core import precompute, registry

    if args.plot not in registry.REGISTRY:
        known = ", ".join(sorted(registry.REGISTRY))
        print(f"Unknown plot {args.plot!r}. Registered: {known}", file=sys.stderr)
        return 1
    spec = registry.get(args.plot)
    if not spec.cached:
        print(
            f"{args.plot!r} is a live view; it has nothing to precompute.",
            file=sys.stderr,
        )
        return 1

    shots = set(args.shot) if args.shot else None
    params = precompute.default_params(spec, args.pixel)
    conn = db.open_db(args.database)
    try:
        targets = precompute.targets_for(conn, spec, args.machine, shots=shots)
        if not targets:
            print(
                f"No indexed shots match plot {args.plot!r} for machine "
                f"{args.machine!r}. Run `fusion-ui rescan` first.",
                file=sys.stderr,
            )
            return 1

        stats = precompute.run(conn, spec, targets, params, force=args.force)
        print(stats.summary())
        return 0
    finally:
        conn.close()


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
        help="run a plot's compute over every matching shot to warm the cache",
    )
    precompute.add_argument("plot", help="plot key, e.g. velocity_contour")
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
        help="restrict to this shot number (repeatable; default: every shot)",
    )
    precompute.add_argument(
        "--pixel",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="set the reference pixel (refx/refy) instead of the spec default",
    )
    precompute.add_argument(
        "--force",
        action="store_true",
        help="recompute even when a cached result already exists",
    )
    precompute.set_defaults(func=cmd_precompute)

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
