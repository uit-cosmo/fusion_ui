"""Central configuration for fusion_ui.

Machine-specific paths -- the read-only ``experimental_database`` descriptor,
the raw data folder, and the two locations this app owns outright (its SQLite
database and its result cache) -- are read from environment variables,
optionally populated from a ``.env`` file.

To set up a new machine, copy ``.env.example`` to ``.env`` next to the checkout
and edit the values (or export the variables in your shell). ``.env`` is
gitignored, so no machine-specific path is ever committed.

``.env`` discovery, in priority order (shell variables always win over every
file below):

1. ``$FUSION_UI_DOTENV``, when set -- an explicit path to a dotenv file.
2. ``.env`` in the current working directory, then each of its parents upward.
   This is what makes ``fusion-ui precompute`` work when run from inside the
   checkout.
3. ``.env`` next to the checkout (the package parent directory). This is the
   editable-install layout the deploy script sets up, and what previous
   versions of this module were the only place they looked.
4. ``.env`` in each parent of the running Python executable, nearest first.
   This covers a non-editable install whose ``.venv`` lives inside the
   checkout: the module itself then lives in ``site-packages`` and the package
   parent has no ``.env``, but ``.../checkout/.venv/bin/python`` walks up to
   ``.../checkout/.env``.

This mirrors ``fusion_scripts/config.py`` deliberately: the two repositories
read the same ``FUSION_DISCHARGE_DB`` and ``FUSION_DATA_FOLDER`` variables, so
a machine configured for one is already configured for the other.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Populate os.environ from a simple ``KEY=VALUE`` file.

    A tiny, dependency-free parser. Lines that are blank or start with ``#`` are
    ignored. Variables already present in the environment take precedence, so an
    explicit ``export`` always overrides the ``.env`` file.
    """
    path = Path(os.path.expanduser(str(path)))
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _candidate_dotenv_paths():
    """``.env`` locations to try, highest priority first. See module docstring."""
    candidates = []
    seen = set()

    def add(path):
        try:
            key = os.path.abspath(os.path.expanduser(str(path)))
        except (TypeError, ValueError):
            return
        if key not in seen:
            seen.add(key)
            candidates.append(Path(key))

    explicit = os.environ.get("FUSION_UI_DOTENV")
    if explicit:
        add(explicit)
    try:
        here = Path.cwd()
    except OSError:
        here = None
    if here is not None:
        add(here / ".env")
        for parent in here.parents:
            add(parent / ".env")
    add(_REPO_ROOT / ".env")
    executable = getattr(sys, "executable", None)
    if executable:
        try:
            exe = Path(executable).resolve()
        except OSError:
            exe = None
        if exe is not None:
            for parent in exe.parents[:8]:
                add(parent / ".env")
    return candidates


def _load_all_dotenvs():
    for path in _candidate_dotenv_paths():
        _load_dotenv(path)


_load_all_dotenvs()


def _require(name):
    value = os.environ.get(name)
    if not value:
        try:
            tried = str(Path.cwd() / ".env")
        except OSError:
            tried = ".env in the current directory"
        raise RuntimeError(
            f"Required environment variable {name!r} is not set.\n"
            f"Copy .env.example to .env next to the checkout and set the "
            f"paths for this machine (tried {tried} and {_REPO_ROOT / '.env'}), "
            f"point $FUSION_UI_DOTENV at a dotenv file, or export {name} in "
            f"your shell."
        )
    return value


# Machine-specific paths, resolved lazily (PEP 562) so importing this module --
# and anything that imports it -- works on a machine without a .env; the
# RuntimeError from _require only fires when a path is actually used.
#   DISCHARGE_DB_PATH -- experimental_database ``plasma_discharges.json``. READ-ONLY.
#   DATA_FOLDER -- root folder holding the raw / preprocessed shot data.
#   UI_DB_PATH -- this app's own SQLite file. Created on first connect.
#   CACHE_DIR -- this app's netCDF result blobs. Unused until phase 02.
_ENV_PATHS = {
    "DISCHARGE_DB_PATH": "FUSION_DISCHARGE_DB",
    "DATA_FOLDER": "FUSION_DATA_FOLDER",
    "UI_DB_PATH": "FUSION_UI_DB",
    "CACHE_DIR": "FUSION_UI_CACHE",
}

# Which machine the data on this server came from. Shot numbers are unique only
# within a machine, so every row the app writes carries it. Everything on disk
# today is Alcator C-Mod; W7-X is expected later.
_DEFAULT_MACHINE = "cmod"


def __getattr__(name):
    if name in _ENV_PATHS:
        return os.path.expanduser(_require(_ENV_PATHS[name]))
    if name == "MACHINE":
        return os.environ.get("FUSION_MACHINE") or _DEFAULT_MACHINE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


REPO_ROOT = _REPO_ROOT
