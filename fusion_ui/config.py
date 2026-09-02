"""Central configuration for fusion_ui.

Machine-specific paths -- the read-only ``experimental_database`` descriptor,
the raw data folder, and the two locations this app owns outright (its SQLite
database and its result cache) -- are read from environment variables,
optionally populated from a ``.env`` file at the repository root.

To set up a new machine, copy ``.env.example`` to ``.env`` and edit the values
(or export the variables in your shell). ``.env`` is gitignored, so no
machine-specific path is ever committed.

This mirrors ``fusion_scripts/config.py`` deliberately: the two repositories
read the same ``FUSION_DISCHARGE_DB`` and ``FUSION_DATA_FOLDER`` variables, so
a machine configured for one is already configured for the other.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Populate os.environ from a simple ``KEY=VALUE`` file.

    A tiny, dependency-free parser. Lines that are blank or start with ``#`` are
    ignored. Variables already present in the environment take precedence, so an
    explicit ``export`` always overrides the ``.env`` file.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(_REPO_ROOT / ".env")


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set.\n"
            f"Copy {_REPO_ROOT / '.env.example'} to {_REPO_ROOT / '.env'} and "
            f"set the paths for this machine, or export {name} in your shell."
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
