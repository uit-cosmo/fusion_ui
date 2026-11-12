"""Central configuration for fusion_ui.

Machine-specific paths are read from environment variables, optionally filled
from a ``.env`` file: copy ``.env.example`` to ``.env`` next to the checkout and
edit it. ``.env`` is gitignored, so no machine-specific path is committed.

A variable set in the shell wins over every file. The files are read in this
order, and the first to set a variable wins:

1. ``$FUSION_UI_DOTENV``, an explicit path to a dotenv file;
2. ``.env`` in the current working directory, then in each of its parents;
3. ``.env`` next to the checkout (the parent of this package), which is where
   an editable install finds it whatever the working directory.

``FUSION_DISCHARGE_DB`` and ``FUSION_DATA_FOLDER`` are the variables the group's
other repositories read, so a machine configured for one is configured for all.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path):
    """Fill ``os.environ`` from a ``KEY=VALUE`` file, if it exists.

    Blank lines and lines starting with ``#`` are ignored. A variable already in
    the environment is left as it is.
    """
    path = Path(path).expanduser()
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _dotenv_paths():
    """``.env`` locations to try, highest priority first (see the module docstring)."""
    paths = []
    explicit = os.environ.get("FUSION_UI_DOTENV")
    if explicit:
        paths.append(Path(explicit))
    try:
        here = Path.cwd()
    except OSError:  # the working directory no longer exists
        here = None
    if here is not None:
        paths += [folder / ".env" for folder in (here, *here.parents)]
    paths.append(_REPO_ROOT / ".env")
    return paths


for _path in _dotenv_paths():
    _load_dotenv(_path)


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set.\n"
            f"Copy .env.example to .env next to the checkout "
            f"({_REPO_ROOT / '.env'}) and set the paths for this machine, "
            f"point $FUSION_UI_DOTENV at a dotenv file, or export {name} in "
            f"your shell."
        )
    return value


# Resolved lazily (PEP 562), so importing this module works on a machine
# without a .env and only using a path raises.
_ENV_PATHS = {
    "DISCHARGE_DB_PATH": "FUSION_DISCHARGE_DB",  # plasma_discharges.json, read-only
    "DATA_FOLDER": "FUSION_DATA_FOLDER",  # one subfolder per diagnostic
    "UI_DB_PATH": "FUSION_UI_DB",  # this app's SQLite file
    "CACHE_DIR": "FUSION_UI_CACHE",  # this app's netCDF result blobs
}

# Shot numbers are unique only within a machine, so every row the app writes
# carries it.
_DEFAULT_MACHINE = "cmod"


def __getattr__(name):
    if name in _ENV_PATHS:
        return os.path.expanduser(_require(_ENV_PATHS[name]))
    if name == "MACHINE":
        return os.environ.get("FUSION_MACHINE") or _DEFAULT_MACHINE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
