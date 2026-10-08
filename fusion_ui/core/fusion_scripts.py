"""fusion_scripts' ``config``, under the bare name its own code imports it by.

fusion_scripts reads its settings with a bare ``import config``, in some
seventy files: ``twodca_manuscript/datasets/cmod.py``, which
``decorrelation.pipeline`` imports, and ``plotting_scripts`` among them. This
app's settings are a ``config.py`` too, harmless as ``fusion_ui.config``, but
the bare name finds it as soon as the package directory is on ``sys.path``.

Under Streamlit it always is: ``streamlit run fusion_ui/app.py`` puts the main
script's directory, this package's, first on ``sys.path`` when it starts
(``bootstrap._fix_sys_path``) and again around every script run
(``exec_code.modified_sys_path``), so taking it out once does not hold. And
fusion_scripts is installed editable, with setuptools' editable finder after
``PathFinder`` on ``sys.meta_path``: ``PathFinder`` finds
``fusion_ui/config.py`` first, and fusion_scripts fails on the first setting
only it has (``module 'config' has no attribute 'W7X_PRESENTATION_DIR'``). The
CLI, pytest and ``AppTest`` on a page file never put that directory on the
path.

:func:`import_config` imports fusion_scripts' module once, with the directory
left out. From then on it is the ``config`` in ``sys.modules``, which every
later ``import config`` gets, the lazy ones inside a Streamlit run included.
Call it before importing fusion_scripts code that reads ``config``.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

# First, so that fusion_ui's .env wins: importing fusion_scripts' config fills
# every FUSION_* variable still unset from fusion_scripts' own .env.
import fusion_ui.config  # noqa: F401 - imported for what it does to os.environ

#: This package's directory, which Streamlit puts on ``sys.path``.
PACKAGE = Path(__file__).resolve().parent.parent

#: The fusion_scripts package that fusion_scripts' ``config.py`` sits beside.
#: fusion_scripts has no package of its own, and :mod:`.versions` finds its
#: checkout by the same one.
ANCHOR = "decorrelation"


def _resolved(path):
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _in_package(path):
    return path is not None and (path == PACKAGE or PACKAGE in path.parents)


def _origin(module):
    """The file ``module`` was loaded from, resolved; ``None`` if it has none."""
    file = getattr(module, "__file__", None)
    return _resolved(file) if file else None


def _home():
    """The directory holding the :data:`ANCHOR` package; ``None`` if not found."""
    try:
        spec = importlib.util.find_spec(ANCHOR)
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or ()) if spec else []
    found = _resolved(locations[0]) if locations else None
    return found.parent if found else None


def _is_fusion_scripts(module):
    origin = _origin(module)
    return origin is not None and origin.parent == _home()


def import_config():
    """fusion_scripts' ``config`` module, made the one a bare ``import config`` gets.

    Returns at once when it already is. A ``config`` loaded from this package
    (``fusion_ui/config.py`` under the bare name) is dropped from
    ``sys.modules`` and imported again with the package left out of
    ``sys.path``, which is then put back exactly as it was. Raises
    ``ImportError`` when what that gives is not fusion_scripts' module.
    """
    module = sys.modules.get("config")
    if module is not None and _is_fusion_scripts(module):
        return module
    if module is not None and _in_package(_origin(module)):
        del sys.modules["config"]

    # A str entry names a directory ('' the current one); anything else is
    # kept as it is, since the path finder ignores it anyway. Streamlit runs
    # sessions on threads, but the only entry another thread adds or drops
    # meanwhile is this package's, which nothing imports through.
    saved = sys.path[:]
    sys.path[:] = [
        entry
        for entry in saved
        if not (isinstance(entry, str) and _in_package(_resolved(entry)))
    ]
    try:
        module = importlib.import_module("config")
    except ModuleNotFoundError as error:
        if error.name != "config":
            raise
        module = None
    finally:
        sys.path[:] = saved

    if module is None or not _is_fusion_scripts(module):
        home = _home()
        found = (
            "cannot be imported"
            if module is None
            else f"imports {_origin(module) or module}"
        )
        wanted = (
            home / "config.py"
            if home
            else f"the config.py beside its {ANCHOR} package, which cannot be"
            " imported either"
        )
        raise ImportError(
            f"The bare name 'config' {found}, where fusion_scripts' configuration"
            f" is wanted: {wanted}. Is fusion_scripts installed in this"
            " environment, or is another config module ahead of it on sys.path?"
        )
    return module
