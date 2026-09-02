"""Helpers shared by the entry point and the pages.

Streamlit executes ``app.py`` and every file in ``pages/`` as a script, so a
page cannot import the entry point to reuse something -- it would run the whole
landing page. Anything two of them need lives here instead.
"""

import os
import subprocess
import threading

import streamlit as st

from fusion_ui import config
from fusion_ui.core import catalog, db

_local = threading.local()


@st.cache_resource
def _schema_ready():
    """Create or migrate the app database once per server process."""
    connection = db.open_db()
    connection.close()
    return True


def get_connection():
    """A connection private to the calling thread.

    Streamlit runs each session's script on a pooled thread, and a sqlite3
    connection may only be used on the thread that created it -- so one cached
    connection shared process-wide breaks the moment a second browser tab
    connects. WAL is what makes several of them (and the ``rescan`` cron job)
    coexist without blocking each other on reads.
    """
    _schema_ready()
    connection = getattr(_local, "connection", None)
    if connection is None:
        connection = _local.connection = db.connect()
    return connection


def resolve(attribute):
    """``(value, error)`` for a lazily-resolved config path."""
    try:
        return getattr(config, attribute), None
    except RuntimeError as error:
        return None, str(error)


@st.cache_data(show_spinner=False)
def code_version():
    """``git describe`` for this app and for imaging_methods, best effort.

    Shown under every figure from phase 02 on: a result computed by last
    month's imaging_methods is the trap that makes people distrust the tool.
    """
    return {
        "fusion_ui": _git_describe(config.REPO_ROOT),
        "imaging_methods": _git_describe(_package_root("imaging_methods")),
    }


def _package_root(name):
    try:
        module = __import__(name)
    except Exception:
        return None
    path = getattr(module, "__file__", None)
    return os.path.dirname(os.path.dirname(path)) if path else None


def _git_describe(path):
    if not path or not os.path.isdir(str(path)):
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


@st.cache_data(show_spinner="Reading the shot index…")
def shot_table(fingerprint, discharge_db_path, discharge_db_mtime):
    """Cached :func:`fusion_ui.core.catalog.shot_table`.

    ``fingerprint`` and ``discharge_db_mtime`` are not used in the body: they
    are in the signature so the cache invalidates when the index or the curated
    descriptor changes, which is what makes the Rescan button visibly work.
    """
    return catalog.shot_table(get_connection(), discharge_db_path)


def cached_shot_table():
    """The browser table, with the cache keys worked out for the caller."""
    discharge_db_path, error = resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(discharge_db_path or ""):
        discharge_db_path, mtime = None, 0.0
    else:
        mtime = os.path.getmtime(discharge_db_path)
    return shot_table(
        catalog.index_fingerprint(get_connection()), discharge_db_path, mtime
    )
