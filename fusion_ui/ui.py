"""Helpers shared by the entry point and the pages.

Streamlit executes ``app.py`` and every file in ``pages/`` as a script, so a
page cannot import another one. Anything two of them need lives here.
"""

import os
import threading

import streamlit as st

from fusion_ui import config
from fusion_ui.core import catalog, db, rundays, versions

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
    connection may only be used on the thread that created it, so one
    connection cached for the whole process breaks when a second browser tab
    connects.
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
    """The commit of each repository a result depends on
    (:func:`fusion_ui.core.versions.code_version`)."""
    return versions.code_version()


@st.cache_data(show_spinner="Reading the shot index…")
def shot_table(fingerprint, discharge_db_path, discharge_db_mtime):
    """Cached :func:`fusion_ui.core.catalog.shot_table`.

    ``fingerprint`` and ``discharge_db_mtime`` are cache keys only: the table
    is read again when the index or the discharge DB changes.
    """
    return catalog.shot_table(get_connection(), discharge_db_path)


def discharge_db():
    """``(path, mtime)`` of the discharge DB, or ``(None, 0.0)`` without one.

    The mtime is for cache keys, so that an edited DB shows without a restart.
    """
    path, error = resolve("DISCHARGE_DB_PATH")
    if error or not os.path.exists(path or ""):
        return None, 0.0
    return path, os.path.getmtime(path)


def discharge(shot):
    """The shot's entry in the discharge DB, or ``None``."""
    path, _ = discharge_db()
    return catalog.load_discharges(path).get(shot) if path else None


def cached_shot_table():
    """The browser table, with the cache keys worked out for the caller."""
    path, mtime = discharge_db()
    return shot_table(catalog.index_fingerprint(get_connection()), path, mtime)


def available_targets(row, diagnostics=catalog.DIAGNOSTICS):
    """``{diagnostic: (has_raw, has_preprocessed)}`` for one shot-table row,
    over the ``diagnostics`` it has a file of."""
    targets = {}
    for diagnostic in diagnostics:
        availability = row[diagnostic]
        if availability:
            targets[diagnostic] = ("R" in availability, "P" in availability)
    return targets


def human_bytes(value):
    """A byte count for reading; ``""`` for none."""
    if not value:
        return ""
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024


@st.cache_data(show_spinner=False)
def run_day_table(fingerprint, discharge_db_path, discharge_db_mtime, run_days_mtime):
    """Cached :func:`fusion_ui.core.rundays.run_day_table`.

    ``fingerprint`` and the two mtimes are cache keys only, so an edit of
    ``run_days.md`` shows on the next rerun.
    """
    return rundays.run_day_table(get_connection(), discharge_db_path)


def cached_run_day_table():
    """The run-day overview, with the cache keys worked out for the caller."""
    path, mtime = discharge_db()
    run_days = rundays.RUN_DAYS_PATH
    return run_day_table(
        catalog.index_fingerprint(get_connection()),
        path,
        mtime,
        os.path.getmtime(run_days) if os.path.exists(run_days) else 0.0,
    )
