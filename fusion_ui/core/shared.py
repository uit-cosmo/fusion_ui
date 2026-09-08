"""Creating the state this app owns so a second user can write it too.

``CACHE_DIR`` and the SQLite file are shared state with two writers: the
Streamlit service, running as its own account, and ``fusion-ui precompute`` or
``rescan``, run by hand by whoever is filling the cache. Nothing coordinates
them, and nothing needs to -- SQLite is in WAL mode and a blob path carries its
own parameter hash -- but they have to be able to write each other's
directories.

By default they cannot. ``os.makedirs`` asks for mode 0o777 and the kernel
masks it with the process umask, so the usual 022 leaves a directory
owner-writable only; the first writer to reach a new ``runs/<plot>/<hash>/``
locks the second out of it with ``PermissionError`` -- *after* the analysis has
run, so the cost is paid and the result is thrown away. Passing ``mode=`` to
``makedirs`` does not help: mkdir(2) masks that argument too. ``chmod`` is not
masked, which is why the mode is set after the fact here.

**This widens the mode, never the group.** The setgid bit makes new entries
inherit the group of the directory above rather than the creator's own, so what
the operator sets once on the state directory (``chown -R fusionui:fusionui``
in ``deploy/install.sh``) is what propagates -- this module cannot and does not
choose who "the group" is.

Directories that already exist are left exactly as they are: one that is
deliberately private is not ours to open up.
"""

import os

#: setgid, so children inherit the state directory's group, plus group write.
DIR_MODE = 0o2775

#: Group-writable, so the other writer can replace a blob it recomputes.
FILE_MODE = 0o664


def _relax(path, mode):
    """``chmod``, ignoring a failure. Best effort by design: the file may
    belong to the other writer, and losing a computed result to a chmod that
    was only ever an optimisation would be the worse outcome."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def makedirs(path):
    """``os.makedirs(path, exist_ok=True)``, group-writable whatever the umask.

    Only the levels this call actually creates are chmod-ed.
    """
    missing = []
    probe = os.path.abspath(path)
    while not os.path.isdir(probe):
        missing.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:  # reached the root; it exists by definition
            break
        probe = parent

    os.makedirs(path, exist_ok=True)
    # Shallowest first, so a failure part-way still leaves the usable prefix.
    for created in reversed(missing):
        _relax(created, DIR_MODE)
    return path


def share_file(path):
    """Make one file group-writable. For a file this app has just written."""
    _relax(path, FILE_MODE)
    return path
