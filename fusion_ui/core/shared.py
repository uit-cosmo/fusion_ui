"""Creating the state this app owns so that a second account can write it too.

``CACHE_DIR`` and the SQLite file have two writers: the Streamlit service, and
whoever runs ``fusion-ui precompute`` or ``rescan`` by hand. Each must be able
to write the other's directories. ``os.makedirs`` cannot arrange that, since
the umask masks its ``mode`` argument, so the mode is set afterwards with
``chmod``, which is not masked.

Only the mode is widened, never the group: the setgid bit makes new entries
inherit the group the operator set on the state directory
(``deploy/install.sh``). A directory that already exists is left as it is.
"""

import os

#: setgid, so children inherit the state directory's group, plus group write.
DIR_MODE = 0o2775

#: Group-writable, so the other writer can replace a blob it recomputes.
FILE_MODE = 0o664


def _relax(path, mode):
    """``chmod``, ignoring a failure: the file may belong to the other writer."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def makedirs(path):
    """``os.makedirs(path, exist_ok=True)``, group-writable whatever the umask.

    Only the levels this call creates are chmod-ed.
    """
    missing = []
    probe = os.path.abspath(path)
    while not os.path.isdir(probe):
        missing.append(probe)
        parent = os.path.dirname(probe)
        if parent == probe:  # the root
            break
        probe = parent

    os.makedirs(path, exist_ok=True)
    for created in reversed(missing):
        _relax(created, DIR_MODE)
    return path


def share_file(path):
    """Make a file this app has just written group-writable."""
    _relax(path, FILE_MODE)
    return path
