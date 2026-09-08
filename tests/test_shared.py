"""Group-writable creation of the state the app owns.

The failure these guard against is not local: the Streamlit service and a
hand-run ``fusion-ui precompute`` are different accounts writing one cache, and
a directory created under the default umask locks the second one out *after* it
has paid for the analysis.
"""

import os
import stat

import pytest

from fusion_ui.core import db, shared, store


@pytest.fixture
def strict_umask():
    """Pin the umask to the value that causes the bug, whatever the developer's
    shell or the CI runner has set."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_makedirs_beats_the_umask(tmp_path, strict_umask):
    target = tmp_path / "runs" / "velocity_field" / "abc123"
    shared.makedirs(target)

    assert target.is_dir()
    for level in (target, target.parent, target.parent.parent):
        assert mode(level) == shared.DIR_MODE, level
        assert mode(level) & stat.S_IWGRP, f"{level} is not group-writable"


def test_makedirs_leaves_an_existing_directory_alone(tmp_path, strict_umask):
    """A directory someone deliberately made private is not ours to open up."""
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)

    shared.makedirs(private / "child")

    assert mode(private) == 0o700
    assert mode(private / "child") == shared.DIR_MODE


def test_makedirs_is_idempotent(tmp_path, strict_umask):
    target = tmp_path / "a" / "b"
    shared.makedirs(target)
    shared.makedirs(target)
    assert mode(target) == shared.DIR_MODE


def test_share_file_is_group_writable(tmp_path, strict_umask):
    path = tmp_path / "blob.nc"
    path.write_bytes(b"")
    assert not mode(path) & stat.S_IWGRP  # what the umask gave us

    shared.share_file(path)
    assert mode(path) == shared.FILE_MODE


def test_share_file_survives_a_path_it_cannot_chmod(tmp_path):
    """Best effort: a blob written by the other account must not raise here."""
    shared.share_file(tmp_path / "does-not-exist.nc")


def test_blob_and_its_directories_are_group_writable(
    tmp_path, monkeypatch, strict_umask
):
    """The end-to-end shape of the server failure: `runs/<plot>/<hash>/` created
    by one writer, a blob written into it by the other."""
    import xarray as xr

    from fusion_ui import config
    from fusion_ui.core import registry

    monkeypatch.setenv("FUSION_UI_CACHE", str(tmp_path / "cache"))
    target = registry.Target(
        machine="cmod",
        shot=1160616027,
        diagnostic="apd",
        preprocessed=True,
        path="/nowhere.nc",
        t_start=0.0,
        t_end=1.0,
        window_source="metadata",
    )
    path = store.blob_path("velocity_field", "abc123", target)
    store._write_blob(
        xr.Dataset({"v": 1.0}), path, "velocity_field", "abc123", "{}", None, "now"
    )

    assert mode(path) == shared.FILE_MODE
    assert mode(os.path.dirname(path)) == shared.DIR_MODE
    assert config.CACHE_DIR  # the fixture really did redirect it


def test_database_file_is_group_writable(tmp_path, strict_umask):
    path = tmp_path / "state" / "shot_explorer.sqlite"
    conn = db.connect(path)
    conn.close()

    assert mode(path) == shared.FILE_MODE
    assert mode(path.parent) == shared.DIR_MODE
    # SQLite copies the database's mode onto its sidecars, so relaxing the one
    # file is what makes WAL usable by the second writer.
    for sidecar in (f"{path}-wal", f"{path}-shm"):
        if os.path.exists(sidecar):
            assert mode(sidecar) & stat.S_IWGRP, sidecar
