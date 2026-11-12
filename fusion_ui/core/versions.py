"""Which code produced a result: one commit per repository, read best effort.

``runs.code_version`` records it and the Fields page compares it with the
checkout's. It is never part of a cache key.

Each repository is found from where its code is imported, never from a
hard-coded path. A version is the commit's first seven hex digits, plus
``-dirty`` when git reports modified tracked files: what ``git describe
--always --dirty`` prints for a repository without tags. The commit is read

1. with ``git rev-parse HEAD``, then ``git status`` for ``-dirty``, or
2. from ``.git/HEAD`` and the ref it names, without running git. The service
   runs as its own user, and git refuses a repository that another user owns.
   Reading the files needs only read access. It cannot see uncommitted
   changes, so it never adds ``-dirty``.

Compare versions through :func:`commit`, which drops ``-dirty``.
"""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

#: What a repository's version reads when neither way finds a commit.
UNKNOWN = "unknown"

#: ``{name used in code_version: a module that repository provides}``. A stored
#: row may name other repositories; :func:`parse` reads any.
REPOSITORIES = {
    "fusion_ui": "fusion_ui",
    "imaging_methods": "imaging_methods",
    "velocity_estimation": "velocity_estimation",
    "experimental_database": "experimental_database",
}

_SHA = re.compile(r"[0-9a-f]{40}")


def code_version():
    """``{repository: version}`` for every entry of :data:`REPOSITORIES`."""
    return {name: describe(checkout(module)) for name, module in REPOSITORIES.items()}


def checkout(module):
    """The root of the git checkout ``module`` is imported from, or ``None``.

    A copy installed under ``site-packages`` has no checkout of its own, and
    gets ``None`` instead of the commit of whichever repository the virtualenv
    sits in.
    """
    try:
        spec = importlib.util.find_spec(module)
    except Exception:  # noqa: BLE001 - a broken install reads as unknown
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        start = Path(list(spec.submodule_search_locations)[0]).parent
    elif spec.origin and os.path.isfile(spec.origin):
        start = Path(spec.origin).parent
    else:
        return None
    start = start.resolve()
    for directory in (start, *start.parents):
        if directory.name in ("site-packages", "dist-packages"):
            return None
        if (directory / ".git").exists():
            return directory
    return None


def describe(root):
    """``abc1234``, ``abc1234-dirty`` or :data:`UNKNOWN` for the checkout at ``root``."""
    if root is None:
        return UNKNOWN
    head = _git(root, "rev-parse", "--verify", "--quiet", "HEAD")
    if head is not None and _SHA.fullmatch(head):
        # Tracked files only, as `git describe --dirty` counts them.
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        return head[:7] + ("-dirty" if status else "")
    head = head_commit(root)
    return head[:7] if head else UNKNOWN


def _git(root, *args):
    """``git <args>``'s stripped stdout in ``root``, or ``None`` if it failed."""
    # Without the GIT_* variables: a git hook that runs the test suite sets
    # GIT_DIR, which would make every checkout report the hook's repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # `git status` must not rewrite the index of a checkout another user owns.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def head_commit(root):
    """The full hash of the commit ``HEAD`` names, read from the files in ``.git``.

    Handles a detached ``HEAD``, a branch stored loose or in ``packed-refs``,
    and a worktree, whose ``.git`` is a file naming its own git directory and
    whose branches are in the main one. Anything else is ``None``.
    """
    git_dir = Path(root) / ".git"
    if not git_dir.is_dir():
        text = _read(git_dir) or ""
        if not text.startswith("gitdir:"):
            return None
        git_dir = git_dir.parent / text[len("gitdir:") :].strip()
    common_dir = git_dir / (_read(git_dir / "commondir") or "")

    value = _read(git_dir / "HEAD") or ""
    if value.startswith("ref:"):
        ref = value[len("ref:") :].strip()
        packed = (_read(common_dir / "packed-refs") or "").splitlines()
        value = (
            _read(git_dir / ref)
            or _read(common_dir / ref)
            or next((ln.split(" ")[0] for ln in packed if ln.endswith(f" {ref}")), "")
        )
    return value if _SHA.fullmatch(value) else None


def _read(path):
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def as_text(versions):
    """``{repository: version}`` as ``runs.code_version`` stores it."""
    return " ".join(f"{name}={value}" for name, value in sorted(versions.items()))


def parse(text):
    """``{repository: version}`` out of a stored ``runs.code_version``.

    The inverse of :func:`as_text`. ``None`` or any text without
    ``name=version`` pairs gives ``{}``.
    """
    pairs = {}
    for part in (text or "").split():
        name, sep, value = part.partition("=")
        if sep:
            pairs[name] = value
    return pairs


def commit(version):
    """The commit a version names, without ``-dirty``; ``None`` when unknown."""
    if not version or version == UNKNOWN:
        return None
    return version.removesuffix("-dirty")
