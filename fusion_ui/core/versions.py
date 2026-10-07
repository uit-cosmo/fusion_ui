"""Which code produced a result: one commit per repository, read best effort.

``runs.code_version`` records it and every figure shows it, because a result
computed by last month's code is the trap that makes people distrust the tool.
It is never part of a cache key (see :mod:`fusion_ui.core.store`).

Four repositories decide what a result is: this app, ``imaging_methods`` (the
2DCA), ``fusion_scripts`` (the decorrelation API) and ``velocity_estimation``
(the TDE). Each is found from where its code is imported from -- the editable
checkout, or a worktree shadowing it from the current directory -- never from
a hard-coded path. fusion_scripts has no package of its own (its modules are
top-level), so its checkout is the directory holding ``decorrelation/``.

A version is the commit's first seven hex digits, plus ``-dirty`` when git
reports modified tracked files. That is what ``git describe --always --dirty``
prints for a repository without tags, which is the form the stored rows have
for fusion_ui and imaging_methods. The form is fixed on purpose: the same
commit must read the same however it was found, because a page compares a
stored version with the current one. ``git describe --tags`` would not do --
fusion_scripts carries a tag, and would read ``<tag>-12-g7b761fd`` through git
but ``7b761fd`` through the files.

Two ways to find the commit, in that order:

1. ``git rev-parse HEAD``, then ``git status`` for ``-dirty``. Run with
   optional locks off: ``git describe --dirty``, run by the service, rewrote
   the index of a checkout that belongs to the maintainer.
2. ``.git/HEAD`` and the ref it names -- a loose ref file, then
   ``packed-refs`` -- read without running git. The service runs as its own
   user while the checkouts belong to the maintainer, and git refuses a
   repository owned by another user unless ``safe.directory`` lists it.
   Reading the files needs only read access, which the group-readable
   checkouts give. It cannot see uncommitted changes, so it never adds
   ``-dirty``.
"""

import importlib.util
import os
import re
import subprocess
from pathlib import Path

#: What a repository's version reads when neither way finds a commit.
UNKNOWN = "unknown"

#: ``{name used in code_version: a module that repository provides}``.
REPOSITORIES = {
    "fusion_ui": "fusion_ui",
    "imaging_methods": "imaging_methods",
    "fusion_scripts": "decorrelation",
    "velocity_estimation": "velocity_estimation",
}

#: Hex digits kept of a commit: git's own default abbreviation.
ABBREV = 7

_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")

#: Variables that point git at some other repository than the one in ``cwd``
#: (``git rev-parse --local-env-vars``). A git hook that runs the test suite
#: sets some of them; inherited, they would make every checkout report the
#: hook's repository.
_REPOSITORY_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_INTERNAL_SUPER_PREFIX",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)


def code_version():
    """``{repository: version}`` for every entry of :data:`REPOSITORIES`."""
    return {name: describe(checkout(module)) for name, module in REPOSITORIES.items()}


def checkout(module):
    """The root of the git checkout ``module`` is imported from, or ``None``.

    Located with ``importlib.util.find_spec``, which hands back an imported
    module's own spec and otherwise finds the module without running it. The
    walk goes up from the directory holding the package to the first one with
    a ``.git`` (a directory, or a file in a worktree). A copy installed under
    ``site-packages`` has no checkout, and gets ``None`` rather than the commit
    of whichever repository the virtualenv sits in.
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
        # Tracked files only, as `git describe --dirty` counts them. A failed
        # status (None) cannot tell, so it adds nothing.
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        return head[:ABBREV] + ("-dirty" if status else "")
    head = head_commit(root)
    return head[:ABBREV] if head else UNKNOWN


def _git(root, *args):
    """``git <args>``'s stripped stdout in ``root``, or ``None`` if it failed."""
    env = {k: v for k, v in os.environ.items() if k not in _REPOSITORY_ENV}
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


# ---------------------------------------------------------------------------
# Reading the commit without git
# ---------------------------------------------------------------------------


def head_commit(root):
    """The full hash of the commit ``HEAD`` names, read from the files in ``.git``.

    Handles a ``.git`` directory and a ``.git`` file (a worktree or submodule:
    ``gitdir: <path>``, with branches shared through its ``commondir``), a
    detached ``HEAD``, and a branch stored loose or in ``packed-refs``.
    Anything else -- a reftable repository, an unborn branch, a file that
    cannot be read -- is ``None``.
    """
    git_dir = _git_dir(Path(root))
    if git_dir is None:
        return None
    common_dir = _common_dir(git_dir)
    value = _read(git_dir / "HEAD")
    for _ in range(5):  # a symbolic ref may name another symbolic ref
        if value is None:
            return None
        if _SHA.fullmatch(value):
            return value
        if not value.startswith("ref:"):
            return None
        value = _resolve(git_dir, common_dir, value[len("ref:") :].strip())
    return None


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def _git_dir(root):
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    text = _read(dot_git)
    if text and text.startswith("gitdir:"):
        path = Path(text[len("gitdir:") :].strip())
        return path if path.is_absolute() else dot_git.parent / path
    return None


def _common_dir(git_dir):
    """Where a worktree's branches, tags and ``packed-refs`` live."""
    text = _read(git_dir / "commondir")
    if not text:
        return git_dir
    path = Path(text)
    return path if path.is_absolute() else git_dir / path


def _resolve(git_dir, common_dir, ref):
    # Per-worktree refs (refs/bisect, refs/worktree) sit in the worktree's own
    # git dir, branches and tags in the common one; packed refs only there.
    for base in (git_dir, common_dir):
        value = _read(base / ref)
        if value:
            return value
    for line in (_read(common_dir / "packed-refs") or "").splitlines():
        if line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref:
            return sha.strip()
    return None


# ---------------------------------------------------------------------------
# Reading a stored version back
# ---------------------------------------------------------------------------


def as_text(versions):
    """``{repository: version}`` as ``runs.code_version`` stores it."""
    return " ".join(f"{name}={value}" for name, value in sorted(versions.items()))


def parse(text):
    """``{repository: version}`` out of a stored ``runs.code_version``.

    The inverse of :func:`as_text`. ``None``, ``"imported"`` (the seeded rows)
    or any other text without ``name=version`` pairs gives ``{}``.
    """
    pairs = {}
    for part in (text or "").split():
        name, sep, value = part.partition("=")
        if sep:
            pairs[name] = value
    return pairs


def commit(version):
    """The commit a version names, without ``-dirty``; ``None`` when unknown.

    Compare these, not the versions: the service may read a checkout through
    its files, which never says ``-dirty``, while the CLI reads it through git.
    """
    if not version or version == UNKNOWN:
        return None
    return version[: -len("-dirty")] if version.endswith("-dirty") else version
