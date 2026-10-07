"""Which code produced a result: the version reader, with git and without it.

The service runs as its own user while the checkouts belong to the maintainer,
and git refuses a repository another user owns unless ``safe.directory`` lists
it. So reading the commit straight from ``.git`` is a path the service may
really take, and it is tested here by making git refuse.
"""

import os
import shutil
import subprocess

import pytest

from fusion_ui.core import store, versions

SHA = "7b761fd792302d5592231cd6829b64e9f3209e68"
OTHER = "0e946cf07eada80506aebf93c2329032707858a9"


def fake_checkout(root, head="ref: refs/heads/main", loose=None, packed=None):
    """A ``.git`` directory with only the files the reader looks at."""
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text(head + "\n")
    for ref, sha in (loose or {}).items():
        (git_dir / ref).parent.mkdir(parents=True, exist_ok=True)
        (git_dir / ref).write_text(sha + "\n")
    if packed:
        lines = ["# pack-refs with: peeled fully-peeled sorted "]
        lines += [f"{sha} {ref}" for ref, sha in packed.items()]
        (git_dir / "packed-refs").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def git_refuses(monkeypatch):
    """git exits 128, as it does on a repository owned by another user."""
    calls = []

    def refuse(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args, 128, "", "fatal: detected dubious ownership in repository"
        )

    monkeypatch.setattr(versions.subprocess, "run", refuse)
    return calls


def test_a_loose_branch_is_read_without_git(tmp_path, git_refuses):
    root = fake_checkout(tmp_path, loose={"refs/heads/main": SHA})
    assert versions.describe(root) == SHA[:7]
    assert git_refuses, "git is tried first"


def test_a_branch_only_in_packed_refs_is_read(tmp_path, git_refuses):
    """How the server's checkouts keep remote branches, and main after a gc."""
    root = fake_checkout(
        tmp_path,
        packed={"refs/heads/main": SHA, "refs/remotes/origin/main": OTHER},
    )
    assert versions.head_commit(root) == SHA


def test_a_loose_ref_overrides_its_packed_copy(tmp_path, git_refuses):
    root = fake_checkout(
        tmp_path, loose={"refs/heads/main": SHA}, packed={"refs/heads/main": OTHER}
    )
    assert versions.head_commit(root) == SHA


def test_a_detached_head_is_the_commit(tmp_path, git_refuses):
    assert versions.head_commit(fake_checkout(tmp_path, head=SHA)) == SHA


def test_a_worktree_reads_its_own_head_and_the_shared_branches(tmp_path, git_refuses):
    """A ``.git`` file, a private ``HEAD``, branches in the common directory --
    what an agent's worktree, or a submodule, looks like."""
    main = fake_checkout(
        tmp_path / "main",
        loose={"refs/heads/feature": OTHER},
        packed={"refs/heads/main": SHA},
    )
    private = main / ".git" / "worktrees" / "wt"
    private.mkdir(parents=True)
    (private / "HEAD").write_text("ref: refs/heads/feature\n")
    (private / "commondir").write_text("../..\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {private}\n")

    assert versions.head_commit(worktree) == OTHER
    assert versions.head_commit(main) == SHA


def test_what_cannot_be_read_is_unknown(tmp_path, git_refuses):
    assert versions.describe(None) == versions.UNKNOWN
    # An unborn branch: HEAD names a ref nobody wrote yet.
    assert versions.describe(fake_checkout(tmp_path / "new")) == versions.UNKNOWN
    garbage = fake_checkout(tmp_path / "garbage", head="not a ref")
    assert versions.describe(garbage) == versions.UNKNOWN


def test_a_missing_git_binary_falls_back_too(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(versions.subprocess, "run", missing)
    root = fake_checkout(tmp_path, loose={"refs/heads/main": SHA})
    assert versions.describe(root) == SHA[:7]


# ---------------------------------------------------------------------------
# Against real git: both ways must name a commit identically, or a page
# comparing a stored version with the current one sees a change that is not.
# ---------------------------------------------------------------------------


def git(root, *args):
    """git in a throwaway repository, isolated from this user's own config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        HOME=str(root.parent),
        XDG_CONFIG_HOME=str(root.parent),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    return subprocess.run(
        ["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def make_repo(root, content):
    """A real repository with one commit of ``module.py``."""
    root.mkdir()
    git(root, "-c", "init.defaultBranch=main", "init", "-q")
    (root / "module.py").write_text(content)
    git(root, "add", "module.py")
    identity = ("-c", "user.name=test", "-c", "user.email=test@example.org")
    git(root, *identity, "commit", "-q", "-m", "one")
    return root


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    return make_repo(tmp_path / "checkout", "x = 1\n")


def test_git_and_the_files_name_the_same_commit(repo):
    head = git(repo, "rev-parse", "HEAD")
    assert versions.describe(repo) == head[:7]
    # The form `git describe` gave the stored rows, for a repository without
    # tags.
    assert versions.describe(repo) == git(repo, "describe", "--always", "--dirty")
    assert versions.head_commit(repo) == head

    git(repo, "pack-refs", "--all")
    assert not (repo / ".git" / "refs" / "heads" / "main").exists()
    assert versions.head_commit(repo) == head


def test_only_modified_tracked_files_make_it_dirty(repo):
    head = git(repo, "rev-parse", "HEAD")[:7]
    (repo / "untracked.txt").write_text("scratch\n")
    assert versions.describe(repo) == head
    (repo / "module.py").write_text("x = 2\n")
    assert versions.describe(repo) == f"{head}-dirty"
    assert versions.describe(repo) == git(repo, "describe", "--always", "--dirty")
    # The file reader cannot see it, and does not pretend to.
    assert versions.head_commit(repo)[:7] == head


def test_git_variables_from_a_hook_do_not_redirect_it(repo, monkeypatch, tmp_path):
    """A pre-commit hook running the suite exports GIT_DIR and GIT_INDEX_FILE;
    inherited, they would make every checkout report the hook's repository."""
    hook = make_repo(tmp_path / "hook", "y = 2\n")
    assert git(hook, "rev-parse", "HEAD") != git(repo, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(hook / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(hook / ".git" / "index"))
    assert versions.describe(repo) == git(repo, "rev-parse", "HEAD")[:7]


# ---------------------------------------------------------------------------
# Finding each checkout from where its code is imported
# ---------------------------------------------------------------------------


def test_a_checkout_is_found_from_the_package_it_provides(tmp_path, monkeypatch):
    root = fake_checkout(tmp_path / "scripts", loose={"refs/heads/main": SHA})
    package = root / "versions_probe_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(root))
    assert versions.checkout("versions_probe_pkg") == root.resolve()


def test_an_installed_copy_has_no_checkout(tmp_path, monkeypatch):
    """Under site-packages there is no repository of its own; walking further
    up would report whatever repository the virtualenv sits in."""
    enclosing = fake_checkout(tmp_path / "app", loose={"refs/heads/main": SHA})
    site = enclosing / ".venv" / "lib" / "python3.10" / "site-packages"
    package = site / "versions_installed_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(site))
    assert versions.checkout("versions_installed_pkg") is None


def test_an_unimportable_module_has_no_checkout():
    assert versions.checkout("no_such_module_anywhere") is None


def test_code_version_includes_fusion_scripts_even_when_git_refuses(
    tmp_path, monkeypatch, git_refuses
):
    """fusion_scripts' modules are top-level: its checkout is the directory
    holding the package it is found by (``decorrelation/`` for real)."""
    root = fake_checkout(tmp_path / "fusion_scripts", loose={"refs/heads/main": SHA})
    (root / "versions_probe_decorrelation").mkdir()
    (root / "versions_probe_decorrelation" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setitem(
        versions.REPOSITORIES, "fusion_scripts", "versions_probe_decorrelation"
    )

    assert versions.code_version()["fusion_scripts"] == SHA[:7]

    store._code_version.cache_clear()
    try:
        recorded = versions.parse(store._code_version())
    finally:
        store._code_version.cache_clear()
    assert recorded["fusion_scripts"] == SHA[:7]
    assert set(recorded) == {
        "fusion_ui",
        "imaging_methods",
        "fusion_scripts",
        "velocity_estimation",
    }


def test_the_installed_fusion_scripts_is_found_through_decorrelation():
    """On a machine set up as the README says: the editable checkout."""
    root = versions.checkout("decorrelation")
    if root is None:
        pytest.skip("fusion_scripts is not installed from a checkout here")
    assert (root / "decorrelation").is_dir()
    assert versions.code_version()["fusion_scripts"] != versions.UNKNOWN


# ---------------------------------------------------------------------------
# Reading a stored version back
# ---------------------------------------------------------------------------


def test_a_stored_version_reads_back_per_repository():
    text = versions.as_text(
        {"imaging_methods": "8a9bc07", "fusion_ui": "ce62a4d-dirty"}
    )
    assert text == "fusion_ui=ce62a4d-dirty imaging_methods=8a9bc07"
    assert versions.parse(text) == {
        "fusion_ui": "ce62a4d-dirty",
        "imaging_methods": "8a9bc07",
    }
    assert versions.parse("imported") == {} == versions.parse(None)


def test_commits_compare_without_the_dirty_mark():
    assert versions.commit("ce62a4d-dirty") == versions.commit("ce62a4d") == "ce62a4d"
    assert versions.commit("unknown") is None
    assert versions.commit(None) is None
