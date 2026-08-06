"""Release-script safety checks that run before any external publication."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "release.sh"
VERSION_FILE = Path("src/roomkit/_version.py")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _write(repo: Path, path: Path, content: str) -> None:
    destination = repo / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "release-test@example.test")
    _git(repo, "config", "user.name", "Release Test")
    _write(repo, VERSION_FILE, '__version__ = "1.2.0.dev0"\n')
    _write(repo, Path("CHANGELOG.md"), "## [1.2.3]\n")
    _commit(repo, "Initial development state")
    return repo


def _tag_release(repo: Path, *, with_code_change: bool = False) -> None:
    _write(repo, VERSION_FILE, '__version__ = "1.2.3"\n')
    if with_code_change:
        _write(repo, Path("src/roomkit/code.py"), "changed = True\n")
    _commit(repo, "Bump version to 1.2.3")
    _git(repo, "tag", "v1.2.3")


def _run(
    repo: Path,
    version: str = "1.2.3",
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LC_ALL": "C", **(extra_env or {})}
    return subprocess.run(
        ["bash", str(SCRIPT), version],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_executable(repo: Path, path: Path, content: str) -> None:
    _write(repo, path, content)
    (repo / path).chmod(0o755)


def _stub_release_tools(repo: Path) -> dict[str, str]:
    """Provide deterministic local substitutes for publication dependencies."""
    bin_dir = repo.parent / "release-test-bin"
    bin_dir.mkdir()
    _write_executable(
        repo.parent,
        Path("release-test-bin/curl"),
        "#!/usr/bin/env bash\nexit 22\n",
    )
    _write_executable(
        repo.parent,
        Path("release-test-bin/gh"),
        """#!/usr/bin/env bash
if [[ "$1" == "run" && "$2" == "list" ]]; then
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--commit" ]]; then
            sha="$2"
            break
        fi
        shift
    done
    printf '{"status":"completed","conclusion":"success","headSha":"%s"}\n' "$sha"
    exit 0
fi
if [[ "$1" == "release" && "$2" == "view" ]]; then
    exit 0
fi
exit 1
""",
    )
    _write_executable(
        repo.parent,
        Path("release-test-bin/uv"),
        """#!/usr/bin/env bash
case "$1" in
    run|publish)
        exit 0
        ;;
    build)
        version=$(sed -n 's/^__version__ = "\\(.*\\)"/\\1/p' src/roomkit/_version.py)
        mkdir -p dist
        : > "dist/roomkit-${version}.tar.gz"
        : > "dist/roomkit-${version}-py3-none-any.whl"
        exit 0
        ;;
    export)
        printf '# empty test requirements\n'
        exit 0
        ;;
esac
exit 1
""",
    )
    _write_executable(
        repo.parent,
        Path("release-test-bin/uvx"),
        """#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        output="$2"
        break
    fi
    shift
done
mkdir -p "$(dirname "$output")"
printf '{}\n' > "$output"
""",
    )
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UV_PUBLISH_TOKEN": "test-token",
    }


def test_resume_rejects_a_stale_tag(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    _tag_release(repo)
    _write(repo, VERSION_FILE, '__version__ = "1.3.0.dev0"\n')
    _commit(repo, "Open 1.3 development")
    _write(repo, Path("later.py"), "later = True\n")
    _commit(repo, "Later code change")

    result = _run(repo)

    assert result.returncode != 0
    assert "must point to HEAD or its direct parent" in result.stdout


def test_resume_rejects_a_tag_on_a_code_bearing_commit(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    _tag_release(repo, with_code_change=True)

    result = _run(repo)

    assert result.returncode != 0
    assert "must point to a version-only commit" in result.stdout


def test_resume_accepts_the_direct_next_dev_commit(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    _tag_release(repo)
    _write(repo, VERSION_FILE, '__version__ = "1.3.0.dev0"\n')
    _commit(repo, "Open 1.3 development")

    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "tag", "local-only")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "already released (HEAD on 1.3.0.dev0)" in result.stdout
    assert _git(remote, "show-ref", "--verify", "refs/tags/v1.2.3").returncode == 0
    assert "local-only" not in _git(remote, "tag").stdout.splitlines()


def test_resume_after_version_commit_before_tag(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "tag", "local-only")
    code_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # State left by an interruption after `git commit` and before `git tag`.
    _write(repo, VERSION_FILE, '__version__ = "1.2.3"\n')
    _commit(repo, "Bump version to 1.2.3")
    release_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = _run(repo, extra_env=_stub_release_tools(repo))

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"CI is green for {code_sha}." in result.stdout
    assert _git(repo, "rev-list", "-n", "1", "v1.2.3").stdout.strip() == release_sha
    assert "local-only" not in _git(remote, "tag").stdout.splitlines()
    assert "1.3.0.dev0" in (repo / VERSION_FILE).read_text(encoding="utf-8")


def test_rejects_hyphenated_semver_prerelease_before_mutation(tmp_path: Path) -> None:
    repo = _repository(tmp_path)

    result = _run(repo, "1.2.3-rc.1")

    assert result.returncode != 0
    assert "PEP 440 release" in result.stdout
    assert _git(repo, "status", "--porcelain").stdout == ""
