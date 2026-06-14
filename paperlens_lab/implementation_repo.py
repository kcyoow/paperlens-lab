from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .ingest import clean_text

Runner = Callable[..., subprocess.CompletedProcess[str]]

DEFAULT_REPO_ROOT = Path(os.getenv("PAPERLENS_IMPLEMENTATION_REPO_DIR", "outputs/implementation_repos"))
MAX_MANIFEST_FILES = 250
MAX_TOTAL_BYTES = 25_000_000
MAX_TEXT_CHARS = 2200


def inspect_implementation_repositories(
    repositories: list[dict[str, Any]] | None,
    *,
    base_dir: Path | str = DEFAULT_REPO_ROOT,
    timeout_seconds: int = 45,
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for repository in repositories or []:
        manifest = inspect_implementation_repository(
            repository,
            base_dir=base_dir,
            timeout_seconds=timeout_seconds,
            runner=runner,
        )
        manifests.append(manifest)
    return manifests


def inspect_implementation_repository(
    repository: dict[str, Any],
    *,
    base_dir: Path | str = DEFAULT_REPO_ROOT,
    timeout_seconds: int = 45,
    runner: Runner | None = None,
) -> dict[str, Any]:
    url = clean_text(str(repository.get("url") or ""))
    source_url = clean_text(str(repository.get("source_url") or url))
    usage = clean_text(str(repository.get("usage") or "source-listed implementation repository"))
    source_id = clean_text(str(repository.get("source_id") or "implementation:github:1"))
    base_manifest = {
        "source_id": source_id,
        "url": url,
        "source_url": source_url,
        "host": "github.com",
        "usage": usage,
        "execution": "none",
        "status": "unavailable",
        "commit": "",
        "default_branch": "",
        "file_count": 0,
        "total_bytes": 0,
        "truncated": False,
        "files": [],
        "readme": None,
        "license": None,
        "error": "",
    }
    if not _valid_github_repo_root(url):
        return {**base_manifest, "status": "rejected", "error": "repository URL is not an approved GitHub repo root"}

    destination = Path(base_dir) / _repo_dir_name(url)
    runner = runner or subprocess.run
    try:
        _ensure_cloned(url, destination, timeout_seconds=timeout_seconds, runner=runner)
        commit = _git_text(["git", "-C", str(destination), "rev-parse", "HEAD"], timeout_seconds, runner)
        branch = _git_text(
            ["git", "-C", str(destination), "rev-parse", "--abbrev-ref", "HEAD"],
            timeout_seconds,
            runner,
        )
        inventory = _repo_inventory(destination)
        return {
            **base_manifest,
            "status": "inspected",
            "commit": commit,
            "default_branch": "" if branch == "HEAD" else branch,
            **inventory,
        }
    except Exception as exc:
        return {**base_manifest, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def _valid_github_repo_root(url: str) -> bool:
    return bool(re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", url))


def _repo_dir_name(url: str) -> str:
    digest = hashlib.sha1(url.lower().encode("utf-8")).hexdigest()[:12]
    owner_repo = url.removeprefix("https://github.com/").replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_repo).strip("_")
    return f"{safe}-{digest}"


def _ensure_cloned(url: str, destination: Path, *, timeout_seconds: int, runner: Runner) -> None:
    if (destination / ".git").exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.parent / f".tmp-{destination.name}-{int(time.time() * 1000)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    env = {
        **os.environ,
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--no-tags",
        "--single-branch",
        "--filter=blob:limit=1048576",
        "--no-recurse-submodules",
        url,
        str(tmp),
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=timeout_seconds, env=env)
        if result.returncode != 0:
            stderr = clean_text(result.stderr or result.stdout or "git clone failed")
            raise RuntimeError(stderr[:500])
        if destination.exists():
            shutil.rmtree(tmp)
        else:
            tmp.rename(destination)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def _git_text(command: list[str], timeout_seconds: int, runner: Runner) -> str:
    result = runner(command, capture_output=True, text=True, timeout=timeout_seconds)
    if result.returncode != 0:
        return ""
    return clean_text(result.stdout)


def _repo_inventory(path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    truncated = False
    readme = None
    license_info = None
    for file_path in sorted(path.rglob("*")):
        if ".git" in file_path.parts or not file_path.is_file() or file_path.is_symlink():
            continue
        relative = file_path.relative_to(path).as_posix()
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        total_bytes += size
        if len(files) < MAX_MANIFEST_FILES:
            files.append({"path": relative, "bytes": size, "kind": _file_kind(relative)})
        else:
            truncated = True
        if total_bytes > MAX_TOTAL_BYTES:
            truncated = True
            break
        lower_name = Path(relative).name.lower()
        if readme is None and lower_name.startswith("readme"):
            readme = _text_excerpt(file_path)
        if license_info is None and (lower_name.startswith("license") or lower_name.startswith("copying")):
            license_info = _text_excerpt(file_path)
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "truncated": truncated,
        "files": files,
        "readme": readme,
        "license": license_info,
    }


def _file_kind(relative: str) -> str:
    name = Path(relative).name.lower()
    if name.startswith("readme"):
        return "readme"
    if name.startswith("license") or name.startswith("copying"):
        return "license"
    if name in {"requirements.txt", "pyproject.toml", "setup.py", "environment.yml", "environment.yaml"}:
        return "dependency"
    if name in {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}:
        return "container"
    if relative.lower().startswith(("examples/", "example/", "demo/", "demos/", "notebooks/")):
        return "example"
    return "source"


def _text_excerpt(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    return {
        "path": path.name,
        "excerpt": clean_text(text[:MAX_TEXT_CHARS]),
    }
