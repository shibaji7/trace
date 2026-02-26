"""hfpytrace package bootstrap."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from loguru import logger

_DEFAULT_CACHE_ROOT = Path.home() / ".hfpytrace"
_DEFAULT_GITHUB_ARCHIVES = (
    "https://codeload.github.com/shibaji7/trace/zip/refs/heads/main",
    "https://codeload.github.com/shibaji7/trace/zip/refs/heads/master",
)

CACHE_ROOT = Path(os.environ.get("HFPYTRACE_CACHE_DIR", _DEFAULT_CACHE_ROOT))
PHARLAP_LIB_PATH = CACHE_ROOT / "pharlap_lib"


def _has_pharlap_lib(path: Path) -> bool:
    return any(path.glob("pharlap_*"))


def _extract_pharlap_lib_from_archive(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        pharlap_prefix = None
        for member in zf.namelist():
            if member.endswith("pharlap_lib/rt_2D.m"):
                pharlap_prefix = member.removesuffix("rt_2D.m")
                break
        if pharlap_prefix is None:
            raise RuntimeError("Archive does not contain pharlap_lib/rt_2D.m")

        staged = destination.parent / f"{destination.name}.tmp"
        if staged.exists():
            shutil.rmtree(staged)
        staged.mkdir(parents=True, exist_ok=True)

        for member in zf.namelist():
            if not member.startswith(pharlap_prefix) or member.endswith("/"):
                continue
            rel_path = member[len(pharlap_prefix) :]
            if rel_path in {"rt_2D.m", "startup_2D.m"}:
                continue
            out_file = staged / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(zf.read(member))

        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)


def _download_pharlap_lib(destination: Path) -> None:
    urls = os.environ.get("HFPYTRACE_PHARLAP_ARCHIVE_URLS")
    candidates = tuple(u.strip() for u in urls.split(",")) if urls else _DEFAULT_GITHUB_ARCHIVES
    last_error = None

    for url in candidates:
        try:
            with tempfile.TemporaryDirectory(prefix="hfpytrace_") as td:
                archive = Path(td) / "trace.zip"
                with urllib.request.urlopen(url, timeout=60) as resp:
                    archive.write_bytes(resp.read())
                destination.parent.mkdir(parents=True, exist_ok=True)
                _extract_pharlap_lib_from_archive(archive, destination)
            logger.info(f"Downloaded pharlap_lib from {url} to {destination}")
            return
        except Exception as exc:  # pragma: no cover - network/platform specific
            last_error = exc

    raise RuntimeError(f"Unable to download pharlap_lib from GitHub: {last_error}")


def ensure_pharlap_lib() -> Path:
    if os.environ.get("HFPYTRACE_SKIP_PHARLAP_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        return PHARLAP_LIB_PATH
    if _has_pharlap_lib(PHARLAP_LIB_PATH):
        return PHARLAP_LIB_PATH

    _download_pharlap_lib(PHARLAP_LIB_PATH)
    return PHARLAP_LIB_PATH


try:
    ensure_pharlap_lib()
except Exception as exc:  # pragma: no cover - import must not hard-fail without network
    logger.warning(f"pharlap_lib bootstrap skipped: {exc}")
