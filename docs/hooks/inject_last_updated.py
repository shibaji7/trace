from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def _format_date(date_str: str) -> str:
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").strftime("%B %-d, %Y")


def _resolve_last_updated(repo_root: Path, target: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%cs", "--", target],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if value:
            return _format_date(value)
    except Exception:
        pass

    file_path = repo_root / target
    return datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%B %-d, %Y")


def on_page_markdown(markdown: str, page, config, files):
    if page.file.src_path != "index.md":
        return markdown

    repo_root = Path(config.config_file_path).resolve().parent
    last_updated = _resolve_last_updated(repo_root, "docs/index.md")
    return markdown.replace("{{ INDEX_LAST_UPDATED }}", last_updated)
