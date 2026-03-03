import os
import sys
from pathlib import Path

# Prefer local package over stdlib `trace` during test collection.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Disable network bootstrap during imports.
os.environ.setdefault("HFPYTRACE_SKIP_PHARLAP_DOWNLOAD", "1")
