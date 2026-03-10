#!/usr/bin/env python3
"""Fetch and verify PHaRLAP support files used by hfpytrace.pharlap."""

import sys
from pathlib import Path

# Ensure local project package is imported instead of stdlib `trace`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfpytrace import ensure_pharlap_lib
from hfpytrace.pharlap import get_matlab_pharlap_lib


def main() -> None:
    cache_root = ensure_pharlap_lib()
    matlab_lib = get_matlab_pharlap_lib(trace_spec=Path(cache_root))
    print(f"PHARLAP cache root: {cache_root}")
    print(f"MATLAB PHARLAP path: {matlab_lib}")


if __name__ == "__main__":
    main()
