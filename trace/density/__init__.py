"""Density model namespace.

Lazy imports keep optional/heavy dependencies from loading at package import time.
"""

from __future__ import annotations

from importlib import import_module

from loguru import logger

__all__ = [
    "IRI2d",
    "IRI3d",
    "GEMINI2d",
    "GITM2d",
    "SAMI3",
    "WACCMX2d",
    "WAMIPE2d",
]


def __getattr__(name: str):
    module_map = {
        "IRI2d": ("trace.density.iri", "IRI2d"),
        "IRI3d": ("trace.density.iri", "IRI3d"),
        "GEMINI2d": ("trace.density.gemini", "GEMINI2d"),
        "GITM2d": ("trace.density.gitm", "GITM2d"),
        "SAMI3": ("trace.density.sami", "SAMI3"),
        "WACCMX2d": ("trace.density.waccm", "WACCMX2d"),
        "WAMIPE2d": ("trace.density.wamipe", "WAMIPE2d"),
    }
    if name not in module_map:
        raise AttributeError(f"module 'trace.density' has no attribute {name!r}")
    mod_name, attr = module_map[name]
    try:
        logger.debug("Lazy-loading density model: {} from {}", attr, mod_name)
        mod = import_module(mod_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Optional dependency missing while loading '{name}'. "
            f"Install model dependencies and retry."
        ) from exc
    return getattr(mod, attr)
