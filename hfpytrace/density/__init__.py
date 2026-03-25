"""hfpytrace.density — ionospheric electron density models.

All classes are lazily imported so that missing optional dependencies only
raise an error when the specific model class is actually instantiated.

Available models
----------------
IRI2d
    IRI-2016 electron density along a 2D great-circle route (PyIRI).
IRI3d
    IRI-2016 electron density on a 3D geographic grid (PyIRI).
SAMI3
    SAMI3 electron density from netCDF simulation output (xarray).
WACCMX2d
    WACCM-X electron density from netCDF simulation output (xarray).
GEMINI2d
    GEMINI electron density from HDF5 simulation output (h5py).
GITM2d
    GITM electron density from netCDF simulation output (xarray).
WAMIPE2d
    WAM-IPE electron density from HDF5 simulation output (h5py).
    *(Work-in-progress — geographic coordinate mapping incomplete.)*
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
        "IRI2d": ("hfpytrace.density.iri", "IRI2d"),
        "IRI3d": ("hfpytrace.density.iri", "IRI3d"),
        "GEMINI2d": ("hfpytrace.density.gemini", "GEMINI2d"),
        "GITM2d": ("hfpytrace.density.gitm", "GITM2d"),
        "SAMI3": ("hfpytrace.density.sami", "SAMI3"),
        "WACCMX2d": ("hfpytrace.density.waccm", "WACCMX2d"),
        "WAMIPE2d": ("hfpytrace.density.wamipe", "WAMIPE2d"),
    }
    if name not in module_map:
        raise AttributeError(f"module 'hfpytrace.density' has no attribute {name!r}")
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
