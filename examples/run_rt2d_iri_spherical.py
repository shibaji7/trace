#!/usr/bin/env python3
"""Example: RT2D spherical oblique tracing on an IRI profile."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dateutil import parser as dparser
from loguru import logger

# Ensure local project package is imported instead of stdlib `trace`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace.model.rt2d import RT2D, RT2DProfile
from trace.plottrace import PlotRays
from trace.utils import build_elevations_from_cfg, build_freqs_from_cfg, load_config_2D


def _load_cfg(config_path: Path | None):
    return load_config_2D(config_path)


def _trace_fan_spherical(
    model: RT2D,
    heights_km: np.ndarray,
    elevs_deg: np.ndarray,
    freqs_mhz: np.ndarray,
    mode: str,
    formulation: str,
    r_earth_km: float,
) -> list[SimpleNamespace]:
    rays: list[SimpleNamespace] = []
    z0 = float(heights_km[0])
    for elev_deg, f_mhz in zip(elevs_deg, freqs_mhz):
        out = model.oblique_trace(
            freq_hz=float(f_mhz) * 1e6,
            elevation_deg=float(elev_deg),
            coordinate_system="spherical",
            x0_km=0.0,
            z0_km=z0,
            s_max_km=7000.0,
            mode=mode,
            formulation=formulation,
            r_earth_km=float(r_earth_km),
            max_step_km=2.0,
        )
        rays.append(
            SimpleNamespace(
                x_km=np.asarray(out.x_km, dtype=float),
                y_km=np.asarray(out.z_km, dtype=float),
                el0_deg=float(elev_deg),
            )
        )
    return rays


def _plot_density_and_rays(
    profile: RT2DProfile,
    rays: list[SimpleNamespace],
    out_file: Path,
) -> None:
    x = np.asarray(profile.x_km, dtype=float)
    z = np.asarray(profile.alt_km, dtype=float)
    X, Z = np.meshgrid(x, z)
    y_max = float(np.nanmax(z))
    for r in rays:
        y_vals = np.asarray(r.y_km, dtype=float)
        if y_vals.size > 0:
            y_max = max(y_max, float(np.nanmax(y_vals)))

    p = PlotRays(
        nrows=1,
        ncols=1,
        oth=True,
        xlim=[0.0, 1500.0],
        ylim=[-100.0, y_max * 1.02],
        figsize=(7, 4),
    )
    p.set_param_lims(edens_lim=(1e3, 1e6))
    p.set_density(X, Z, np.asarray(profile.ne_cm3, dtype=float), pf=None)
    p.lay_rays(outputs=rays, kind="edens", lcolor="k", lw=0.6, param_alpha=0.85)
    p.save(str(out_file))
    p.close()
    logger.info("Saved RT2D spherical plot: {}", out_file)


def _run(
    cfg,
    event_time: dt.datetime,
    mode: str,
    formulation: str,
    r_earth_km: float,
) -> None:
    workers = int(getattr(cfg, "worker", 1))
    h_top = float(cfg.end_height_km)
    dh = float(getattr(cfg, "height_incriment_km", 1.0))
    alt_km = np.arange(0.0, h_top, dh, dtype=float)
    profile = RT2DProfile.from_cfg(
        cfg=cfg,
        time=event_time,
        alt_km=alt_km,
        fetch_iri=True,
        fetch_msise=False,
        fetch_geomag=False,
        workers=workers,
    )
    zmin_cfg = float(cfg.start_height_km)
    n_rows = profile.force_zero_density_below(zmin_cfg)
    if n_rows > 0:
        logger.info(
            "Forced Ne=0 below config lower altitude: z < {:.1f} km (rows={})",
            zmin_cfg,
            n_rows,
        )
    logger.info(
        "RT2D spherical profile altitude bounds: {:.1f} to {:.1f} km (n={})",
        float(profile.alt_km[0]),
        float(profile.alt_km[-1]),
        profile.alt_km.size,
    )

    model = RT2D(profile=profile)
    elevs_deg = build_elevations_from_cfg(cfg)
    freqs_mhz = build_freqs_from_cfg(cfg, elevs_deg)
    rays = _trace_fan_spherical(
        model=model,
        heights_km=profile.alt_km,
        elevs_deg=elevs_deg,
        freqs_mhz=freqs_mhz,
        mode=mode,
        formulation=formulation,
        r_earth_km=float(r_earth_km),
    )

    out_file = Path.cwd() / "docs/examples/figures/rt2d_iri_spherical_ray_paths.png"
    _plot_density_and_rays(profile, rays, out_file)
    logger.info(
        "Completed RT2D spherical run: rays={}, mode={}, model={}",
        len(rays),
        mode,
        formulation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RT2D spherical oblique rays using IRI profile and plot rays."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config. If omitted, uses installed trace/cfg/config2D.json",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="UTC timestamp override, e.g. 2017-05-27T16:00:00 (default from config)",
    )
    parser.add_argument(
        "--mode",
        default="O",
        choices=["O", "X", "R", "L", "noB"],
        help="Dispersion mode for RT2D tracing",
    )
    parser.add_argument(
        "--formulation",
        default="appleton-hartree",
        help="Dispersion model name: appleton-hartree or sen-wyller",
    )
    parser.add_argument(
        "--r-earth-km",
        default=6371.0,
        type=float,
        help="Earth radius in km used by spherical tracer",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = _load_cfg(config_path)
    event_time = (
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )
    logger.info(
        "Using config: {}",
        str(config_path) if config_path else "installed default config2D.json",
    )
    _run(
        cfg=cfg,
        event_time=event_time,
        mode=str(args.mode),
        formulation=str(args.formulation),
        r_earth_km=float(args.r_earth_km),
    )


if __name__ == "__main__":
    main()
