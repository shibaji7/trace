#!/usr/bin/env python3
"""Example: build PyIRI electron density grid and run trace.pharlap.Engine."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dateutil import parser as dparser

# Ensure local project package is imported instead of stdlib `trace`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace import ensure_pharlap_lib
from trace.collision import ComputeCollision
from trace.density.iri import IRI2d
from trace.pharlap import Engine
from trace.plottrace import PlotRays
from trace.utils import (
    build_elevations_from_cfg,
    build_freqs_from_cfg,
    build_heights_from_cfg,
    build_route_from_cfg,
    load_config_2D,
)


def _load_cfg(config_path: Path | str | None):
    return load_config_2D(config_path)


def _plot_ray_paths(
    ne_grid: np.ndarray,
    ray_path_data,
    heights: np.ndarray,
    route_km: float,
    out_file: Path,
) -> None:
    X, Z = np.meshgrid(np.linspace(0.0, route_km, ne_grid.shape[1]), heights)
    rays = []
    paths = ray_path_data if isinstance(ray_path_data, list) else [ray_path_data]
    y_max = float(np.max(heights))
    for r in paths:
        y_vals = np.asarray(r.height, dtype=float)
        if y_vals.size > 0:
            y_max = max(y_max, float(np.nanmax(y_vals)))
        rays.append(
            SimpleNamespace(
                x_km=np.asarray(r.ground_range, dtype=float),
                y_km=y_vals,
                el0_deg=float(r.initial_elev),
            )
        )

    rp = PlotRays(
        nrows=1,
        ncols=1,
        oth=True,
        xlim=[0.0, 1500],
        ylim=[-100, y_max * 1.02],
        figsize=(7, 4),
    )
    rp.set_param_lims(edens_lim=(1e3, 1e6))
    rp.set_density(X, Z, ne_grid, pf=None)
    rp.lay_rays(outputs=rays, kind="edens", lcolor="k", lw=0.6, param_alpha=0.85)
    rp.save(str(out_file))
    rp.close()


def _run(cfg, event_time: dt.datetime, no_matlab: bool) -> None:
    ensure_pharlap_lib()
    ip = getattr(cfg, "iri_param", SimpleNamespace())
    print(
        "PyIRI params:",
        {
            "f107": getattr(ip, "f107", 150.0),
            "foF2_coeff": getattr(ip, "foF2_coeff", "CCIR"),
            "hmF2_model": getattr(ip, "hmF2_model", "SHU2015"),
            "coord": getattr(ip, "coord", "GEO"),
        },
    )

    n_range = int(cfg.number_of_ground_step_km)
    workers = int(getattr(cfg, "worker", 1))
    lats, lons, rb, route_km = build_route_from_cfg(cfg, n_range)
    heights = build_heights_from_cfg(cfg)
    iri = IRI2d(cfg, event_time)
    ne_grid, _ = iri.fetch_dataset(event_time, lats, lons, heights, workers=workers)

    # Build collision frequency grid from NRLMSISE neutrals + simple plasma assumptions.
    # If richer IRI plasma fields are available in your workflow, replace these defaults.
    te_grid = np.full_like(ne_grid, 1000.0, dtype=float)
    ti_grid = np.full_like(ne_grid, 1000.0, dtype=float)
    op_grid = 0.9 * ne_grid
    o2p_grid = 0.1 * ne_grid
    try:
        cc = ComputeCollision.from_nrlmsise(
            date=event_time,
            lats=lats,
            lons=lons,
            heights_km=heights,
            Te=te_grid,
            Ti=ti_grid,
            edens=ne_grid,
            O2p=o2p_grid,
            Op=op_grid,
            workers=workers,
            update_spaceweather=False,
            suppress_spaceweather_warning=True,
        )
        collision_freq = cc.collision.nu_ft
    except Exception as exc:
        print(f"Collision model unavailable, using zeros: {exc}")
        collision_freq = np.zeros_like(ne_grid, dtype=float)
    irreg = np.zeros((4, ne_grid.shape[1]), dtype=float)
    elevs = build_elevations_from_cfg(cfg)
    freqs = build_freqs_from_cfg(cfg, elevs)
    range_inc = float(route_km) / max(ne_grid.shape[1] - 1, 1)
    height_inc = float(getattr(cfg, "height_incriment_km", 1.0))

    print(f"IRI Ne grid shape: {ne_grid.shape} [heights x ranges]")
    print(f"Elevations: {elevs[0]:.2f}..{elevs[-1]:.2f} deg ({elevs.size} rays)")
    print(f"Bearing: {rb:.2f} deg")
    print(f"Range increment: {range_inc:.3f} km")
    print(f"Height increment: {height_inc:.3f} km")

    if no_matlab:
        return

    engine = Engine()
    try:
        ray_data, ray_path_data = engine.run_pharlap(
            ne_grid=ne_grid,
            collision_freq=collision_freq,
            elevs=elevs,
            rb=rb,
            freqs=freqs,  # MHz for PHaRLAP raytrace_2d_sp
            irreg=irreg,
            nhops=int(cfg.nhops),
            tol=float(getattr(cfg, "threshold", 1e-7)),
            radius_earth=float(cfg.radius_earth),
            irregs_flag=0,
            start_height=int(round(float(cfg.start_height_km))),
            height_inc=float(height_inc),
            range_inc=float(range_inc),
        )
    finally:
        engine.close()

    print(f"ray_data entries: {len(ray_data)}")
    print(f"ray_path_data entries: {len(ray_path_data)}")
    out_file = Path.cwd() / "docs/examples/figures/pharlap_iri_ray_paths.png"
    _plot_ray_paths(ne_grid, ray_path_data, heights, route_km, out_file)
    print(f"Saved plot: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trace.pharlap with IRI electron density model."
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
        "--no-matlab",
        action="store_true",
        help="Build IRI + PHaRLAP inputs but skip MATLAB engine execution.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = _load_cfg(config_path)
    print(
        f"Using config: {config_path if config_path else 'installed default config2D.json'}"
    )
    event_time = (
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )

    _run(cfg=cfg, event_time=event_time, no_matlab=args.no_matlab)


if __name__ == "__main__":
    main()
