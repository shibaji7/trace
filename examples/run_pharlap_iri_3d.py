#!/usr/bin/env python3
"""Example: build 3D PyIRI/collision grids and run PHaRLAP 3D ray tracing."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from dateutil import parser as dparser

# Ensure local project package is imported instead of stdlib `trace`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trace import ensure_pharlap_lib
from trace.collision import ComputeCollision
from trace.density.iri import IRI3d
from trace.geomag import build_geomag_grid
from trace.pharlap import Engine
from trace.plottrace import PlotRays3D
from trace.utils import build_elevations_from_cfg, load_config_3D


def _load_cfg(config_path: Path | str | None):
    return load_config_3D(config_path)


def _to_utc_naive(ts: dt.datetime) -> dt.datetime:
    """
    Convert datetime to UTC-naive to avoid aware/naive comparison issues
    inside downstream model libraries.
    """
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _build_axis(start: float, step: float, count: int) -> np.ndarray:
    return start + step * np.arange(int(count), dtype=float)


def _build_rays(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elevs_1d = build_elevations_from_cfg(cfg)

    if all(
        hasattr(cfg, k) for k in ("start_bearing", "end_bearing", "bearing_increment")
    ):
        b0 = float(cfg.start_bearing)
        b1 = float(cfg.end_bearing)
        db = float(cfg.bearing_increment)
        bearings_1d = np.arange(b0, b1 + (0.5 * db), db, dtype=float)
    elif hasattr(cfg.route, "bearing"):
        bearings_1d = np.array([float(cfg.route.bearing)], dtype=float)
    else:
        raise ValueError(
            "Provide start_bearing/end_bearing/bearing_increment or route.bearing"
        )

    # Trace the full elevation x bearing fan.
    elev_grid, bearing_grid = np.meshgrid(elevs_1d, bearings_1d, indexing="ij")
    elevs = elev_grid.ravel()
    ray_bearings = bearing_grid.ravel()
    freqs = np.full(elevs.shape, float(cfg.frequency), dtype=float)
    return elevs, ray_bearings, freqs


def _build_iri_3d(cfg, event_time: dt.datetime, lats, lons, heights):
    """
    Build PHaRLAP iono grid as (nlat, nlon, nheight) in electrons/cm^3.
    """
    iri = IRI3d(cfg, event_time)
    ne_grid, _ = iri.fetch_dataset(
        event_time,
        lats,
        lons,
        heights,
    )
    return ne_grid


def _build_collision_3d(
    cfg,
    event_time: dt.datetime,
    ne_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
):
    """
    Build collision frequency grid as (nlat, nlon, nheight) in Hz.
    """
    te_3d = np.full_like(ne_grid, 1000.0, dtype=float)
    ti_3d = np.full_like(ne_grid, 1000.0, dtype=float)
    op_3d = 0.9 * ne_grid
    o2p_3d = 0.1 * ne_grid

    cc = ComputeCollision.from_nrlmsise_3d(
        date=event_time,
        lats=lats,
        lons=lons,
        heights_km=heights,
        Te=te_3d,
        Ti=ti_3d,
        edens=ne_grid,
        O2p=o2p_3d,
        Op=op_3d,
        workers=int(getattr(cfg, "worker", 1)),
        update_spaceweather=False,
        suppress_spaceweather_warning=True,
    )
    return cc.collision.nu_ft


def _build_geomag_grids(geomag_cfg, event_time: dt.datetime):
    glats = _build_axis(
        float(geomag_cfg.lat_start),
        float(geomag_cfg.lat_step),
        int(geomag_cfg.num_lats),
    )
    glons = _build_axis(
        float(geomag_cfg.lon_start),
        float(geomag_cfg.lon_step),
        int(geomag_cfg.num_lons),
    )
    ghs = _build_axis(
        float(geomag_cfg.height_start_km),
        float(geomag_cfg.height_step_km),
        int(geomag_cfg.num_heights),
    )

    try:
        coeff_dir = getattr(geomag_cfg, "coeff_dir", None)
        if isinstance(coeff_dir, str) and coeff_dir.strip() == "":
            coeff_dir = None
        gm = build_geomag_grid(
            lats=glats,
            lons=glons,
            alts_km=ghs,
            time=event_time,
            coord_input=str(getattr(geomag_cfg, "coord_input", "GEO")),
            coeff_dir=coeff_dir,
        )
        Bx, By, Bz = gm.Bx, gm.By, gm.Bz
        print(
            "Geomag source: PyIRI IGRF "
            f"(coord_input={getattr(geomag_cfg, 'coord_input', 'GEO')})"
        )
    except Exception as exc:
        # Keep a deterministic fallback so raytrace remains runnable.
        shape = (glats.size, glons.size, ghs.size)
        Bx = np.zeros(shape, dtype=float)
        By = np.zeros(shape, dtype=float)
        Bz = np.full(shape, 5.0e-5, dtype=float)
        print(f"Geomag source fallback: uniform field (reason: {exc})")
    return Bx, By, Bz


def _plot_ray_faces(
    ne_grid: np.ndarray,
    ray_path_data,
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    out_file: Path,
) -> None:
    """Plot side/front faces with 2D-style Earth arc and ray overlays."""
    paths = ray_path_data if isinstance(ray_path_data, list) else [ray_path_data]
    i_lat0 = int(np.argmin(np.abs(lats - origin_lat)))
    i_lon0 = int(np.argmin(np.abs(lons - origin_lon)))

    side_ne = np.clip(ne_grid[i_lat0, :, :], 1.0, None).T  # h x lon
    front_ne = np.clip(ne_grid[:, i_lon0, :], 1.0, None).T  # h x lat
    # Keep horizontal axes in native angular coordinates.
    x_side = lons
    x_front = lats

    ray_side_x, ray_front_x, ray_h = [], [], []
    for rp in paths:
        lat = np.asarray(getattr(rp, "lat", []), dtype=float).ravel()
        lon = np.asarray(getattr(rp, "lon", []), dtype=float).ravel()
        h = np.asarray(getattr(rp, "height", []), dtype=float).ravel()
        if lat.size == 0 or lon.size == 0 or h.size == 0:
            continue
        ray_side_x.append(lon)
        ray_front_x.append(lat)
        ray_h.append(h)

    # Zoom to ray envelope in horizontal coordinates, with small padding.
    def _axis_limits_from_rays(ray_series, fallback):
        if len(ray_series) == 0:
            return [float(np.min(fallback)), float(np.max(fallback))]
        xx = np.concatenate([np.asarray(x, dtype=float).ravel() for x in ray_series])
        xx = xx[np.isfinite(xx)]
        if xx.size == 0:
            return [float(np.min(fallback)), float(np.max(fallback))]
        xmin, xmax = float(np.min(xx)), float(np.max(xx))
        pad = max(0.2, 0.05 * max(1e-6, xmax - xmin))
        return [xmin - pad, xmax + pad]

    xlim_side = _axis_limits_from_rays(ray_side_x, lons)
    xlim_front = _axis_limits_from_rays(ray_front_x, lats)

    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(origin_lat)))
    pr = PlotRays3D(oth=True, figsize=(6.5, 4.5))
    pr.set_param_lims(edens_lim=(1e4, 1e6))
    pr.plot_faces(
        ne_side=side_ne,
        ne_front=front_ne,
        x_side=x_side,
        x_front=x_front,
        heights=heights,
        ray_side_x=ray_side_x,
        ray_front_x=ray_front_x,
        ray_h=ray_h,
        kind="edens",
        xlim_side=xlim_side,
        xlim_front=xlim_front,
        ylim=[-300.0, 600.0],
        xlabel_side="Longitude (deg)",
        xlabel_front="Latitude (deg)",
        x_scale_side_km=km_per_deg_lon,
        x_scale_front_km=km_per_deg_lat,
        x_center_side=origin_lon,
        x_center_front=origin_lat,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pr.save(str(out_file))
    pr.close()


def _run(cfg, event_time: dt.datetime, no_matlab: bool) -> None:
    ensure_pharlap_lib()
    ip = getattr(cfg, "iri_param", object())
    print(
        "PyIRI params:",
        {
            "f107": getattr(ip, "f107", 150.0),
            "foF2_coeff": getattr(ip, "foF2_coeff", "CCIR"),
            "hmF2_model": getattr(ip, "hmF2_model", "SHU2015"),
            "coord": getattr(ip, "coord", "GEO"),
        },
    )

    iono_cfg = cfg.iono_grid
    lats = _build_axis(
        float(iono_cfg.lat_start), float(iono_cfg.lat_step), int(iono_cfg.num_lats)
    )
    lons = _build_axis(
        float(iono_cfg.lon_start), float(iono_cfg.lon_step), int(iono_cfg.num_lons)
    )
    heights = _build_axis(
        float(iono_cfg.height_start_km),
        float(iono_cfg.height_step_km),
        int(iono_cfg.num_heights),
    )

    ne_grid = _build_iri_3d(cfg, event_time, lats, lons, heights)
    collision_freq = _build_collision_3d(cfg, event_time, ne_grid, lats, lons, heights)
    iono_en_grid_5 = ne_grid.copy()

    Bx, By, Bz = _build_geomag_grids(cfg.geomag_grid, event_time)

    elevs, ray_bearings, freqs = _build_rays(cfg)
    origin = cfg.origin if hasattr(cfg, "origin") else cfg.route.start
    origin_lat = float(origin.lat)
    origin_lon = float(origin.lon)
    origin_ht = float(getattr(origin, "height_km", 0.0))

    iono_grid_parms = np.array(
        [
            float(iono_cfg.lat_start),
            float(iono_cfg.lat_step),
            int(iono_cfg.num_lats),
            float(iono_cfg.lon_start),
            float(iono_cfg.lon_step),
            int(iono_cfg.num_lons),
            float(iono_cfg.height_start_km),
            float(iono_cfg.height_step_km),
            int(iono_cfg.num_heights),
        ],
        dtype=float,
    )

    geomag_cfg = cfg.geomag_grid
    geomag_grid_parms = np.array(
        [
            float(geomag_cfg.lat_start),
            float(geomag_cfg.lat_step),
            int(geomag_cfg.num_lats),
            float(geomag_cfg.lon_start),
            float(geomag_cfg.lon_step),
            int(geomag_cfg.num_lons),
            float(geomag_cfg.height_start_km),
            float(geomag_cfg.height_step_km),
            int(geomag_cfg.num_heights),
        ],
        dtype=float,
    )

    print(f"Ne grid shape [lat, lon, h]: {ne_grid.shape}")
    print(f"Collision grid shape [lat, lon, h]: {collision_freq.shape}")
    print(f"Geomag grid shape [lat, lon, h]: {Bx.shape}")
    print(f"Total rays: {elevs.size}")
    print(f"Spherical mode: {bool(getattr(cfg, 'use_spherical', True))}")

    if no_matlab:
        return

    eng = Engine()
    try:
        if bool(getattr(cfg, "use_spherical", True)):
            ray_data, ray_path_data, ray_state_vec = eng.run_pharlap_3d_sp(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                origin_ht=origin_ht,
                elevs=elevs,
                ray_bearings=ray_bearings,
                freqs=freqs,
                OX_mode=int(getattr(cfg, "OX_mode", 1)),
                nhops=int(getattr(cfg, "nhops", 1)),
                tol=float(getattr(cfg, "threshold", 1e-7)),
                rad_earth_m=float(getattr(cfg, "radius_earth_m", 6371000.0)),
                iono_en_grid=ne_grid,
                iono_en_grid_5=iono_en_grid_5,
                collision_freq=collision_freq,
                iono_grid_parms=iono_grid_parms,
                Bx=Bx,
                By=By,
                Bz=Bz,
                geomag_grid_parms=geomag_grid_parms,
            )
        else:
            ray_data, ray_path_data, ray_state_vec = eng.run_pharlap_3d(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                origin_ht=origin_ht,
                elevs=elevs,
                ray_bearings=ray_bearings,
                freqs=freqs,
                OX_mode=int(getattr(cfg, "OX_mode", 1)),
                nhops=int(getattr(cfg, "nhops", 1)),
                tol=float(getattr(cfg, "threshold", 1e-7)),
                iono_en_grid=ne_grid,
                iono_en_grid_5=iono_en_grid_5,
                collision_freq=collision_freq,
                iono_grid_parms=iono_grid_parms,
                Bx=Bx,
                By=By,
                Bz=Bz,
                geomag_grid_parms=geomag_grid_parms,
            )
    finally:
        eng.close()

    n_rays = len(ray_data) if isinstance(ray_data, list) else 1
    print(f"Raytrace completed. ray_data entries: {n_rays}")
    print(
        f"Returned: ray_data={type(ray_data)}, "
        f"ray_path_data={type(ray_path_data)}, ray_state_vec={type(ray_state_vec)}"
    )
    out_file = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "examples"
        / "figures"
        / "pharlap_iri_3d_ray_faces.png"
    )
    _plot_ray_faces(
        ne_grid=ne_grid,
        ray_path_data=ray_path_data,
        lats=lats,
        lons=lons,
        heights=heights,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        out_file=out_file,
    )
    print(f"Saved plot: {out_file}")
    print("MATLAB geoplot3 output is temporarily disabled in this example.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D PHaRLAP with IRI electron-density and collision model."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config. If omitted, uses installed trace/cfg/config3D.json",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="UTC timestamp override, e.g. 2017-05-27T16:00:00Z",
    )
    parser.add_argument(
        "--no-matlab",
        action="store_true",
        help="Build all inputs but skip MATLAB raytrace call.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = _load_cfg(cfg_path)
    event_time = (
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )
    event_time = _to_utc_naive(event_time)
    _run(cfg=cfg, event_time=event_time, no_matlab=args.no_matlab)


if __name__ == "__main__":
    main()
