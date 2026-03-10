#!/usr/bin/env python3
"""Example: run RT3D Cartesian oblique fan on IRI 3D grid and plot ray faces."""

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

from hfpytrace.collision import ComputeCollision
from hfpytrace.model.rt3d import RT3D, RT3DProfile
from hfpytrace.plottrace import PlotRays3D, PlotRays3DRouteFaces
from hfpytrace.utils import build_elevations_from_cfg, load_config_3D


def _load_cfg(config_path: Path | str | None):
    return load_config_3D(config_path)


def _to_utc_naive(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)


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
    elev_grid, bearing_grid = np.meshgrid(elevs_1d, bearings_1d, indexing="ij")
    elevs = elev_grid.ravel()
    bearings = bearing_grid.ravel()
    freqs = np.full(elevs.shape, float(cfg.frequency), dtype=float)
    return elevs, bearings, freqs


def _bearing_to_cart_azimuth_deg(bearing_deg: float) -> float:
    # Bearing is typically CW from geographic north. RT3D Cartesian azimuth is
    # defined from +x (east) toward +y (north), CCW-positive.
    return (90.0 - float(bearing_deg)) % 360.0


def _bearing_to_launch_azimuth_deg(bearing_deg: float, coordinate_system: str) -> float:
    # Spherical tracer uses geographic convention (north=0, east=90).
    coord = str(coordinate_system).strip().lower()
    if coord in {"spherical", "sph", "rll"}:
        return float(bearing_deg) % 360.0
    return _bearing_to_cart_azimuth_deg(float(bearing_deg))


def _trace_fan(
    rt: RT3D,
    cfg,
    elevs: np.ndarray,
    bearings: np.ndarray,
    freqs: np.ndarray,
    collision_hz: np.ndarray | None = None,
    coordinate_system: str = "cartesian",
    solver: str = "gradient",
    b_abs_t: np.ndarray | None = None,
    b_psi_deg: np.ndarray | None = None,
):
    origin = cfg.origin if hasattr(cfg, "origin") else cfg.route.start
    origin_lat = float(origin.lat)
    origin_lon = float(origin.lon)
    origin_ht = float(getattr(origin, "height_km", 0.0))
    # Launch at least at the model grid floor (PHaRLAP can propagate from ground
    # through vacuum; current RT3D solver needs an in-domain launch point).
    z0_km = float(max(origin_ht, float(np.min(rt.alts_km))))

    # Local tangent-plane conversion referenced at grid center.
    lat_ref = float(np.mean(rt.lats))
    lon_ref = float(np.mean(rt.lons))
    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(lat_ref)))
    x0_km = (origin_lon - lon_ref) * km_per_deg_lon
    y0_km = (origin_lat - lat_ref) * km_per_deg_lat

    ray_paths = []
    for el, brg, f_mhz in zip(elevs, bearings, freqs):
        az = _bearing_to_launch_azimuth_deg(
            float(brg), coordinate_system=coordinate_system
        )
        out = rt.oblique_trace(
            freq_hz=float(f_mhz) * 1e6,
            elevation_deg=float(el),
            azimuth_deg=float(az),
            coordinate_system=str(coordinate_system),
            x0_km=float(x0_km),
            y0_km=float(y0_km),
            z0_km=float(z0_km),
            s_max_km=float(getattr(cfg, "max_ground_range_km", 3000.0)),
            mode="O",
            formulation="appleton-hartree",
            collision_hz=collision_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            solver=str(solver),
            max_step_km=2.0,
        )
        # Convert local x/y back to lat/lon for face plotting.
        lat = lat_ref + np.asarray(out.y_km, dtype=float) / km_per_deg_lat
        lon = lon_ref + np.asarray(out.x_km, dtype=float) / km_per_deg_lon
        h = np.asarray(out.z_km, dtype=float)
        # Solver runs from model floor (typically 100 km), but for figure
        # readability we prepend a short launch segment from ground.
        if h.size > 0 and float(origin_ht) < float(z0_km):
            lat = np.concatenate(([origin_lat], lat))
            lon = np.concatenate(([origin_lon], lon))
            h = np.concatenate(([float(origin_ht)], h))
        ray_paths.append(
            SimpleNamespace(
                lat=np.asarray(lat, dtype=float),
                lon=np.asarray(lon, dtype=float),
                height=np.asarray(h, dtype=float),
                initial_elev=float(el),
                initial_bearing=float(brg),
                status=str(out.status),
                npts=int(np.asarray(out.x_km, dtype=float).size),
            )
        )
    return ray_paths, origin_lat, origin_lon


def _all_or_most_failed(ray_paths, frac: float = 0.8) -> bool:
    if len(ray_paths) == 0:
        return True
    n_fail = sum(
        1 for rp in ray_paths if str(getattr(rp, "status", "")).lower() == "failure"
    )
    return (n_fail / float(len(ray_paths))) >= float(frac)


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
    paths = ray_path_data if isinstance(ray_path_data, list) else [ray_path_data]
    i_lat0 = int(np.argmin(np.abs(lats - origin_lat)))
    i_lon0 = int(np.argmin(np.abs(lons - origin_lon)))

    side_ne = np.clip(ne_grid[i_lat0, :, :], 1.0, None).T  # h x lon
    front_ne = np.clip(ne_grid[:, i_lon0, :], 1.0, None).T  # h x lat
    heights_plot = np.asarray(heights, dtype=float)
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
        heights=heights_plot,
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
        curve_density=True,
        curve_rays=True,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pr.save(str(out_file))
    pr.close()


def _plot_route_faces(
    ne_grid: np.ndarray,
    ray_path_data,
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    bearing_deg: float,
    out_file: Path,
) -> None:
    pr = PlotRays3DRouteFaces(oth=True, figsize=(6.5, 4.5))
    pr.set_param_lims(edens_lim=(1e4, 1e6))
    pr.plot_route_faces(
        ne_grid=ne_grid,
        ray_path_data=ray_path_data,
        lats=lats,
        lons=lons,
        heights=heights,
        origin_lat=float(origin_lat),
        origin_lon=float(origin_lon),
        bearing_deg=float(bearing_deg),
        kind="edens",
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pr.save(str(out_file))
    pr.close()


def _run(cfg, event_time: dt.datetime):
    workers = int(getattr(cfg, "worker", 1))
    use_collisions = bool(getattr(cfg, "use_collisions", True))
    fetch_geomag = bool(getattr(cfg, "fetch_geomag", True))
    coordinate_system = (
        "spherical" if bool(getattr(cfg, "use_spherical", True)) else "cartesian"
    )
    solver = str(
        getattr(
            cfg,
            "solver",
            "gradient" if coordinate_system == "spherical" else "hamiltonian",
        )
    )
    rt_profile = RT3DProfile.from_cfg(
        cfg=cfg,
        time=event_time,
        fetch_iri=True,
        fetch_msise=False,
        fetch_geomag=fetch_geomag,
        workers=workers,
    )
    # Keep original grid for plotting so density starts at configured floor.
    ne_plot = np.asarray(rt_profile.ne_cm3, dtype=float).copy()
    heights_plot = np.asarray(rt_profile.alts_km, dtype=float).copy()
    z_floor = float(np.min(heights_plot))

    # Extend tracing grid to ground by default (can be disabled via cfg).
    if bool(getattr(cfg, "extend_to_ground", True)) and z_floor > 0.0:
        if heights_plot.size >= 2:
            dz = float(np.min(np.diff(heights_plot)))
            dz = max(dz, 1e-3)
        else:
            dz = z_floor
        alts_low = np.arange(0.0, z_floor, dz, dtype=float)
        if alts_low.size > 0:
            n_low = int(alts_low.size)
            ne_low = np.zeros(
                (rt_profile.lats.size, rt_profile.lons.size, n_low),
                dtype=float,
            )

            # Keep optional 3D auxiliary fields shape-consistent with the
            # extended altitude axis by prepending their first altitude slice.
            if getattr(rt_profile, "geomag", None) is not None:
                for k in ("Bx", "By", "Bz", "bmag_t", "inc_deg", "dec_deg", "psi_deg"):
                    if hasattr(rt_profile.geomag, k):
                        arr = np.asarray(getattr(rt_profile.geomag, k), dtype=float)
                        pad = np.repeat(arr[:, :, :1], n_low, axis=2)
                        setattr(
                            rt_profile.geomag, k, np.concatenate([pad, arr], axis=2)
                        )
            if getattr(rt_profile, "msise", None) is not None:
                for k in ("N2", "O2", "O", "H", "He", "Tn", "t_nn"):
                    if hasattr(rt_profile.msise, k):
                        arr = np.asarray(getattr(rt_profile.msise, k), dtype=float)
                        pad = np.repeat(arr[:, :, :1], n_low, axis=2)
                        setattr(rt_profile.msise, k, np.concatenate([pad, arr], axis=2))

            rt_profile.alts_km = np.concatenate([alts_low, rt_profile.alts_km])
            rt_profile.ne_cm3 = np.concatenate([ne_low, rt_profile.ne_cm3], axis=2)
            rt_profile.ne_m3 = rt_profile.ne_cm3 * 1e6
            rt_profile.validate()
            print(
                "Extended tracing altitude grid to ground: "
                f"{float(rt_profile.alts_km[0]):.1f}..{float(rt_profile.alts_km[-1]):.1f} km"
            )

    # Build collision grid (Hz) on the same tracing grid.
    if use_collisions:
        try:
            te_grid = np.full_like(rt_profile.ne_cm3, 1000.0, dtype=float)
            ti_grid = np.full_like(rt_profile.ne_cm3, 1000.0, dtype=float)
            op_grid = 0.9 * rt_profile.ne_cm3
            o2p_grid = 0.1 * rt_profile.ne_cm3
            cc = ComputeCollision.from_nrlmsise_3d(
                date=event_time,
                lats=rt_profile.lats,
                lons=rt_profile.lons,
                heights_km=rt_profile.alts_km,
                Te=te_grid,
                Ti=ti_grid,
                edens=rt_profile.ne_cm3,
                O2p=o2p_grid,
                Op=op_grid,
                workers=workers,
                update_spaceweather=False,
                suppress_spaceweather_warning=True,
            )
            collision_hz = np.asarray(cc.collision.nu_ft, dtype=float)
        except Exception as exc:
            print(f"Collision model unavailable, using zeros: {exc}")
            collision_hz = np.zeros_like(rt_profile.ne_cm3, dtype=float)
    else:
        collision_hz = np.zeros_like(rt_profile.ne_cm3, dtype=float)
        print("Collision modeling disabled (cfg.use_collisions=false).")

    rt = RT3D(profile=rt_profile)
    b_abs_t = (
        np.asarray(rt_profile.geomag.bmag_t, dtype=float)
        if getattr(rt_profile, "geomag", None) is not None
        else None
    )
    b_psi_deg = (
        np.asarray(rt_profile.geomag.psi_deg, dtype=float)
        if getattr(rt_profile, "geomag", None) is not None
        else None
    )
    elevs, bearings, freqs = _build_rays(cfg)
    ray_paths, origin_lat, origin_lon = _trace_fan(
        rt=rt,
        cfg=cfg,
        elevs=elevs,
        bearings=bearings,
        freqs=freqs,
        collision_hz=collision_hz,
        coordinate_system=coordinate_system,
        solver=solver,
        b_abs_t=b_abs_t,
        b_psi_deg=b_psi_deg,
    )

    # Robust fallback: if Hamiltonian fails for most rays, retry with gradient.
    if str(solver).strip().lower() in {"hamiltonian", "ham"} and _all_or_most_failed(
        ray_paths
    ):
        print("Most Hamiltonian rays failed; retrying with solver=gradient.")
        ray_paths, origin_lat, origin_lon = _trace_fan(
            rt=rt,
            cfg=cfg,
            elevs=elevs,
            bearings=bearings,
            freqs=freqs,
            collision_hz=collision_hz,
            coordinate_system=coordinate_system,
            solver="gradient",
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
        )
        solver = "gradient"

    print(f"Ne grid shape [lat, lon, h]: {rt_profile.ne_cm3.shape}")
    print(f"RT3D coordinate_system={coordinate_system}, solver={solver}")
    print(f"use_collisions={use_collisions}, fetch_geomag={fetch_geomag}")
    print(f"Total rays traced: {len(ray_paths)}")
    if len(ray_paths) > 0:
        statuses = {}
        n_valid = 0
        for rp in ray_paths:
            statuses[rp.status] = statuses.get(rp.status, 0) + 1
            if int(getattr(rp, "npts", 0)) >= 2:
                n_valid += 1
        print(f"Ray statuses: {statuses}")
        print(f"Rays with >=2 points (plottable): {n_valid}/{len(ray_paths)}")

    out_file = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "examples"
        / "figures"
        / "rt3d_iri_cartesian_ray_faces.png"
    )
    _plot_ray_faces(
        ne_grid=ne_plot,
        ray_path_data=ray_paths,
        lats=rt_profile.lats,
        lons=rt_profile.lons,
        heights=heights_plot,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        out_file=out_file,
    )
    print(f"Saved plot: {out_file}")
    bearing_ref = float(
        getattr(
            cfg.route,
            "bearing",
            0.5
            * (
                float(getattr(cfg, "start_bearing", 0.0))
                + float(getattr(cfg, "end_bearing", 0.0))
            ),
        )
    )
    out_file_route = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "examples"
        / "figures"
        / "rt3d_iri_cartesian_route_faces.png"
    )
    _plot_route_faces(
        ne_grid=ne_plot,
        ray_path_data=ray_paths,
        lats=rt_profile.lats,
        lons=rt_profile.lons,
        heights=heights_plot,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        bearing_deg=bearing_ref,
        out_file=out_file_route,
    )
    print(f"Saved plot: {out_file_route}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RT3D Cartesian oblique fan with IRI 3D grid and plot faces."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config. If omitted, uses installed hfpytrace/cfg/config3D.json",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="UTC timestamp override, e.g. 2017-05-27T16:00:00Z",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = _load_cfg(cfg_path)
    event_time = (
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )
    event_time = _to_utc_naive(event_time)
    _run(cfg=cfg, event_time=event_time)


if __name__ == "__main__":
    main()
