#!/usr/bin/env python3
"""Example: RT3D oblique fan with multi-hop ground reflections.

Demonstrates ``nhops`` parameter added to ``RT3D.oblique_trace``:

* nhops=1  →  standard single-hop (identical to previous behaviour)
* nhops=2  →  one ground reflection (2-hop)
* nhops=N  →  N-1 ground reflections

Each ground reflection uses specular geometry (vz → −vz for cartesian,
vr → −vr for spherical) and restarts the ODE from the domain left edge
with the accumulated x/y offset applied to the output coordinates
(horizontal-homogeneity assumption).

Usage
-----
python examples/run_rt3d_multihop.py --nhops 2
python examples/run_rt3d_multihop.py --nhops 3 --solver hamiltonian
python examples/run_rt3d_multihop.py --nhops 2 --coordinate-system spherical
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dateutil import parser as dparser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfpytrace.model.rt3d import RT3D, RT3DProfile
from hfpytrace.plottrace import PlotRays3D, PlotRays3DRouteFaces
from hfpytrace.utils import build_elevations_from_cfg, load_config_3D

# ── helpers ───────────────────────────────────────────────────────────────────


def _to_utc_naive(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _bearing_to_launch_azimuth_deg(bearing_deg: float, coordinate_system: str) -> float:
    """Convert geographic bearing to RT3D azimuth convention."""
    coord = str(coordinate_system).strip().lower()
    if coord in {"spherical", "sph", "rll"}:
        return float(bearing_deg) % 360.0  # geographic: N=0, E=90
    return (90.0 - float(bearing_deg)) % 360.0  # cartesian:  E=0, N=90 CCW


def _extend_grid_to_ground(rt_profile: RT3DProfile) -> RT3DProfile:
    """Prepend zero-density altitude rows so the ODE can trace to ground."""
    z_floor = float(np.min(rt_profile.alts_km))
    if z_floor <= 0.0:
        return rt_profile
    dz = max(float(np.min(np.diff(rt_profile.alts_km))), 1e-3)
    alts_low = np.arange(0.0, z_floor, dz, dtype=float)
    if alts_low.size == 0:
        return rt_profile
    n_low = int(alts_low.size)
    ne_low = np.zeros((rt_profile.lats.size, rt_profile.lons.size, n_low), dtype=float)
    # Extend auxiliary fields (geomag, msise) by repeating their lowest slice.
    for obj_name in ("geomag", "msise"):
        obj = getattr(rt_profile, obj_name, None)
        if obj is None:
            continue
        for k in vars(obj):
            arr = np.asarray(getattr(obj, k), dtype=float)
            if arr.ndim == 3 and arr.shape[2] == rt_profile.alts_km.size:
                pad = np.repeat(arr[:, :, :1], n_low, axis=2)
                setattr(obj, k, np.concatenate([pad, arr], axis=2))
    rt_profile.alts_km = np.concatenate([alts_low, rt_profile.alts_km])
    rt_profile.ne_cm3 = np.concatenate([ne_low, rt_profile.ne_cm3], axis=2)
    rt_profile.ne_m3 = rt_profile.ne_cm3 * 1e6
    rt_profile.validate()
    return rt_profile


# ── tracing ───────────────────────────────────────────────────────────────────


def _trace_multihop_fan(
    rt: RT3D,
    cfg,
    nhops: int = 2,
    coordinate_system: str = "cartesian",
    solver: str = "gradient",
    collision_hz: np.ndarray | None = None,
    b_abs_t: np.ndarray | None = None,
    b_psi_deg: np.ndarray | None = None,
) -> tuple[list[SimpleNamespace], float, float]:
    """Trace elevation fan at a single bearing with nhops reflections."""
    elevs_1d = build_elevations_from_cfg(cfg)

    if hasattr(cfg.route, "bearing"):
        bearing = float(cfg.route.bearing)
    else:
        bearing = 0.5 * (
            float(getattr(cfg, "start_bearing", 0.0))
            + float(getattr(cfg, "end_bearing", 90.0))
        )

    origin = cfg.origin if hasattr(cfg, "origin") else cfg.route.start
    origin_lat = float(origin.lat)
    origin_lon = float(origin.lon)
    z0_km = float(max(getattr(origin, "height_km", 0.0), float(np.min(rt.alts_km))))

    lat_ref = float(np.mean(rt.lats))
    lon_ref = float(np.mean(rt.lons))
    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(lat_ref)))
    x0_km = (origin_lon - lon_ref) * km_per_deg_lon
    y0_km = (origin_lat - lat_ref) * km_per_deg_lat

    az = _bearing_to_launch_azimuth_deg(bearing, coordinate_system)
    freq_hz = float(cfg.frequency) * 1e6
    s_max = float(getattr(cfg, "max_ground_range_km", 4000.0))

    ray_paths: list[SimpleNamespace] = []
    for elev in elevs_1d:
        out = rt.oblique_trace(
            freq_hz=freq_hz,
            elevation_deg=float(elev),
            azimuth_deg=float(az),
            coordinate_system=coordinate_system,
            solver=solver,
            nhops=nhops,
            x0_km=float(x0_km),
            y0_km=float(y0_km),
            z0_km=float(z0_km),
            s_max_km=s_max,
            mode="O",
            formulation="appleton-hartree",
            collision_hz=collision_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            max_step_km=2.0,
        )
        # Convert local tangent-plane x/y → geographic lat/lon for plotting.
        x_km = np.asarray(out.x_km, dtype=float)
        y_km = np.asarray(out.y_km, dtype=float)
        z_km = np.asarray(out.z_km, dtype=float)
        lat = lat_ref + y_km / km_per_deg_lat
        lon = lon_ref + x_km / km_per_deg_lon

        # Prepend a ground-level launch segment for visual clarity.
        if z_km.size > 0 and float(z0_km) > 0.0:
            lat = np.concatenate(([origin_lat], lat))
            lon = np.concatenate(([origin_lon], lon))
            z_km = np.concatenate(([0.0], z_km))

        ray_paths.append(
            SimpleNamespace(
                lat=lat,
                lon=lon,
                height=z_km,
                initial_elev=float(elev),
                bearing=float(bearing),
                status=str(out.status),
                nhops_completed=int(getattr(out, "nhops_completed", 1)),
                group_path_km=float(out.group_path_km),
            )
        )

    return ray_paths, origin_lat, origin_lon


# ── plotting ──────────────────────────────────────────────────────────────────


def _plot_ray_faces(
    ne_grid: np.ndarray,
    ray_paths: list[SimpleNamespace],
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    nhops: int,
    out_file: Path,
) -> None:
    i_lat0 = int(np.argmin(np.abs(lats - origin_lat)))
    i_lon0 = int(np.argmin(np.abs(lons - origin_lon)))

    side_ne = np.clip(ne_grid[i_lat0, :, :], 1.0, None).T  # (h, lon)
    front_ne = np.clip(ne_grid[:, i_lon0, :], 1.0, None).T  # (h, lat)

    ray_side_x, ray_front_x, ray_h = [], [], []
    for rp in ray_paths:
        lat = np.asarray(rp.lat, dtype=float).ravel()
        lon = np.asarray(rp.lon, dtype=float).ravel()
        h = np.asarray(rp.height, dtype=float).ravel()
        if lat.size == 0:
            continue
        ray_side_x.append(lon)
        ray_front_x.append(lat)
        ray_h.append(h)

    def _xlim(series, fallback):
        if not series:
            return [float(np.min(fallback)), float(np.max(fallback))]
        xx = np.concatenate([x.ravel() for x in series])
        xx = xx[np.isfinite(xx)]
        if xx.size == 0:
            return [float(np.min(fallback)), float(np.max(fallback))]
        pad = max(0.2, 0.05 * max(1e-6, float(np.max(xx)) - float(np.min(xx))))
        return [float(np.min(xx)) - pad, float(np.max(xx)) + pad]

    km_per_deg_lat = 111.32
    km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(origin_lat)))

    pr = PlotRays3D(oth=True, figsize=(6.5, 4.5))
    pr.set_param_lims(edens_lim=(1e4, 1e6))
    pr.plot_faces(
        ne_side=side_ne,
        ne_front=front_ne,
        x_side=lons,
        x_front=lats,
        heights=np.asarray(heights, dtype=float),
        ray_side_x=ray_side_x,
        ray_front_x=ray_front_x,
        ray_h=ray_h,
        kind="edens",
        xlim_side=_xlim(ray_side_x, lons),
        xlim_front=_xlim(ray_front_x, lats),
        ylim=[-300.0, 800.0],
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
    print(f"Saved: {out_file}")


def _plot_route_faces(
    ne_grid: np.ndarray,
    ray_paths: list[SimpleNamespace],
    lats: np.ndarray,
    lons: np.ndarray,
    heights: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    bearing_deg: float,
    nhops: int,
    out_file: Path,
) -> None:
    pr = PlotRays3DRouteFaces(oth=True, figsize=(6.5, 4.5))
    pr.set_param_lims(edens_lim=(1e4, 1e6))
    pr.plot_route_faces(
        ne_grid=ne_grid,
        ray_path_data=ray_paths,
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
    print(f"Saved: {out_file}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="RT3D oblique fan with multi-hop ground reflections."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to JSON config3D.  Defaults to installed hfpytrace/cfg/config3D.json.",
    )
    p.add_argument(
        "--event",
        default=None,
        help="UTC timestamp override, e.g. 2017-05-27T16:00:00Z",
    )
    p.add_argument(
        "--nhops",
        type=int,
        default=2,
        help="Number of ionospheric hops (1=single, 2=one reflection, …).",
    )
    p.add_argument(
        "--coordinate-system",
        default="cartesian",
        choices=("cartesian", "spherical"),
        help="Ray-ODE coordinate system.",
    )
    p.add_argument(
        "--solver",
        default="gradient",
        choices=("gradient", "hamiltonian"),
        help="Solver backend (hamiltonian requires cartesian).",
    )
    p.add_argument(
        "--no-geomag", action="store_true", help="Skip geomagnetic field fetch."
    )
    p.add_argument(
        "--no-collision", action="store_true", help="Skip collision-frequency model."
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: docs/examples/figures/).",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = load_config_3D(cfg_path)
    event_time = _to_utc_naive(
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )

    workers = int(getattr(cfg, "worker", 1))
    fetch_geomag = not args.no_geomag

    print(f"Event:             {event_time.isoformat()}")
    print(f"nhops:             {args.nhops}")
    print(f"coordinate_system: {args.coordinate_system}")
    print(f"solver:            {args.solver}")

    # ── build profile ─────────────────────────────────────────────────────
    rt_profile = RT3DProfile.from_cfg(
        cfg=cfg,
        time=event_time,
        fetch_iri=True,
        fetch_msise=False,
        fetch_geomag=fetch_geomag,
        workers=workers,
    )
    ne_plot = np.asarray(rt_profile.ne_cm3, dtype=float).copy()
    heights_plot = np.asarray(rt_profile.alts_km, dtype=float).copy()

    rt_profile = _extend_grid_to_ground(rt_profile)
    print(
        f"Grid: lat {rt_profile.lats[0]:.1f}–{rt_profile.lats[-1]:.1f}°, "
        f"lon {rt_profile.lons[0]:.1f}–{rt_profile.lons[-1]:.1f}°, "
        f"alt {rt_profile.alts_km[0]:.0f}–{rt_profile.alts_km[-1]:.0f} km"
    )

    # ── collision model ───────────────────────────────────────────────────
    if not args.no_collision:
        try:
            from hfpytrace.collision import ComputeCollision

            te = np.full_like(rt_profile.ne_cm3, 1000.0, dtype=float)
            ti = np.full_like(rt_profile.ne_cm3, 1000.0, dtype=float)
            op = 0.9 * rt_profile.ne_cm3
            o2p = 0.1 * rt_profile.ne_cm3
            cc = ComputeCollision.from_nrlmsise_3d(
                date=event_time,
                lats=rt_profile.lats,
                lons=rt_profile.lons,
                heights_km=rt_profile.alts_km,
                Te=te,
                Ti=ti,
                edens=rt_profile.ne_cm3,
                O2p=o2p,
                Op=op,
                workers=workers,
                update_spaceweather=False,
                suppress_spaceweather_warning=True,
            )
            collision_hz = np.asarray(cc.collision.nu_ft, dtype=float)
            print("Collision model: Friedrich-Tonker (nu_ft)")
        except Exception as exc:
            print(f"Collision unavailable ({exc}); using zeros.")
            collision_hz = np.zeros_like(rt_profile.ne_cm3, dtype=float)
    else:
        collision_hz = np.zeros_like(rt_profile.ne_cm3, dtype=float)
        print("Collision modeling disabled.")

    # ── geomag ────────────────────────────────────────────────────────────
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

    # ── trace ─────────────────────────────────────────────────────────────
    rt = RT3D(profile=rt_profile)
    ray_paths, origin_lat, origin_lon = _trace_multihop_fan(
        rt=rt,
        cfg=cfg,
        nhops=args.nhops,
        coordinate_system=args.coordinate_system,
        solver=args.solver,
        collision_hz=collision_hz,
        b_abs_t=b_abs_t,
        b_psi_deg=b_psi_deg,
    )

    # Summary statistics
    statuses = {}
    hops_hist: dict[int, int] = {}
    for rp in ray_paths:
        statuses[rp.status] = statuses.get(rp.status, 0) + 1
        h = rp.nhops_completed
        hops_hist[h] = hops_hist.get(h, 0) + 1
    print(f"Rays traced: {len(ray_paths)}")
    print(f"Statuses:    {statuses}")
    print(f"Hops completed histogram: {hops_hist}")

    # ── output paths ──────────────────────────────────────────────────────
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else PROJECT_ROOT / "docs" / "examples" / "figures"
    )
    tag = f"nhops{args.nhops}"
    _plot_ray_faces(
        ne_grid=ne_plot,
        ray_paths=ray_paths,
        lats=rt_profile.lats,
        lons=rt_profile.lons,
        heights=heights_plot,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        nhops=args.nhops,
        out_file=out_dir / f"rt3d_multihop_{tag}_ray_faces.png",
    )

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
    _plot_route_faces(
        ne_grid=ne_plot,
        ray_paths=ray_paths,
        lats=rt_profile.lats,
        lons=rt_profile.lons,
        heights=heights_plot,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        bearing_deg=bearing_ref,
        nhops=args.nhops,
        out_file=out_dir / f"rt3d_multihop_{tag}_route_faces.png",
    )


if __name__ == "__main__":
    main()
