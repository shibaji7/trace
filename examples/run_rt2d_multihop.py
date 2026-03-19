#!/usr/bin/env python3
"""Example: RT2D oblique fan with multi-hop ground reflections.

Demonstrates the ``nhops`` parameter added to ``RT2D.oblique_trace``:

* nhops=1  →  standard single-hop (identical to previous behaviour)
* nhops=2  →  one ground reflection (2-hop)
* nhops=N  →  N-1 ground reflections

Each ground reflection uses specular geometry:

* Cartesian: ``vz → −vz``, elevation = arctan2(|vz|, |vx|)
* Spherical: ``v_r → −v_r``, elevation = arctan2(|v_r|, |v_phi|)

The ODE is restarted from x=0 at hops 2+ (horizontal-homogeneity assumption)
so the full n-field grid is available.  Output x coordinates are shifted by
the accumulated physical ground-hit position to form a continuous path.

The plot uses ``PlotRays(oth=True)`` with ``xlim`` auto-extended to the
maximum ground range reached by any ray across all hops.

Usage
-----
python examples/run_rt2d_multihop.py --nhops 2
python examples/run_rt2d_multihop.py --nhops 3 --coordinate-system spherical
python examples/run_rt2d_multihop.py --nhops 2 --freq-mhz 10.0
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dateutil import parser as dparser
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfpytrace.model.rt2d import RT2D, RT2DProfile
from hfpytrace.plottrace import PlotRays
from hfpytrace.utils import (
    build_elevations_from_cfg,
    build_freqs_from_cfg,
    load_config_2D,
)


# ── tracing ───────────────────────────────────────────────────────────────────

def _trace_multihop_fan(
    model: RT2D,
    elevs_deg: np.ndarray,
    freq_hz: float,
    nhops: int,
    coordinate_system: str,
    r_earth_km: float,
    mode: str,
    formulation: str,
    s_max_km: float,
) -> list[SimpleNamespace]:
    """Trace an elevation fan with ``nhops`` ground reflections."""
    z0 = float(model.z_km[0])
    rays: list[SimpleNamespace] = []
    for elev_deg in elevs_deg:
        out = model.oblique_trace(
            freq_hz=float(freq_hz),
            elevation_deg=float(elev_deg),
            coordinate_system=coordinate_system,
            nhops=nhops,
            x0_km=0.0,
            z0_km=z0,
            s_max_km=s_max_km,
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
                nhops_completed=int(getattr(out, "nhops_completed", 1)),
                group_path_km=float(out.group_path_km),
                status=str(out.status),
            )
        )
    return rays


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_multihop_rays(
    profile: RT2DProfile,
    rays: list[SimpleNamespace],
    nhops: int,
    out_file: Path,
) -> None:
    x = np.asarray(profile.x_km, dtype=float)
    z = np.asarray(profile.alt_km, dtype=float)
    X, Z = np.meshgrid(x, z)

    # Auto-extend xlim to the maximum x reached by any ray across all hops.
    x_max = float(np.nanmax(x))
    for r in rays:
        xv = np.asarray(r.x_km, dtype=float)
        xv = xv[np.isfinite(xv)]
        if xv.size > 0:
            x_max = max(x_max, float(np.nanmax(xv)))

    # For multi-hop figures keep ylim deep enough to show the ground arc.
    ylim = [-600.0, 700.0] if nhops > 1 else [-200.0, 700.0]

    p = PlotRays(
        nrows=1,
        ncols=1,
        oth=True,
        xlim=[0.0, x_max],
        ylim=ylim,
        figsize=(9 if nhops > 1 else 7, 4),
        xlabel_loc=(x_max * 0.4, ylim[0] * 0.7),
    )
    p.set_param_lims(edens_lim=(1e3, 1e6))
    p.set_density(X, Z, np.asarray(profile.ne_cm3, dtype=float), pf=None)

    # Colour ground-reaching rays by hop count; others in grey.
    hop_colors = {1: "royalblue", 2: "firebrick", 3: "forestgreen", 4: "darkorange"}
    for r in rays:
        color = hop_colors.get(r.nhops_completed, "grey")
        single = SimpleNamespace(
            x_km=r.x_km,
            y_km=r.y_km,
            el0_deg=r.el0_deg,
        )
        p.lay_rays(
            outputs=[single],
            kind="edens",
            lcolor=color,
            lw=0.8,
            param_alpha=0.7,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    p.save(str(out_file))
    p.close()
    logger.info("Saved RT2D multi-hop plot: {}", out_file)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RT2D oblique fan with multi-hop ground reflections."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to JSON config2D. Defaults to installed hfpytrace/cfg/config2D.json.",
    )
    parser.add_argument(
        "--event", default=None,
        help="UTC timestamp override, e.g. 2017-05-27T16:00:00Z",
    )
    parser.add_argument(
        "--nhops", type=int, default=2,
        help="Number of ionospheric hops (1=single, 2=one reflection, …).",
    )
    parser.add_argument(
        "--coordinate-system", default="cartesian",
        choices=("cartesian", "spherical"),
        help="Ray-ODE coordinate system.",
    )
    parser.add_argument(
        "--freq-mhz", type=float, default=None,
        help="Override transmit frequency in MHz (default: from config).",
    )
    parser.add_argument(
        "--mode", default="O",
        choices=["O", "X", "R", "L", "noB"],
        help="Dispersion mode.",
    )
    parser.add_argument(
        "--formulation", default="appleton-hartree",
        help="Dispersion model: appleton-hartree or sen-wyller.",
    )
    parser.add_argument(
        "--r-earth-km", type=float, default=6371.0,
        help="Earth radius in km for the spherical tracer.",
    )
    parser.add_argument(
        "--s-max-km", type=float, default=None,
        help="Maximum arc-length per hop in km (auto-scaled if omitted).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: docs/examples/figures/).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = load_config_2D(cfg_path)
    event_time = (
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )

    logger.info("Event:             {}", event_time.isoformat())
    logger.info("nhops:             {}", args.nhops)
    logger.info("coordinate_system: {}", args.coordinate_system)

    # ── build profile ──────────────────────────────────────────────────────
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
            zmin_cfg, n_rows,
        )
    logger.info(
        "Profile altitude bounds: {:.1f} to {:.1f} km (n={})",
        float(profile.alt_km[0]), float(profile.alt_km[-1]), profile.alt_km.size,
    )

    model = RT2D(profile=profile)

    # ── elevation fan ──────────────────────────────────────────────────────
    elevs_deg = build_elevations_from_cfg(cfg)
    freqs_mhz = build_freqs_from_cfg(cfg, elevs_deg)
    freq_hz = (
        float(args.freq_mhz) * 1e6
        if args.freq_mhz is not None
        else float(freqs_mhz[len(freqs_mhz) // 2]) * 1e6
    )

    # Default s_max_km: scale with nhops so each hop has enough room.
    route_km = float(np.nanmax(profile.x_km))
    s_max_km = args.s_max_km if args.s_max_km is not None else route_km * args.nhops

    logger.info(
        "freq={:.2f} MHz, s_max_km={:.0f}, elev fan={} rays",
        freq_hz / 1e6, s_max_km, len(elevs_deg),
    )

    rays = _trace_multihop_fan(
        model=model,
        elevs_deg=elevs_deg,
        freq_hz=freq_hz,
        nhops=args.nhops,
        coordinate_system=args.coordinate_system,
        r_earth_km=float(args.r_earth_km),
        mode=args.mode,
        formulation=args.formulation,
        s_max_km=s_max_km,
    )

    # ── summary ────────────────────────────────────────────────────────────
    statuses: dict[str, int] = {}
    hops_hist: dict[int, int] = {}
    for r in rays:
        statuses[r.status] = statuses.get(r.status, 0) + 1
        hops_hist[r.nhops_completed] = hops_hist.get(r.nhops_completed, 0) + 1
    logger.info("Rays traced: {}", len(rays))
    logger.info("Statuses:    {}", statuses)
    logger.info("Hops completed histogram: {}", hops_hist)

    # ── output ─────────────────────────────────────────────────────────────
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else PROJECT_ROOT / "docs" / "examples" / "figures"
    )
    tag = f"nhops{args.nhops}_{args.coordinate_system}"
    out_file = out_dir / f"rt2d_multihop_{tag}_ray_paths.png"
    _plot_multihop_rays(profile, rays, nhops=args.nhops, out_file=out_file)


if __name__ == "__main__":
    main()
