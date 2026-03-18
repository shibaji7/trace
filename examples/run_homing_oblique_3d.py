#!/usr/bin/env python3
"""
Example: Oblique HF link homing using Homing3D on an IRI-2016 3-D profile.

What this script does
---------------------
1. Builds a 3-D IRI ionospheric volume covering the region between a
   transmitter (TX) and a target receiver (RX).
2. Uses ``Homing3D`` to find all (azimuth, elevation) pairs whose ray
   lands within a user-specified radius around the RX.
3. Plots:
     (a) A map showing TX, RX, and all homed ray landing points.
     (b) A virtual-height vs frequency trace for the best path per azimuth.

Usage
-----
    python examples/run_homing_oblique_3d.py \
        --date 2017-05-27T18:00 \
        --tx-lat 40.0 --tx-lon -95.0 \
        --rx-lat 45.0 --rx-lon -85.0 \
        --tol 30

The script writes to ``./output/`` by default.

Requirements
------------
    hfpytrace, IRI-2016 coefficients, cartopy (for the map plot)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfpytrace.homing import Homing3D, HomingConfig, HomingResult
from hfpytrace.model.rt3d import RT3D, RT3DProfile


# ─────────────────────────────────────────────────────────────────────────── #
#  CLI                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--date", default="2017-05-27T18:00",
                   help="ISO event time (default: 2017-05-27T18:00)")
    p.add_argument("--tx-lat", type=float, default=40.0,
                   help="Transmitter latitude [°N] (default: 40.0)")
    p.add_argument("--tx-lon", type=float, default=-95.0,
                   help="Transmitter longitude [°E] (default: -95.0)")
    p.add_argument("--rx-lat", type=float, default=45.0,
                   help="Target RX latitude [°N] (default: 45.0)")
    p.add_argument("--rx-lon", type=float, default=-85.0,
                   help="Target RX longitude [°E] (default: -85.0)")
    p.add_argument("--tol", type=float, default=30.0,
                   help="Homing acceptance radius [km] (default: 30.0)")
    p.add_argument("--fmin", type=float, default=3.0,
                   help="Minimum frequency [MHz] (default: 3.0)")
    p.add_argument("--fmax", type=float, default=10.0,
                   help="Maximum frequency [MHz] (default: 10.0)")
    p.add_argument("--fstep", type=float, default=0.2,
                   help="Frequency step [MHz] (default: 0.2)")
    p.add_argument("--az-step", type=float, default=5.0,
                   help="Azimuth sweep step [°] (default: 5.0)")
    p.add_argument("--el-step", type=float, default=3.0,
                   help="Elevation sweep step [°] (default: 3.0)")
    p.add_argument("--out", default="./output",
                   help="Output directory (default: ./output)")
    p.add_argument("--workers", type=int, default=4,
                   help="IRI fetch worker threads (default: 4)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────── #
#  Profile builder                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def build_profile(
    time: dt.datetime,
    tx_lat: float, tx_lon: float,
    rx_lat: float, rx_lon: float,
    pad_deg: float = 5.0,
    workers: int = 4,
) -> RT3DProfile:
    """
    Build a 3-D IRI volume that covers the TX→RX path plus *pad_deg* margin.

    Grid resolution: 0.5° lat × 0.5° lon × 5 km altitude (80–500 km).
    Reduce the steps for faster (but coarser) runs.
    """
    lat_lo = min(tx_lat, rx_lat) - pad_deg
    lat_hi = max(tx_lat, rx_lat) + pad_deg
    lon_lo = min(tx_lon, rx_lon) - pad_deg
    lon_hi = max(tx_lon, rx_lon) + pad_deg

    lats = np.arange(lat_lo, lat_hi + 0.5, 0.5)
    lons = np.arange(lon_lo, lon_hi + 0.5, 0.5)
    alts = np.arange(80.0, 501.0, 5.0)

    profile = RT3DProfile(lats=lats, lons=lons, alts_km=alts, time=time)
    profile.fetch_iri(workers=workers)
    return profile


# ─────────────────────────────────────────────────────────────────────────── #
#  Plotting                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def plot_map(
    rays: list[HomingResult],
    tx_lat: float, tx_lon: float,
    rx_lat: float, rx_lon: float,
    tol_km: float,
    out_path: Path,
) -> None:
    """Map of TX, RX acceptance circle, and all homed ray landing points."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=proj)

        ax.add_feature(cfeature.LAND, facecolor="lightgrey")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.STATES, linewidth=0.2)

        # TX
        ax.plot(tx_lon, tx_lat, "r^", ms=10, transform=proj, label="TX", zorder=5)
        # RX + tolerance circle (approximate: 1° ≈ 111 km)
        ax.plot(rx_lon, rx_lat, "bs", ms=10, transform=proj, label="RX", zorder=5)
        tol_deg = tol_km / 111.0
        circle = Circle((rx_lon, rx_lat), tol_deg, fill=False,
                         edgecolor="blue", linewidth=1.5, linestyle="--",
                         transform=proj, zorder=4)
        ax.add_patch(circle)

        # Landing points
        if rays:
            lats = [r.landing_lat for r in rays]
            lons = [r.landing_lon for r in rays]
            freqs = np.array([r.freq_hz for r in rays]) / 1e6
            sc = ax.scatter(lons, lats, c=freqs, cmap="plasma",
                            s=20, transform=proj, zorder=6, label="Homed landings")
            plt.colorbar(sc, ax=ax, label="Frequency (MHz)", shrink=0.7)

        # Route line
        ax.plot([tx_lon, rx_lon], [tx_lat, rx_lat], "k--", lw=0.8,
                transform=proj, label="Great-circle path")

        lon_margin = abs(rx_lon - tx_lon) * 0.3 + 3
        lat_margin = abs(rx_lat - tx_lat) * 0.3 + 3
        ax.set_extent([
            min(tx_lon, rx_lon) - lon_margin, max(tx_lon, rx_lon) + lon_margin,
            min(tx_lat, rx_lat) - lat_margin, max(tx_lat, rx_lat) + lat_margin,
        ], crs=proj)

        ax.set_title(f"3-D Homing: Landing Points (tol = {tol_km} km)")
        ax.legend(loc="lower left", fontsize=8)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved map → {out_path}")

    except ImportError:
        print("cartopy not available – skipping map plot.")


def plot_vh_vs_freq(
    iono: np.ndarray,
    tx_lat: float, tx_lon: float,
    rx_lat: float, rx_lon: float,
    out_path: Path,
) -> None:
    """Virtual-height trace vs frequency for all homed paths."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if iono.shape[0] > 0:
        freq_mhz = iono[:, 0] / 1e6
        vh_km = iono[:, 1]
        az_deg = iono[:, 2]
        sc = ax.scatter(freq_mhz, vh_km, c=az_deg, cmap="hsv",
                        vmin=0, vmax=360, s=8)
        plt.colorbar(sc, ax=ax, label="Launch azimuth (°)")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Virtual Height h' (km)")
    ax.set_title(
        f"Synthetic Oblique Ionogram\nTX=({tx_lat:.1f}°N,{tx_lon:.1f}°E) "
        f"→ RX=({rx_lat:.1f}°N,{rx_lon:.1f}°E)"
    )
    ax.set_ylim([80, 700])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved h'-vs-f plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_time = dt.datetime.fromisoformat(args.date)
    freqs_hz = np.arange(args.fmin, args.fmax + args.fstep / 2, args.fstep) * 1e6

    print(f"Event time : {event_time}")
    print(f"TX         : ({args.tx_lat:.1f}°N, {args.tx_lon:.1f}°E)")
    print(f"RX target  : ({args.rx_lat:.1f}°N, {args.rx_lon:.1f}°E)")
    print(f"Tolerance  : {args.tol} km radius")
    print(f"Frequency  : {args.fmin}–{args.fmax} MHz (step {args.fstep} MHz)")
    print()

    # ── Build 3-D profile ─────────────────────────────────────────────────
    print("Fetching IRI-2016 3-D profile …")
    profile = build_profile(
        event_time,
        args.tx_lat, args.tx_lon,
        args.rx_lat, args.rx_lon,
        workers=args.workers,
    )
    model = RT3D(profile=profile)

    # ── Configure homing ─────────────────────────────────────────────────
    cfg = HomingConfig(
        tol_km=args.tol,
        az_min_deg=0.0,
        az_max_deg=360.0,
        az_step_deg=args.az_step,
        elev_min_deg=2.0,
        elev_max_deg=89.0,
        elev_step_deg=args.el_step,
        fine_points=1000,
        max_roots_per_az=5,
        mode="O",
    )
    homing = Homing3D(
        model,
        tx_lat=args.tx_lat,
        tx_lon=args.tx_lon,
        config=cfg,
        coordinate_system="spherical",
        solver="gradient",
        trace_kw=dict(s_max_km=6000.0, max_step_km=5.0),
    )

    # ── Sweep frequencies ─────────────────────────────────────────────────
    print(f"Homing at {len(freqs_hz)} frequencies "
          f"({int(360 / args.az_step)} azimuths × "
          f"{int(87 / args.el_step)} elevations each) …")

    all_rays: list[HomingResult] = []
    iono_rows: list[tuple] = []

    for f_hz in freqs_hz:
        rays = homing.home(f_hz, target_lat=args.rx_lat, target_lon=args.rx_lon)
        all_rays.extend(rays)
        for r in rays:
            iono_rows.append((f_hz, r.virtual_height_km, r.azimuth_deg,
                              r.elevation_deg, r.landing_lat, r.landing_lon))
        if rays:
            print(f"  {f_hz/1e6:.2f} MHz → {len(rays)} path(s), "
                  f"h' = {[f'{r.virtual_height_km:.0f}' for r in rays]} km")

    iono = np.array(iono_rows, dtype=float) if iono_rows else np.empty((0, 6))
    print(f"\nTotal homed rays across all frequencies: {len(all_rays)}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_map(all_rays, args.tx_lat, args.tx_lon, args.rx_lat, args.rx_lon,
             args.tol, out_dir / "homing_3d_map.png")
    plot_vh_vs_freq(iono, args.tx_lat, args.tx_lon, args.rx_lat, args.rx_lon,
                    out_dir / "homing_3d_ionogram.png")


if __name__ == "__main__":
    main()
