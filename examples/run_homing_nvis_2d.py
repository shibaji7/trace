#!/usr/bin/env python3
"""
Example: NVIS ionogram synthesis using Homing2D on an IRI-2016 profile.

What this script does
---------------------
1. Builds a 2-D ionospheric slice along a short meridional route using IRI.
2. Constructs an ``RT2D`` tracer from that profile.
3. Uses ``Homing2D`` to find all ray paths that return to the transmitter
   (ground range = 0, i.e. vertical / near-vertical incidence) at each
   operating frequency.
4. Plots the synthetic ionogram (frequency vs virtual height h') alongside
   the IRI electron-density profile.

Usage
-----
    python examples/run_homing_nvis_2d.py [--date YYYY-MM-DDTHH:MM] [--out DIR]

The script writes two files to *out* (default ``./output``):
    nvis_ionogram_2d.png   – synthetic ionogram (O-mode pixels)
    nvis_profile_2d.png    – electron density cross-section with homed rays

Requirements
------------
    hfpytrace, IRI-2016 coefficients accessible via ``hfpytrace.density.iri``
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

from hfpytrace.homing import Homing2D, HomingConfig
from hfpytrace.model.rt2d import RT2D, RT2DProfile


# ─────────────────────────────────────────────────────────────────────────── #
#  CLI                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2017-05-27T18:00",
                   help="ISO event time (default: 2017-05-27T18:00)")
    p.add_argument("--lat", type=float, default=40.0,
                   help="Transmitter / ionosonde latitude [°N] (default: 40.0)")
    p.add_argument("--lon", type=float, default=-95.0,
                   help="Transmitter / ionosonde longitude [°E] (default: -95.0)")
    p.add_argument("--fmin", type=float, default=2.0,
                   help="Minimum sounding frequency [MHz] (default: 2.0)")
    p.add_argument("--fmax", type=float, default=12.0,
                   help="Maximum sounding frequency [MHz] (default: 12.0)")
    p.add_argument("--fstep", type=float, default=0.1,
                   help="Frequency step [MHz] (default: 0.1)")
    p.add_argument("--tol", type=float, default=15.0,
                   help="Homing tolerance [km] (default: 15.0)")
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
    center_lat: float,
    center_lon: float,
    workers: int = 4,
) -> RT2DProfile:
    """
    Build a short meridional 2-D IRI profile centred on the ionosonde.

    The route spans ±3° latitude (~330 km) and has 2 km height resolution
    from 60 km to 500 km.  For pure NVIS the route only needs to be wide
    enough to accommodate the steepest oblique rays.
    """
    n_pts = 61
    lats = np.linspace(center_lat - 3.0, center_lat + 3.0, n_pts)
    lons = np.full(n_pts, center_lon)
    alt_km = np.arange(60.0, 501.0, 2.0)

    profile = RT2DProfile(alt_km=alt_km, lats=lats, lons=lons, time=time)
    profile.fetch_iri(workers=workers)
    return profile


# ─────────────────────────────────────────────────────────────────────────── #
#  Plotting                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def plot_ionogram(iono: np.ndarray, out_path: Path) -> None:
    """Plot synthetic ionogram: freq [MHz] on X-axis, h' [km] on Y-axis."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if iono.shape[0] > 0:
        freq_mhz = iono[:, 0] / 1e6
        vh_km = iono[:, 1]
        ax.scatter(freq_mhz, vh_km, s=4, c="steelblue", label="O-mode")

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Virtual Height h' (km)")
    ax.set_title("Synthetic NVIS Ionogram (Homing2D + IRI-2016)")
    ax.set_xlim([iono[:, 0].min() / 1e6 - 0.5, iono[:, 0].max() / 1e6 + 0.5] if iono.shape[0] else [0, 15])
    ax.set_ylim([80, 600])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved ionogram → {out_path}")


def plot_profile_with_rays(
    profile: RT2DProfile,
    homed_rays_by_freq: dict,
    out_path: Path,
    n_freq: int = 5,
) -> None:
    """
    Plot the electron-density cross-section with a sample of homed ray paths.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 5))

    # Density background
    X, Z = np.meshgrid(profile.x_km, profile.alt_km)
    ne_cm3 = np.where(profile.ne_cm3 > 0, profile.ne_cm3, np.nan)
    pcm = ax.pcolormesh(X, Z, np.log10(ne_cm3), cmap="viridis",
                        vmin=3, vmax=6, shading="auto")
    plt.colorbar(pcm, ax=ax, label="log₁₀ Nₑ (cm⁻³)")

    # Overlay sample homed rays
    freqs = sorted(homed_rays_by_freq.keys())
    sample_freqs = freqs[::max(1, len(freqs) // n_freq)][:n_freq]
    colours = cm.plasma(np.linspace(0.1, 0.9, len(sample_freqs)))
    for f, col in zip(sample_freqs, colours):
        for ray in homed_rays_by_freq[f]:
            ax.plot(ray.x_km, ray.z_km, color=col, lw=0.8, alpha=0.7)

    ax.set_xlabel("Ground range (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("IRI-2016 Electron Density + Homed Ray Paths")
    ax.set_ylim([60, 500])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved profile plot → {out_path}")


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
    print(f"Ionosonde  : ({args.lat:.1f}°N, {args.lon:.1f}°E)")
    print(f"Frequency  : {args.fmin}–{args.fmax} MHz (step {args.fstep} MHz)")
    print(f"Tolerance  : ±{args.tol} km")
    print()

    # ── Build profile and model ───────────────────────────────────────────
    print("Fetching IRI-2016 profile …")
    profile = build_profile(event_time, args.lat, args.lon, workers=args.workers)
    model = RT2D(profile=profile)

    # ── Configure homing ─────────────────────────────────────────────────
    cfg = HomingConfig(
        tol_km=args.tol,
        elev_min_deg=0.0,
        elev_max_deg=89.0,
        elev_step_deg=2.0,     # 2° coarse step → smooth D(φ) spline
        fine_points=2000,
        max_roots=10,
        mode="O",
    )
    homing = Homing2D(
        model,
        config=cfg,
        trace_kw=dict(
            x0_km=float(profile.x_km[profile.x_km.size // 2]),   # launch from centre
            z0_km=float(profile.alt_km[0]),
            s_max_km=3000.0,
            formulation="appleton",
            max_step_km=2.0,
        ),
    )

    # ── Sweep frequencies ────────────────────────────────────────────────
    print(f"Homing at {len(freqs_hz)} frequencies …")
    all_rows: list = []
    homed_by_freq: dict = {}

    for f_hz in freqs_hz:
        rays = homing.home(freq_hz=f_hz)
        homed_by_freq[f_hz] = rays
        for r in rays:
            all_rows.append((f_hz, r.virtual_height_km, r.elevation_deg,
                             r.ground_range_km, r.miss_km))

    iono = np.array(all_rows, dtype=float) if all_rows else np.empty((0, 5))
    print(f"Total ionogram pixels: {iono.shape[0]}")

    # ── Plots ────────────────────────────────────────────────────────────
    plot_ionogram(iono, out_dir / "nvis_ionogram_2d.png")
    plot_profile_with_rays(profile, homed_by_freq, out_dir / "nvis_profile_2d.png")


if __name__ == "__main__":
    main()
