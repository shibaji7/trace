#!/usr/bin/env python3
"""Example: 1D NVIS O/X-mode tracing from IRI profile using config1D."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser as dparser
from loguru import logger

# Ensure local project package is imported instead of stdlib `trace`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfpytrace.model.rt1d import RT1D
from hfpytrace.plottrace import setup
from hfpytrace.utils import load_config_1D


def _parse_event(event_text: str | None, cfg_event: str) -> dt.datetime:
    t = dparser.isoparse(event_text) if event_text else dparser.isoparse(cfg_event)
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t


def _plot_results(
    rt: RT1D,
    freqs_mhz: np.ndarray,
    o_res,
    x_res,
    out_file: Path,
) -> None:
    setup(18)
    alt_km = np.asarray(rt.profile.alt_km, dtype=float)
    ne_m3 = np.asarray(rt.profile.ne_m3, dtype=float)
    pf_mhz = RT1D.den_to_plasma_freq_hz(ne_m3) / 1e6

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    ax.plot(freqs_mhz, o_res.vh_km, color="#D1495B", lw=1.5, label="O-mode trace")
    ax.plot(freqs_mhz, x_res.vh_km, color="#2A9D8F", lw=1.5, label="X-mode trace")
    ax.plot(
        pf_mhz,
        alt_km,
        color="#6D597A",
        lw=1.4,
        ls="-",
        label=r"IRI $f_p$ profile",
    )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Height / Altitude [km]")
    ax.set_title("1D NVIS Traces + IRI Profile", pad=10)
    ax.set_xlim(float(np.min(freqs_mhz)), float(np.max(freqs_mhz)))
    y_min = float(np.nanmin(alt_km))
    y_max = float(np.nanmax(alt_km))
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    ax.set_facecolor("#FCFCFC")
    ax.set_xlim(1, 8)

    ax_top = ax.twiny()
    ax_top.semilogx(
        ne_m3,
        alt_km,
        color="#355070",
        lw=1.8,
        ls="-",
        label=r"IRI $N_e$ profile",
    )
    ax_top.set_xlabel(r"Electron density $N_e$ [m$^{-3}$]")
    ne_min = max(1.0, float(np.nanmin(ne_m3[np.isfinite(ne_m3)])))
    ne_max = float(np.nanmax(ne_m3[np.isfinite(ne_m3)]))
    ax_top.set_xlim(ne_min, ne_max * 1.05)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_top.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=False,
        fontsize=12,
        handlelength=2.4,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run 1D NVIS tracer for O/X modes with IRI profile from config1D."
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to JSON config. If omitted, uses installed hfpytrace/cfg/config1D.json",
    )
    p.add_argument(
        "--event",
        default=None,
        help="UTC timestamp override, e.g. 2017-05-27T16:00:00Z (default from config)",
    )
    p.add_argument("--fmin", type=float, default=1.0, help="Min frequency [MHz]")
    p.add_argument("--fmax", type=float, default=12.0, help="Max frequency [MHz]")
    p.add_argument(
        "--nfreq", type=int, default=20001, help="Number of frequency samples"
    )
    p.add_argument(
        "--formulation",
        default="appleton",
        choices=("appleton", "senwyller"),
        help="Dispersion relation backend",
    )
    p.add_argument(
        "--no-geomag",
        action="store_true",
        help="Skip geomagnetic fetch (O/X will be less distinct).",
    )
    p.add_argument(
        "--out",
        default="docs/examples/figures/rt1d_nvis_ox_iri.png",
        help="Output plot path",
    )
    p.add_argument(
        "--uniform-grid",
        action="store_true",
        help="Disable nonuniform vertical regridding in NVIS tracer.",
    )
    p.add_argument(
        "--nonuniform-points",
        type=int,
        default=240,
        help="Number of stretched vertical points when nonuniform grid is enabled.",
    )
    p.add_argument(
        "--nonuniform-sharpness",
        type=float,
        default=10.0,
        help="Stretch sharpness for nonuniform grid (higher => denser near turning height).",
    )
    args = p.parse_args()

    if args.fmax <= args.fmin:
        raise ValueError("--fmax must be greater than --fmin")
    if args.nfreq < 3:
        raise ValueError("--nfreq must be >= 3")
    if args.nonuniform_points < 3:
        raise ValueError("--nonuniform-points must be >= 3")
    if args.nonuniform_sharpness < 0:
        raise ValueError("--nonuniform-sharpness must be >= 0")

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = load_config_1D(config_path)
    event_time = _parse_event(args.event, cfg.event)

    logger.info(
        "Using config: {}",
        config_path if config_path else "installed default config1D.json",
    )
    logger.info("Event (UTC naive): {}", event_time.isoformat())

    rt = RT1D(
        cfg=cfg,
        time=event_time,
        fetch_iri=True,
        fetch_geomag=not args.no_geomag,
        fetch_msise=False,
        workers=max(1, int(getattr(cfg, "worker", 1))),
    )

    freqs_mhz = np.linspace(args.fmin, args.fmax, args.nfreq)
    o_res = rt.NVIS_tracer(
        freq_mhz=freqs_mhz,
        mode="O",
        formulation=args.formulation,
        use_nonuniform_grid=not args.uniform_grid,
        nonuniform_points=int(args.nonuniform_points),
        nonuniform_sharpness=float(args.nonuniform_sharpness),
    )
    x_res = rt.NVIS_tracer(
        freq_mhz=freqs_mhz,
        mode="X",
        formulation=args.formulation,
        use_nonuniform_grid=not args.uniform_grid,
        nonuniform_points=int(args.nonuniform_points),
        nonuniform_sharpness=float(args.nonuniform_sharpness),
    )

    out_file = (PROJECT_ROOT / args.out).resolve()
    _plot_results(rt, freqs_mhz, o_res, x_res, out_file)

    o_ok = int(np.isfinite(o_res.vh_km).sum())
    x_ok = int(np.isfinite(x_res.vh_km).sum())
    ne_peak = float(np.nanmax(rt.profile.ne_m3))
    pf_peak = float(np.nanmax(RT1D.den_to_plasma_freq_hz(rt.profile.ne_m3) / 1e6))
    # Diagnostic for common unit confusion: cm^-3 treated as m^-3.
    pf_peak_if_cm3 = float(
        np.nanmax(RT1D.den_to_plasma_freq_hz(rt.profile.ne_m3 * 1e6) / 1e6)
    )
    logger.info("O-mode finite points: {}/{}", o_ok, freqs_mhz.size)
    logger.info("X-mode finite points: {}/{}", x_ok, freqs_mhz.size)
    logger.info("Profile peak density: {:.3e} m^-3", ne_peak)
    logger.info("Profile peak plasma frequency: {:.3f} MHz", pf_peak)
    logger.info(
        "Unit check (if m^-3 were accidentally cm^-3): "
        "equivalent peak would be {:.3f} MHz",
        pf_peak_if_cm3,
    )
    logger.info("Saved plot: {}", out_file)


if __name__ == "__main__":
    main()
