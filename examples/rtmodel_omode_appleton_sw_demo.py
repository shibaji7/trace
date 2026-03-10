#!/usr/bin/env python3
"""Example: 1D O-mode comparison (Appleton vs Sen-Wyller) at 3 frequencies."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dateutil import parser as dparser
from loguru import logger

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


def _plot(
    freqs_mhz: np.ndarray,
    vh_app: np.ndarray,
    vh_sw: np.ndarray,
    alt_km: np.ndarray,
    ne_m3: np.ndarray,
    out_file: Path,
) -> None:
    setup(18)
    pf_mhz = RT1D.den_to_plasma_freq_hz(ne_m3) / 1e6
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)

    ax.plot(
        freqs_mhz,
        vh_app,
        color="#D1495B",
        lw=1.8,
        label="Appleton-Hartree VH",
    )
    ax.plot(
        freqs_mhz,
        vh_sw,
        color="#2A9D8F",
        lw=1.8,
        label="Sen-Wyller VH",
    )
    ax.plot(
        pf_mhz,
        alt_km,
        color="#6D597A",
        lw=1.5,
        ls="-",
        label=r"IRI $f_p$ profile",
    )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Height / Altitude [km]")
    ax.set_title("1D O-Mode Traces + IRI Profile")
    ax.set_xlim(float(np.min(freqs_mhz)), float(np.max(freqs_mhz)))
    ax.set_ylim(float(np.nanmin(alt_km)), float(np.nanmax(alt_km)))
    ax.grid(False)
    ax.set_facecolor("#FCFCFC")

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
        description="Compare O-mode Appleton vs Sen-Wyller at 3 frequencies (1D IRI profile)."
    )
    p.add_argument("--config", default=None)
    p.add_argument("--event", default=None)
    p.add_argument("--fmin", type=float, default=1.0, help="Min frequency [MHz]")
    p.add_argument("--fmax", type=float, default=12.0, help="Max frequency [MHz]")
    p.add_argument(
        "--nfreq",
        type=int,
        default=101,
        help="Number of frequencies (use 3 for the requested 3-frequency demo).",
    )
    p.add_argument(
        "--mode",
        default="O",
        help="Propagation mode passed to both formulations (e.g., O, X, R, L).",
    )
    p.add_argument("--no-geomag", action="store_true")
    p.add_argument("--uniform-grid", action="store_true")
    p.add_argument("--nonuniform-points", type=int, default=240)
    p.add_argument("--nonuniform-sharpness", type=float, default=10.0)
    p.add_argument(
        "--out",
        default="docs/examples/figures/rt1d_omode_appleton_vs_sw.png",
    )
    args = p.parse_args()

    if args.fmax <= args.fmin:
        raise ValueError("--fmax must be greater than --fmin")
    if args.nfreq < 1:
        raise ValueError("--nfreq must be >= 1")
    if args.nonuniform_points < 3:
        raise ValueError("--nonuniform-points must be >= 3")
    if args.nonuniform_sharpness < 0:
        raise ValueError("--nonuniform-sharpness must be >= 0")
    freqs_mhz = np.linspace(float(args.fmin), float(args.fmax), int(args.nfreq))

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = load_config_1D(config_path)
    event_time = _parse_event(args.event, cfg.event)

    rt = RT1D(
        cfg=cfg,
        time=event_time,
        fetch_iri=True,
        fetch_geomag=not args.no_geomag,
        fetch_msise=False,
        workers=max(1, int(getattr(cfg, "worker", 1))),
    )
    out_a = rt.NVIS_tracer(
        freq_mhz=freqs_mhz,
        mode=args.mode,
        formulation="appleton",
        use_nonuniform_grid=not args.uniform_grid,
        nonuniform_points=int(args.nonuniform_points),
        nonuniform_sharpness=float(args.nonuniform_sharpness),
    )
    out_s = rt.NVIS_tracer(
        freq_mhz=freqs_mhz,
        mode=args.mode,
        formulation="senwyller",
        use_nonuniform_grid=not args.uniform_grid,
        nonuniform_points=int(args.nonuniform_points),
        nonuniform_sharpness=float(args.nonuniform_sharpness),
    )
    vh_app = np.asarray(out_a.vh_km, dtype=float)
    vh_sw = np.asarray(out_s.vh_km, dtype=float)
    alt_km = np.asarray(rt.profile.alt_km, dtype=float)
    ne_m3 = np.asarray(rt.profile.ne_m3, dtype=float)

    out_file = (PROJECT_ROOT / args.out).resolve()
    _plot(freqs_mhz, vh_app, vh_sw, alt_km, ne_m3, out_file)

    logger.info("Frequencies [MHz]: {}", freqs_mhz)
    logger.info("Appleton O-mode VH [km]: {}", vh_app)
    logger.info("Sen-Wyller O-mode VH [km]: {}", vh_sw)
    logger.info("Saved plot: {}", out_file)


if __name__ == "__main__":
    main()
