#!/usr/bin/env python3
"""Example: O-mode Appleton-Hartree absorption (dB/km) on a 1D IRI profile."""

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

from trace.collision import ComputeCollision
from trace.model.dispersion import AppletonHartreeDispersion
from trace.model.rt1d import RT1DProfile
from trace.plottrace import setup as plot_setup
from trace.utils import load_config_1D


def _parse_freqs(text: str) -> np.ndarray:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if len(vals) != 3:
        raise ValueError("Provide exactly 3 comma-separated frequencies in MHz.")
    arr = np.asarray(vals, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("Frequencies must be > 0 MHz.")
    return arr


def _build_profile(cfg, event_time: dt.datetime, workers: int) -> RT1DProfile:
    profile = RT1DProfile.from_cfg(
        cfg=cfg,
        time=event_time,
        fetch_iri=True,
        fetch_msise=True,
        fetch_geomag=True,
        workers=max(1, int(workers)),
    )
    return profile


def _build_collision_hz(profile: RT1DProfile, workers: int) -> np.ndarray:
    lats = np.array([profile.lat], dtype=float)
    lons = np.array([profile.lon], dtype=float)
    alts = np.asarray(profile.alt_km, dtype=float)
    ne_cm3 = np.asarray(profile.ne_cm3, dtype=float)
    te = np.full_like(ne_cm3, 1000.0)
    ti = np.full_like(ne_cm3, 1000.0)
    op = 0.9 * ne_cm3
    o2p = 0.1 * ne_cm3

    cc = ComputeCollision.from_nrlmsise(
        date=profile.time,
        lats=lats,
        lons=lons,
        heights_km=alts,
        Te=te[:, None],
        Ti=ti[:, None],
        edens=ne_cm3[:, None],
        O2p=o2p[:, None],
        Op=op[:, None],
        workers=max(1, int(workers)),
        update_spaceweather=False,
        suppress_spaceweather_warning=True,
    )
    nu = np.asarray(cc.collision.nu_ft[:, 0], dtype=float)
    print(
        "Collision frequencies (nu) [Hz]: [{:.3e}, {:.3e}]".format(
            float(np.nanmin(nu)), float(np.nanmax(nu))
        )
    )
    return nu


def _diagnose_units_and_scales(
    profile: RT1DProfile,
    freqs_mhz: np.ndarray,
    collision_hz: np.ndarray,
    auto_nu_rescale: bool = True,
) -> np.ndarray:
    ne_m3 = np.asarray(profile.ne_m3, dtype=float)
    pf_mhz = RT1DProfile.den_to_plasma_freq_hz(ne_m3) / 1e6
    nu = np.asarray(collision_hz, dtype=float)

    logger.info(
        "IRI density range: [{:.3e}, {:.3e}] m^-3",
        float(np.nanmin(ne_m3)),
        float(np.nanmax(ne_m3)),
    )
    logger.info(
        "Plasma frequency range: [{:.3f}, {:.3f}] MHz",
        float(np.nanmin(pf_mhz)),
        float(np.nanmax(pf_mhz)),
    )
    logger.info(
        "Collision frequency range (nu): [{:.3e}, {:.3e}] Hz",
        float(np.nanmin(nu)),
        float(np.nanmax(nu)),
    )
    logger.info("Requested frequencies [MHz]: {}", freqs_mhz.tolist())

    if float(np.nanmax(freqs_mhz)) > (1.5 * float(np.nanmax(pf_mhz))):
        logger.warning(
            "Frequencies are well above profile peak plasma frequency. "
            "O-mode absorption can be very small in this regime."
        )

    # Safety guard for potential number-density unit mismatch in nu inputs.
    if auto_nu_rescale and float(np.nanmax(nu)) < 1.0:
        logger.warning(
            "Collision frequencies are extremely small (<1 Hz max). "
            "Applying x1e6 rescale as a unit-sanity fallback."
        )
        nu = nu * 1e6
        logger.info(
            "Rescaled collision range: [{:.3e}, {:.3e}] Hz",
            float(np.nanmin(nu)),
            float(np.nanmax(nu)),
        )
    return nu


def _compute_absorption_curves(
    profile: RT1DProfile,
    freqs_mhz: np.ndarray,
    collision_hz: np.ndarray,
) -> dict[float, np.ndarray]:
    b_t = (
        np.asarray(profile.geomag.bmag_t, dtype=float)
        if profile.geomag is not None
        else np.zeros_like(profile.ne_m3)
    )
    theta = (
        np.asarray(profile.geomag.psi_deg, dtype=float)
        if profile.geomag is not None
        else np.zeros_like(profile.ne_m3)
    )
    curves: dict[float, np.ndarray] = {}
    for f_mhz in freqs_mhz:
        model = AppletonHartreeDispersion(
            frequency_hz=float(f_mhz) * 1e6,
            ne_m3=np.asarray(profile.ne_m3, dtype=float),
            collision_hz=collision_hz,
            b_t=b_t,
            theta_deg=theta,
        )
        out = model.evaluate(mode="O")
        curves[float(f_mhz)] = np.asarray(out.absorption_db_per_km, dtype=float)
    return curves


def _plot_absorption(
    alt_km: np.ndarray,
    curves: dict[float, np.ndarray],
    pf_mhz: np.ndarray,
    nu_mhz: np.ndarray,
    out_file: Path,
) -> None:
    plot_setup(size=12)
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(9, 5), dpi=200, sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (f_mhz, y) in enumerate(sorted(curves.items(), key=lambda kv: kv[0])):
        ax.plot(
            y,
            alt_km,
            lw=2.0,
            color=colors[i % len(colors)],
            label=f"{f_mhz:.1f} MHz",
        )
    ax.set_xlabel("Absorption [dB/km]")
    ax.set_ylabel("Height / Altitude [km]")
    ax.set_title("O-mode Appleton-Hartree Absorption (IRI + NRLMSISE)")
    ax.grid(False)
    ax.legend(loc="best", fontsize=10, frameon=False)
    ax.set_ylim(float(np.nanmin(alt_km)), float(np.nanmax(alt_km)))
    x_all = np.concatenate([np.asarray(v, dtype=float) for v in curves.values()])
    x_max = float(np.nanmax(x_all)) if x_all.size else 1.0
    ax.set_xlim(0.0, max(0.01, x_max * 1.05))

    axf.plot(pf_mhz, alt_km, lw=2.0, color="#9467bd", label=r"$f_p$ (IRI)")
    print(np.min(nu_mhz), np.max(nu_mhz))
    axf.plot(
        nu_mhz,
        alt_km,
        lw=2.0,
        color="#8c564b",
        ls="--",
        label=r"$\nu$ (collision)",
    )
    axf.set_xlabel("Frequency [MHz]")
    axf.set_title("Plasma / Collision Frequency")
    axf.grid(False)
    axf.legend(loc="best", fontsize=10, frameon=False)
    fmax = max(float(np.nanmax(pf_mhz)), float(np.nanmax(nu_mhz)), 1e-3)
    # axf.set_xlim(0.0, fmax * 1.05)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: {}", out_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute O-mode Appleton-Hartree absorption from IRI profile "
            "for three frequencies and plot in dB/km."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config1D.json (default: bundled trace/cfg/config1D.json)",
    )
    parser.add_argument(
        "--event",
        default=None,
        help="UTC timestamp override; default from config event",
    )
    parser.add_argument(
        "--freqs",
        default="10,20,30",
        help="Three comma-separated frequencies in MHz (default: 10,20,30)",
    )
    parser.add_argument(
        "--out",
        default="docs/examples/figures/rt1d_omode_absorption_appleton_iri.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--no-auto-nu-rescale",
        action="store_true",
        help="Disable safety x1e6 rescale for very small collision frequencies.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve() if args.config else None
    cfg = load_config_1D(cfg_path)
    event_time = (
        dparser.isoparse(args.event) if args.event else dparser.isoparse(cfg.event)
    )
    freqs_mhz = _parse_freqs(args.freqs)
    workers = max(1, int(getattr(cfg, "worker", 1)))

    logger.info("Using config: {}", str(cfg_path) if cfg_path else "bundled config1D")
    logger.info("Event time: {}", event_time.isoformat())
    logger.info("Frequencies [MHz]: {}", freqs_mhz.tolist())

    profile = _build_profile(cfg=cfg, event_time=event_time, workers=workers)
    collision_hz = _build_collision_hz(profile=profile, workers=workers)
    collision_hz = _diagnose_units_and_scales(
        profile=profile,
        freqs_mhz=freqs_mhz,
        collision_hz=collision_hz,
        auto_nu_rescale=not args.no_auto_nu_rescale,
    )
    curves = _compute_absorption_curves(
        profile=profile,
        freqs_mhz=freqs_mhz,
        collision_hz=collision_hz,
    )
    pf_mhz = (
        RT1DProfile.den_to_plasma_freq_hz(np.asarray(profile.ne_m3, dtype=float)) / 1e6
    )
    nu_mhz = np.asarray(collision_hz, dtype=float) / 1e6
    _plot_absorption(
        alt_km=np.asarray(profile.alt_km, dtype=float),
        curves=curves,
        pf_mhz=pf_mhz,
        nu_mhz=nu_mhz,
        out_file=Path(args.out),
    )


if __name__ == "__main__":
    main()
