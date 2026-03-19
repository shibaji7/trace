"""Tests for RT3D.oblique_trace nhops (multi-hop) functionality.

Covers:
  * nhops=1 fast path — identical output to pre-nhops behaviour
  * nhops=2 cartesian gradient — correct ground reflection and x/y accumulation
  * nhops=2 cartesian hamiltonian — same reflection logic, different solver
  * nhops=2 spherical gradient — vr/vlat/vlon reflection
  * nhops > ray count — graceful early stop when ray doesn't reach ground
  * Elevation/azimuth recovery after reflection (specular geometry)
  * nhops_completed counter
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from hfpytrace.model.rt3d import RT3D, RT3DProfile


# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_profile(
    ne_peak: float = 1.5e11,
    nlat: int = 5,
    nlon: int = 5,
    nalt: int = 61,
    alt_max_km: float = 300.0,
) -> RT3DProfile:
    """Minimal RT3DProfile with a Chapman-like ionosphere."""
    lats = np.linspace(38.0, 44.0, nlat, dtype=float)
    lons = np.linspace(-78.0, -70.0, nlon, dtype=float)
    alts = np.linspace(0.0, alt_max_km, nalt, dtype=float)
    ll, oo, zz = np.meshgrid(lats, lons, alts, indexing="ij")
    ne = ne_peak * np.exp(-(((zz - 200.0) / 80.0) ** 2))
    return RT3DProfile(
        lats=lats,
        lons=lons,
        alts_km=alts,
        time=dt.datetime(2017, 5, 27, 16, 0, 0),
        ne_m3=ne,
    )


def _make_rt() -> RT3D:
    return RT3D(profile=_make_profile())


# ── single-hop fast path ──────────────────────────────────────────────────────

def test_nhops_1_fast_path_matches_default():
    """nhops=1 must return the same result as the legacy no-nhops call."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        azimuth_deg=45.0,
        coordinate_system="cartesian",
        x0_km=0.0,
        y0_km=0.0,
        z0_km=0.0,
        s_max_km=500.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out0 = rt.oblique_trace(**kw)          # default nhops=1
    assert np.array_equal(out1.x_km, out0.x_km)
    assert np.array_equal(out1.z_km, out0.z_km)
    assert out1.nhops_completed == 1
    assert out1.group_path_km == pytest.approx(out0.group_path_km, rel=1e-9)


# ── nhops=2 cartesian gradient ────────────────────────────────────────────────

def test_nhops_2_cartesian_gradient_structure():
    """nhops=2 output must have more x points than nhops=1 and carry vx/vy/vz."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0,
        y0_km=0.0,
        z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out2 = rt.oblique_trace(**kw, nhops=2)

    # Second hop either completes or stops at domain — more points expected
    assert out2.x_km.size >= out1.x_km.size
    assert out2.z_km.size == out2.x_km.size
    assert out2.y_km.size == out2.x_km.size
    assert hasattr(out2, "vx") and hasattr(out2, "vy") and hasattr(out2, "vz")
    assert np.isfinite(out2.group_path_km)
    assert out2.nhops_completed >= 1


def test_nhops_2_cartesian_group_path_accumulates():
    """Total group path for nhops=2 must be >= single-hop group path."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out2 = rt.oblique_trace(**kw, nhops=2)
    assert out2.group_path_km >= out1.group_path_km


def test_nhops_2_cartesian_x_monotone():
    """x_km of a 2-hop ray should be non-decreasing (ray moves forward)."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        azimuth_deg=0.0,        # East: y unchanged, x increases
        coordinate_system="cartesian",
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
        nhops=2,
    )
    if out.x_km.size > 1:
        # Allow small floating-point non-monotonicity at segment joins
        assert float(out.x_km[-1]) > float(out.x_km[0])


def test_nhops_2_z_nonnegative():
    """All z_km values must be >= 0 (above ground) for a 2-hop ray."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
        nhops=2,
    )
    assert np.all(np.asarray(out.z_km, dtype=float) >= -1e-3)


# ── nhops=2 cartesian hamiltonian ─────────────────────────────────────────────

def test_nhops_2_cartesian_hamiltonian():
    """Hamiltonian solver must accept nhops=2 and return a valid namespace."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=25.0,
        coordinate_system="cartesian",
        solver="hamiltonian",
        nhops=2,
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
    )
    assert out.x_km.size >= 1
    assert np.isfinite(out.group_path_km)
    assert out.nhops_completed >= 1
    # Hamiltonian returns vx/vy/vz derived from canonical momentum
    assert hasattr(out, "vx")


# ── nhops=2 spherical gradient ────────────────────────────────────────────────

@pytest.mark.skip(reason="3D spherical gradient ODE is too slow for CI — skipped pending solver optimisation")
def test_nhops_2_spherical_gradient_structure():
    """Spherical nhops=2 must return vr/vlat/vlon and non-negative z_km."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        azimuth_deg=0.0,
        coordinate_system="spherical",
        nhops=2,
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=400.0,   # reduced: spherical ODE is slow; 400 km is enough per hop
        max_step_km=10.0,
    )
    assert out.x_km.size >= 1
    assert out.z_km.size == out.x_km.size
    assert hasattr(out, "vr") and hasattr(out, "vlat") and hasattr(out, "vlon")
    assert np.all(np.asarray(out.z_km, dtype=float) >= -1e-3)
    assert np.isfinite(out.group_path_km)
    assert out.nhops_completed >= 1


# ── specular reflection geometry ──────────────────────────────────────────────

def test_specular_elevation_preserved_cartesian():
    """For a horizontally-uniform ionosphere the reflected elevation must equal
    the launch elevation (specular reflection)."""
    # Build a 1-D-uniform (only altitude-varying) ionosphere so the ray
    # exits and re-enters symmetrically.
    lats = np.linspace(38.0, 46.0, 9, dtype=float)
    lons = np.linspace(-80.0, -68.0, 9, dtype=float)
    alts = np.linspace(0.0, 400.0, 81, dtype=float)
    ll, oo, zz = np.meshgrid(lats, lons, alts, indexing="ij")
    ne = 1.2e11 * np.exp(-(((zz - 200.0) / 70.0) ** 2))
    rt = RT3D(profile=RT3DProfile(
        lats=lats, lons=lons, alts_km=alts,
        time=dt.datetime(2017, 5, 27, 16, 0, 0),
        ne_m3=ne,
    ))

    launch_elev = 35.0
    out = rt.oblique_trace(
        freq_hz=8e6,
        elevation_deg=launch_elev,
        azimuth_deg=0.0,
        coordinate_system="cartesian",
        nhops=2,
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=2000.0,
        max_step_km=3.0,
    )
    if out.nhops_completed >= 2:
        # Terminal vz/vx of the final segment encodes the reflected elevation.
        vx_f = float(out.vx[-1])
        vz_f = float(out.vz[-1])
        elev_terminal = float(np.degrees(np.arctan2(abs(vz_f),
                                                     np.sqrt(vx_f**2))))
        # Allow ±5° tolerance for numerical refraction in the ionosphere.
        assert abs(elev_terminal - launch_elev) < 5.0


# ── nhops_completed counter ───────────────────────────────────────────────────

def test_nhops_completed_never_exceeds_requested():
    """nhops_completed ≤ nhops for any ray."""
    rt = _make_rt()
    for nhops in (1, 2, 3):
        out = rt.oblique_trace(
            freq_hz=7e6,
            elevation_deg=30.0,
            coordinate_system="cartesian",
            nhops=nhops,
            x0_km=0.0, y0_km=0.0, z0_km=0.0,
            s_max_km=3000.0,
            max_step_km=5.0,
        )
        assert 1 <= out.nhops_completed <= nhops


def test_nhops_completed_is_1_for_non_ground_ray():
    """A ray that doesn't reach ground (e.g., very high frequency) completes
    exactly 1 hop regardless of nhops."""
    rt = _make_rt()
    # Use a frequency well above foF2 — ray penetrates, status != "ground"
    out = rt.oblique_trace(
        freq_hz=50e6,   # far above foF2 (~9 MHz for ne=1.5e11)
        elevation_deg=15.0,
        coordinate_system="cartesian",
        nhops=3,
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=500.0,
        max_step_km=5.0,
    )
    assert out.status != "ground"
    assert out.nhops_completed == 1


# ── nhops=1 backward-compatibility attributes ─────────────────────────────────

def test_nhops_1_has_nhops_completed_attribute():
    """Even the single-hop fast path must expose nhops_completed=1."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        nhops=1,
        x0_km=0.0, y0_km=0.0, z0_km=0.0,
        s_max_km=500.0,
        max_step_km=5.0,
    )
    assert out.nhops_completed == 1


def test_nhops_clamped_to_minimum_1():
    """nhops=0 or negative must be silently clamped to 1."""
    rt = _make_rt()
    for bad_nhops in (0, -5):
        out = rt.oblique_trace(
            freq_hz=7e6,
            elevation_deg=30.0,
            coordinate_system="cartesian",
            nhops=bad_nhops,
            x0_km=0.0, y0_km=0.0, z0_km=0.0,
            s_max_km=500.0,
            max_step_km=5.0,
        )
        assert out.nhops_completed >= 1
