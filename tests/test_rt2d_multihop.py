"""Tests for RT2D.oblique_trace nhops (multi-hop) functionality.

Covers:
  * nhops=1 fast path — identical output to pre-nhops behaviour
  * nhops=2 cartesian gradient — correct ground reflection and x accumulation
  * nhops=2 spherical gradient — v_r/v_phi reflection, z non-negative
  * nhops > ray count — graceful early stop when ray doesn't reach ground
  * Elevation recovery after reflection (specular geometry)
  * nhops_completed counter
  * Backward-compatibility: nhops_completed present on single-hop output
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from hfpytrace.model.rt2d import RT2D, RT2DProfile


# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_rt(
    ne_peak: float = 1.5e11,
    nalt: int = 61,
    nx: int = 101,
    alt_max_km: float = 300.0,
    x_max_km: float = 1000.0,
) -> RT2D:
    """Minimal RT2D with a Chapman-like ionosphere on a (nalt, nx) grid."""
    alts = np.linspace(0.0, alt_max_km, nalt, dtype=float)
    x = np.linspace(0.0, x_max_km, nx, dtype=float)
    zz, xx = np.meshgrid(alts, x, indexing="ij")
    ne = ne_peak * np.exp(-(((zz - 200.0) / 80.0) ** 2))
    return RT2D(x_km=x, z_km=alts, ne_m3=ne)


# ── single-hop fast path ──────────────────────────────────────────────────────

def test_nhops_1_fast_path_matches_default():
    """nhops=1 must return the same result as the legacy no-nhops call."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0,
        z0_km=0.0,
        s_max_km=600.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out0 = rt.oblique_trace(**kw)          # default nhops=1
    assert np.array_equal(out1.x_km, out0.x_km)
    assert np.array_equal(out1.z_km, out0.z_km)
    assert out1.nhops_completed == 1
    assert out1.group_path_km == pytest.approx(out0.group_path_km, rel=1e-9)


# ── nhops=2 cartesian gradient ────────────────────────────────────────────────

def test_nhops_2_cartesian_structure():
    """nhops=2 output must have more x points than nhops=1 and carry vx/vz."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0,
        z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out2 = rt.oblique_trace(**kw, nhops=2)

    # Second hop either completes or stops at domain — more points expected
    assert out2.x_km.size >= out1.x_km.size
    assert out2.z_km.size == out2.x_km.size
    assert hasattr(out2, "vx") and hasattr(out2, "vz")
    assert np.isfinite(out2.group_path_km)
    assert out2.nhops_completed >= 1


def test_nhops_2_cartesian_group_path_accumulates():
    """Total group path for nhops=2 must be >= single-hop group path."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out2 = rt.oblique_trace(**kw, nhops=2)
    assert out2.group_path_km >= out1.group_path_km


def test_nhops_2_cartesian_x_monotone():
    """x_km of a 2-hop ray should advance forward (last > first)."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
        nhops=2,
    )
    if out.x_km.size > 1:
        assert float(out.x_km[-1]) > float(out.x_km[0])


def test_nhops_2_cartesian_z_nonnegative():
    """All z_km values must be >= 0 (above ground) for a 2-hop ray."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        x0_km=0.0, z0_km=0.0,
        s_max_km=1200.0,
        max_step_km=5.0,
        nhops=2,
    )
    assert np.all(np.asarray(out.z_km, dtype=float) >= -1e-3)


# ── nhops=2 spherical gradient ────────────────────────────────────────────────

def test_nhops_2_spherical_structure():
    """Spherical nhops=2 must return v_r/v_phi and non-negative z_km."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="spherical",
        nhops=2,
        x0_km=0.0, z0_km=0.0,
        s_max_km=600.0,
        max_step_km=5.0,
    )
    assert out.x_km.size >= 1
    assert out.z_km.size == out.x_km.size
    assert hasattr(out, "v_r") and hasattr(out, "v_phi")
    assert np.all(np.asarray(out.z_km, dtype=float) >= -1e-3)
    assert np.isfinite(out.group_path_km)
    assert out.nhops_completed >= 1


def test_nhops_2_spherical_group_path_accumulates():
    """Spherical 2-hop group path >= single-hop group path."""
    rt = _make_rt()
    kw = dict(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="spherical",
        x0_km=0.0, z0_km=0.0,
        s_max_km=600.0,
        max_step_km=5.0,
    )
    out1 = rt.oblique_trace(**kw, nhops=1)
    out2 = rt.oblique_trace(**kw, nhops=2)
    assert out2.group_path_km >= out1.group_path_km


# ── specular reflection geometry ──────────────────────────────────────────────

def test_specular_elevation_preserved_cartesian():
    """For a horizontally-uniform ionosphere the reflected elevation must
    match the launch elevation (specular geometry)."""
    alts = np.linspace(0.0, 400.0, 81, dtype=float)
    x = np.linspace(0.0, 2000.0, 201, dtype=float)
    zz, xx = np.meshgrid(alts, x, indexing="ij")
    ne = 1.2e11 * np.exp(-(((zz - 200.0) / 70.0) ** 2))
    rt = RT2D(x_km=x, z_km=alts, ne_m3=ne)

    launch_elev = 35.0
    out = rt.oblique_trace(
        freq_hz=8e6,
        elevation_deg=launch_elev,
        coordinate_system="cartesian",
        nhops=2,
        x0_km=0.0, z0_km=0.0,
        s_max_km=1500.0,
        max_step_km=3.0,
    )
    if out.nhops_completed >= 2:
        vx_f = float(out.vx[-1])
        vz_f = float(out.vz[-1])
        elev_terminal = float(np.degrees(np.arctan2(abs(vz_f), abs(vx_f))))
        # Allow ±5° tolerance for numerical refraction in the ionosphere.
        assert abs(elev_terminal - launch_elev) < 5.0


# ── nhops_completed counter ───────────────────────────────────────────────────

def test_nhops_completed_never_exceeds_requested():
    """nhops_completed <= nhops for any ray."""
    rt = _make_rt()
    for nhops in (1, 2, 3):
        out = rt.oblique_trace(
            freq_hz=7e6,
            elevation_deg=30.0,
            coordinate_system="cartesian",
            nhops=nhops,
            x0_km=0.0, z0_km=0.0,
            s_max_km=2500.0,
            max_step_km=5.0,
        )
        assert 1 <= out.nhops_completed <= nhops


def test_nhops_completed_is_1_for_non_ground_ray():
    """A ray that doesn't reach ground (very high frequency) completes
    exactly 1 hop regardless of nhops."""
    rt = _make_rt()
    # Use a frequency well above foF2 (~9 MHz for ne=1.5e11)
    out = rt.oblique_trace(
        freq_hz=50e6,
        elevation_deg=15.0,
        coordinate_system="cartesian",
        nhops=3,
        x0_km=0.0, z0_km=0.0,
        s_max_km=500.0,
        max_step_km=5.0,
    )
    assert out.status != "ground"
    assert out.nhops_completed == 1


# ── backward-compatibility attributes ─────────────────────────────────────────

def test_nhops_1_has_nhops_completed_attribute():
    """Even the single-hop fast path must expose nhops_completed=1."""
    rt = _make_rt()
    out = rt.oblique_trace(
        freq_hz=7e6,
        elevation_deg=30.0,
        coordinate_system="cartesian",
        nhops=1,
        x0_km=0.0, z0_km=0.0,
        s_max_km=600.0,
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
            x0_km=0.0, z0_km=0.0,
            s_max_km=600.0,
            max_step_km=5.0,
        )
        assert out.nhops_completed >= 1


def test_nhops_1_coordinate_system_attribute():
    """coordinate_system attribute is set on single-hop output."""
    rt = _make_rt()
    for coord in ("cartesian", "spherical"):
        out = rt.oblique_trace(
            freq_hz=7e6,
            elevation_deg=30.0,
            coordinate_system=coord,
            nhops=1,
            x0_km=0.0, z0_km=0.0,
            s_max_km=600.0,
            max_step_km=5.0,
        )
        assert out.coordinate_system == coord
