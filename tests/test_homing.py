"""
Tests for hfpytrace.homing  (Homing2D, Homing3D, HomingConfig, HomingResult).

Strategy
--------
All ray-tracer calls are monkeypatched with lightweight stub functions so
the tests run without a full IRI / MSIS environment.  The stubs model a
simple Chapman-like parabolic layer and return analytical group delays,
making it straightforward to verify that the homing algorithm finds the
correct elevation angle.

Test groups
-----------
``TestHomingConfig``        – dataclass construction and field defaults.
``TestHomingResult``        – immutable record helpers (to_dict, frozen).
``TestHoming2D``            – fan sweep, root-finding, NVIS, oblique.
``TestHoming2DSynthesize``  – synthesize_ionogram column layout and filtering.
``TestHoming3D``            – per-azimuth sweep, distance metric, acceptance.
``TestHoming3DSynthesize``  – synthesize_ionogram column layout.
``TestGeometryHelpers``     – internal haversine and xy→latlon utilities.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from hfpytrace.homing import (
    Homing2D,
    Homing3D,
    HomingConfig,
    HomingResult,
    _haversine_km,
    _xy_to_latlon,
)

# ─────────────────────────────────────────────────────────────────────────── #
#  Stub helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_2d_trace_stub(noise_km: float = 0.0):
    """
    Return a callable that mimics RT2D.trace.

    The stub models ground_range = 500·cos(φ) − 500·cos(88°) km so that the
    function crosses zero at φ ≈ 88° (allowing brentq to find a root near
    vertical incidence) and is negative for φ > 88° (overshot / vertical).
    Rays with negative elevation are given status='domain'.
    """
    _gr_offset = 500.0 * math.cos(math.radians(88.0))

    def _trace(freq_hz: float, elevation_deg: float, mode: str = "O", **kw):
        el = float(elevation_deg)
        if el < 0:
            return SimpleNamespace(
                status="domain",
                ground_range_km=float("nan"),
                group_path_km=float("nan"),
                group_delay_sec=float("nan"),
                x_km=np.array([]),
                z_km=np.array([]),
            )
        gr = 500.0 * math.cos(math.radians(el)) - _gr_offset + noise_km
        gp = 2.0 * 300.0
        gd = gp / 299_792.458
        return SimpleNamespace(
            status="ground",
            ground_range_km=max(gr, 0.0),      # physical floor
            group_path_km=gp,
            group_delay_sec=gd,
            x_km=np.linspace(0, max(gr, 0.0), 10),
            z_km=np.linspace(0, 200, 10),
        )
    return _trace


def _make_3d_trace_stub(target_lat: float, target_lon: float, hit_az: float):
    """
    Return a callable that mimics RT3D.oblique_trace.

    For the azimuth nearest *hit_az* (within 10°) AND elevation in [10°,80°],
    the stub lands at (*target_lat*, *target_lon*) + a small jitter so that
    dist_to_target ≈ 0.5 km (within any reasonable tolerance but distinguishable
    from 0).  All other angles land ~500 km north of target.
    """
    def _trace(
        freq_hz: float,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        mode: str = "O",
        coordinate_system: str = "spherical",
        solver: str = "gradient",
        **kw,
    ):
        near_az = abs(((float(azimuth_deg) - hit_az) + 180) % 360 - 180) < 10.0
        if near_az and 10.0 < float(elevation_deg) < 80.0:
            # jitter ~0.005° ≈ 0.5 km so that dist > 0 but still within tolerance
            land_lat = target_lat + 0.005
            land_lon = target_lon
        else:
            land_lat = target_lat + 4.5   # ~500 km north
            land_lon = target_lon
        gp = 600.0
        gd = gp / 299_792.458
        return SimpleNamespace(
            status="ground",
            ground_range_km=float("nan"),
            group_path_km=gp,
            group_delay_sec=gd,
            lat_deg=np.array([target_lat, land_lat]),
            lon_deg=np.array([target_lon, land_lon]),
            x_km=None,
            y_km=None,
            z_km=np.array([0.0, 200.0]),
        )
    return _trace


class _FakeRT2D:
    def __init__(self):
        self.trace = _make_2d_trace_stub()


class _FakeRT3D:
    def __init__(self, target_lat: float, target_lon: float, hit_az: float):
        self._fn = _make_3d_trace_stub(target_lat, target_lon, hit_az)

    def oblique_trace(self, **kw):
        return self._fn(**kw)


# ─────────────────────────────────────────────────────────────────────────── #
#  HomingConfig                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestHomingConfig:
    def test_defaults(self):
        cfg = HomingConfig()
        assert cfg.tol_km == 10.0
        assert cfg.elev_min_deg == -30.0
        assert cfg.elev_max_deg == 89.0
        assert cfg.elev_step_deg == 2.0
        assert cfg.az_min_deg == 0.0
        assert cfg.az_max_deg == 360.0
        assert cfg.az_step_deg == 5.0
        assert cfg.fine_points == 2000
        assert cfg.max_roots == 10
        assert cfg.max_roots_per_az == 5
        assert cfg.mode == "O"

    def test_custom_values(self):
        cfg = HomingConfig(tol_km=25.0, mode="X", elev_step_deg=5.0)
        assert cfg.tol_km == 25.0
        assert cfg.mode == "X"
        assert cfg.elev_step_deg == 5.0

    def test_is_mutable(self):
        cfg = HomingConfig()
        cfg.tol_km = 50.0
        assert cfg.tol_km == 50.0


# ─────────────────────────────────────────────────────────────────────────── #
#  HomingResult                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class TestHomingResult:
    def _make(self, **overrides):
        defaults = dict(
            freq_hz=5e6,
            elevation_deg=45.0,
            group_path_km=600.0,
            group_delay_sec=0.002,
            virtual_height_km=300.0,
            status="ground",
            mode="O",
        )
        defaults.update(overrides)
        return HomingResult(**defaults)

    def test_frozen(self):
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.freq_hz = 7e6       # type: ignore[misc]

    def test_to_dict_excludes_arrays(self):
        r = self._make(x_km=np.array([0, 1, 2]), z_km=np.array([0, 100, 200]))
        d = r.to_dict()
        assert "x_km" not in d
        assert "z_km" not in d
        assert d["freq_hz"] == 5e6
        assert d["elevation_deg"] == 45.0

    def test_default_nan_fields(self):
        r = self._make()
        assert math.isnan(r.ground_range_km)
        assert math.isnan(r.azimuth_deg)
        assert math.isnan(r.landing_lat)
        assert math.isnan(r.dist_to_target_km)

    def test_virtual_height_stored(self):
        r = self._make(virtual_height_km=250.0)
        assert r.virtual_height_km == 250.0


# ─────────────────────────────────────────────────────────────────────────── #
#  Homing2D                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestHoming2D:
    """
    Stub: ground_range = 500·cos(φ) km.
    At x_target=0 the theoretical root is φ = 90° (vertical).
    At x_target=250 km the theoretical root is cos⁻¹(0.5) = 60°.
    """

    def _homing(self, tol=15.0, **cfg_kw):
        rt = _FakeRT2D()
        defaults = dict(
            tol_km=tol,
            elev_min_deg=0.0,
            elev_max_deg=89.0,
            elev_step_deg=3.0,
            fine_points=500,
        )
        defaults.update(cfg_kw)
        cfg = HomingConfig(**defaults)
        return Homing2D(rt, config=cfg, trace_fn=rt.trace)

    def test_nvis_finds_root_near_88deg(self):
        """
        NVIS (x_target=0): stub crosses zero at ~88° so brentq should
        find a root there.  We accept any root above 80°.
        """
        h = self._homing(tol=10.0)
        rays = h.home(freq_hz=5e6, x_target_km=0.0)
        assert len(rays) >= 1
        top_elev = max(r.elevation_deg for r in rays)
        assert top_elev > 80.0, f"Expected root above 80°, got {top_elev:.1f}°"

    def test_oblique_root_at_60deg(self):
        """Oblique target 250 km: stub gives root where 500·cos(φ)−offset = 250."""
        h = self._homing(tol=15.0)
        rays = h.home(freq_hz=5e6, x_target_km=250.0)
        assert len(rays) >= 1
        found_60 = any(abs(r.elevation_deg - 60.0) < 8.0 for r in rays)
        assert found_60, f"No root near 60°; got {[r.elevation_deg for r in rays]}"

    def test_returns_homingresult_instances(self):
        h = self._homing()
        rays = h.home(freq_hz=5e6)
        for r in rays:
            assert isinstance(r, HomingResult)

    def test_virtual_height_positive(self):
        h = self._homing()
        rays = h.home(freq_hz=5e6)
        for r in rays:
            assert r.virtual_height_km > 0.0

    def test_miss_within_tolerance(self):
        h = self._homing(tol=20.0)
        rays = h.home(freq_hz=5e6, x_target_km=250.0)
        for r in rays:
            assert abs(r.miss_km) <= 20.0

    def test_path_arrays_present(self):
        h = self._homing()
        rays = h.home(freq_hz=5e6)
        for r in rays:
            assert r.x_km is not None and r.x_km.size > 0
            assert r.z_km is not None and r.z_km.size > 0

    def test_per_call_tol_override(self):
        """Tight tolerance should accept fewer rays than loose."""
        h = self._homing(tol=5.0)
        tight = h.home(freq_hz=5e6, x_target_km=0.0, tol_km=2.0)
        loose = h.home(freq_hz=5e6, x_target_km=0.0, tol_km=50.0)
        assert len(loose) >= len(tight)

    def test_config_not_mutated_by_call(self):
        h = self._homing(tol=10.0)
        h.home(freq_hz=5e6, tol_km=999.0)
        assert h.config.tol_km == 10.0

    def test_empty_above_muf(self):
        """If stub returns domain for all elevations, result is empty."""
        rt = _FakeRT2D()
        # Override with a stub that always returns domain
        def _always_domain(**kw):
            return SimpleNamespace(
                status="domain",
                ground_range_km=float("nan"),
                group_path_km=float("nan"),
                group_delay_sec=float("nan"),
                x_km=np.array([]),
                z_km=np.array([]),
            )
        cfg = HomingConfig(elev_min_deg=0.0, elev_max_deg=89.0, elev_step_deg=5.0)
        h = Homing2D(rt, config=cfg, trace_fn=_always_domain)
        rays = h.home(freq_hz=20e6)
        assert rays == []


# ─────────────────────────────────────────────────────────────────────────── #
#  Homing2D – synthesize_ionogram                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestHoming2DSynthesize:
    def test_column_count(self):
        rt = _FakeRT2D()
        cfg = HomingConfig(elev_min_deg=0.0, elev_max_deg=89.0, elev_step_deg=5.0,
                           tol_km=30.0, fine_points=200)
        h = Homing2D(rt, config=cfg, trace_fn=rt.trace)
        freqs = np.array([3e6, 5e6, 7e6])
        iono = h.synthesize_ionogram(freqs, x_target_km=0.0)
        assert iono.ndim == 2
        assert iono.shape[1] == 5

    def test_empty_when_no_returns(self):
        def _no_ground(**kw):
            return SimpleNamespace(
                status="domain", ground_range_km=float("nan"),
                group_path_km=float("nan"), group_delay_sec=float("nan"),
                x_km=np.array([]), z_km=np.array([]),
            )
        rt = _FakeRT2D()
        cfg = HomingConfig(elev_min_deg=0.0, elev_max_deg=89.0, elev_step_deg=5.0)
        h = Homing2D(rt, config=cfg, trace_fn=_no_ground)
        iono = h.synthesize_ionogram([5e6, 7e6])
        assert iono.shape == (0, 5)

    def test_frequencies_in_output(self):
        rt = _FakeRT2D()
        cfg = HomingConfig(elev_min_deg=0.0, elev_max_deg=89.0,
                           elev_step_deg=5.0, tol_km=30.0, fine_points=200)
        h = Homing2D(rt, config=cfg, trace_fn=rt.trace)
        freqs = np.array([4e6, 8e6])
        iono = h.synthesize_ionogram(freqs)
        if iono.shape[0] > 0:
            assert set(iono[:, 0]).issubset({4e6, 8e6})


# ─────────────────────────────────────────────────────────────────────────── #
#  Homing3D                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class TestHoming3D:
    TARGET_LAT, TARGET_LON = 45.0, -90.0
    TX_LAT, TX_LON = 40.0, -95.0
    HIT_AZ = 35.0   # azimuth at which stub lands on target

    def _homing(self, **cfg_kw):
        rt = _FakeRT3D(self.TARGET_LAT, self.TARGET_LON, self.HIT_AZ)
        defaults = dict(
            tol_km=50.0,
            az_step_deg=10.0,
            elev_min_deg=10.0,
            elev_max_deg=80.0,
            elev_step_deg=10.0,
            fine_points=200,
        )
        defaults.update(cfg_kw)
        cfg = HomingConfig(**defaults)
        return Homing3D(
            rt,
            tx_lat=self.TX_LAT,
            tx_lon=self.TX_LON,
            config=cfg,
        )

    def test_finds_hit_azimuth(self):
        h = self._homing()
        rays = h.home(5e6, target_lat=self.TARGET_LAT, target_lon=self.TARGET_LON)
        assert len(rays) >= 1
        found = any(abs(r.azimuth_deg - self.HIT_AZ) <= 15.0 for r in rays)
        assert found, f"No ray near az={self.HIT_AZ}°; got {[r.azimuth_deg for r in rays]}"

    def test_returns_homingresult_instances(self):
        h = self._homing()
        rays = h.home(5e6, target_lat=self.TARGET_LAT, target_lon=self.TARGET_LON)
        for r in rays:
            assert isinstance(r, HomingResult)

    def test_dist_within_tolerance(self):
        h = self._homing()
        rays = h.home(5e6, target_lat=self.TARGET_LAT, target_lon=self.TARGET_LON)
        for r in rays:
            assert r.dist_to_target_km <= 50.0

    def test_sorted_by_azimuth(self):
        h = self._homing()
        rays = h.home(5e6, target_lat=self.TARGET_LAT, target_lon=self.TARGET_LON)
        azimuths = [r.azimuth_deg for r in rays]
        assert azimuths == sorted(azimuths)

    def test_virtual_height_positive(self):
        h = self._homing()
        rays = h.home(5e6, target_lat=self.TARGET_LAT, target_lon=self.TARGET_LON)
        for r in rays:
            assert r.virtual_height_km > 0.0

    def test_landing_coords_stored(self):
        h = self._homing()
        rays = h.home(5e6, target_lat=self.TARGET_LAT, target_lon=self.TARGET_LON)
        for r in rays:
            assert math.isfinite(r.landing_lat)
            assert math.isfinite(r.landing_lon)

    def test_per_call_tol_override(self):
        h = self._homing()
        # Very tight tolerance should reject all stubs that don't land exactly
        very_tight = h.home(5e6, target_lat=self.TARGET_LAT,
                            target_lon=self.TARGET_LON, tol_km=0.001)
        loose = h.home(5e6, target_lat=self.TARGET_LAT,
                       target_lon=self.TARGET_LON, tol_km=100.0)
        assert len(loose) >= len(very_tight)

    def test_empty_when_no_hits(self):
        """Move target far away so stub never lands within tolerance."""
        h = self._homing(tol_km=5.0)
        rays = h.home(5e6, target_lat=0.0, target_lon=0.0)
        assert rays == []


# ─────────────────────────────────────────────────────────────────────────── #
#  Homing3D – synthesize_ionogram                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class TestHoming3DSynthesize:
    def test_column_count(self):
        rt = _FakeRT3D(45.0, -90.0, 35.0)
        cfg = HomingConfig(tol_km=50.0, az_step_deg=10.0,
                           elev_min_deg=10.0, elev_max_deg=80.0,
                           elev_step_deg=10.0, fine_points=200)
        h = Homing3D(rt, tx_lat=40.0, tx_lon=-95.0, config=cfg)
        freqs = np.array([5e6, 7e6])
        iono = h.synthesize_ionogram(freqs, target_lat=45.0, target_lon=-90.0)
        assert iono.ndim == 2
        assert iono.shape[1] == 6

    def test_empty_output_shape(self):
        # Target far from stub's landing point (~4.5° = ~500 km offset) → no hits
        rt = _FakeRT3D(45.0, -90.0, 35.0)
        cfg = HomingConfig(tol_km=10.0, az_step_deg=10.0,
                           elev_min_deg=10.0, elev_max_deg=80.0,
                           elev_step_deg=10.0, fine_points=200)
        h = Homing3D(rt, tx_lat=40.0, tx_lon=-95.0, config=cfg)
        # Point the homing at a target far from where the stub actually lands
        iono = h.synthesize_ionogram([5e6], target_lat=0.0, target_lon=0.0)
        assert iono.shape == (0, 6)


# ─────────────────────────────────────────────────────────────────────────── #
#  Geometry helpers                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

class TestGeometryHelpers:
    def test_haversine_same_point(self):
        assert _haversine_km(45.0, -90.0, 45.0, -90.0) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        # Distance between (0°,0°) and (0°,1°) ≈ 111.195 km
        d = _haversine_km(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(111.195, rel=1e-3)

    def test_haversine_symmetry(self):
        d1 = _haversine_km(40.0, -95.0, 45.0, -90.0)
        d2 = _haversine_km(45.0, -90.0, 40.0, -95.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_haversine_north_south(self):
        # 1° of latitude ≈ 111.195 km
        d = _haversine_km(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111.195, rel=1e-3)

    def test_xy_to_latlon_north(self):
        # 111.195 km north of (0°, 0°) should be ≈ (1°, 0°)
        lat, lon = _xy_to_latlon(0.0, 111.195, 0.0, 0.0)
        assert lat == pytest.approx(1.0, abs=0.01)
        assert lon == pytest.approx(0.0, abs=0.01)

    def test_xy_to_latlon_east(self):
        # East displacement at equator: 111.195 km ≈ 1° longitude
        lat, lon = _xy_to_latlon(111.195, 0.0, 0.0, 0.0)
        assert lat == pytest.approx(0.0, abs=0.01)
        assert lon == pytest.approx(1.0, abs=0.01)

    def test_xy_to_latlon_zero_offset(self):
        lat, lon = _xy_to_latlon(0.0, 0.0, 37.5, -122.4)
        assert lat == pytest.approx(37.5)
        assert lon == pytest.approx(-122.4)
