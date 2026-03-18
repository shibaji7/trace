"""
hfpytrace.homing
================

HF ray homing-in for 2-D and 3-D ionospheric tracers.

Overview
--------
"Homing-in" is the process of finding all launch angles whose ray path
arrives at (or within a tolerance of) a prescribed target on the ground.
In vertical-incidence sounding (NVIS / ionosonde) the target ground range
is zero; for oblique links it is a specific distance or lat/lon.

The algorithm follows Laryunin (2025) [1]_:

1. **Fan sweep** – launch rays over a coarse elevation grid (and, for 3-D,
   over a coarse azimuth grid as well) and record the signed miss-distance
   ``D`` for each launch angle.
2. **Root-finding** – fit a cubic spline to ``D(φ)`` (and ``D(φ,ψ)`` per
   azimuth slice) and locate zero-crossings with Brent's method.  Each
   crossing is one propagation mode (ordinary, loop-like, off-vertical …).
3. **Re-trace** – integrate the ODE again at the refined launch angle and
   collect group delay, virtual height, and path geometry.

Classes
-------
HomingConfig
    Shared sweep / tolerance parameters, passed to either tracer class.
HomingResult
    Immutable result record for a single homed ray.
Homing2D
    Homing over a 2-D ionospheric slice (``RT2D``).
Homing3D
    Homing over a 3-D ionospheric volume (``RT3D``).

Quick start
-----------
>>> from hfpytrace.homing import Homing2D, Homing3D, HomingConfig
>>> cfg = HomingConfig(tol_km=15.0, elev_step_deg=2.0)
>>> h2 = Homing2D(rt2d, config=cfg)
>>> rays = h2.home(freq_hz=5e6)                    # NVIS (x_target=0)
>>> rays = h2.home(freq_hz=7e6, x_target_km=800)   # oblique
>>> iono = h2.synthesize_ionogram(freqs_hz)         # full ionogram
>>> h3 = Homing3D(rt3d, tx_lat=40.0, tx_lon=-95.0, config=cfg)
>>> rays = h3.home(freq_hz=5e6, target_lat=45.0, target_lon=-90.0)

References
----------
.. [1] Laryunin, O. (2025). Reconstruction of medium-scale TID
       characteristics from a series of vertical incidence ionograms with
       inner cusps. *Advances in Space Research*, 75, 6425–6430.
       https://doi.org/10.1016/j.asr.2025.01.069
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

__all__ = ["HomingConfig", "HomingResult", "Homing2D", "Homing3D"]

# ─────────────────────────────────────────────────────────────────────────── #
#  Physical / geodetic constants                                              #
# ─────────────────────────────────────────────────────────────────────────── #

C_KM_S: float = 299_792.458   # speed of light [km s⁻¹]
R_EARTH_KM: float = 6_371.0   # mean Earth radius [km]


# ─────────────────────────────────────────────────────────────────────────── #
#  Internal geometry helpers                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance [km] between two (lat°, lon°) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2.0 * R_EARTH_KM * math.asin(math.sqrt(a))


def _xy_to_latlon(
    x_km: float, y_km: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    """
    Local flat-Earth (x = East, y = North) offset [km] → (lat°, lon°).
    Accurate for offsets up to ~2 000 km.
    """
    dlat = math.degrees(y_km / R_EARTH_KM)
    dlon = math.degrees(
        x_km / (R_EARTH_KM * math.cos(math.radians(origin_lat)))
    )
    return origin_lat + dlat, origin_lon + dlon


def _landing_latlon(
    ray: SimpleNamespace,
    tx_lat: float,
    tx_lon: float,
) -> tuple[float, float] | tuple[None, None]:
    """Extract landing (lat, lon) from a 3-D ray; returns (None, None) on failure."""
    if ray.status != "ground":
        return None, None
    if hasattr(ray, "lat_deg") and ray.lat_deg is not None:
        return float(ray.lat_deg[-1]), float(ray.lon_deg[-1])
    if hasattr(ray, "x_km") and hasattr(ray, "y_km"):
        return _xy_to_latlon(float(ray.x_km[-1]), float(ray.y_km[-1]), tx_lat, tx_lon)
    return None, None


# ─────────────────────────────────────────────────────────────────────────── #
#  Public data classes                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class HomingConfig:
    """
    Sweep and convergence parameters shared by both :class:`Homing2D` and
    :class:`Homing3D`.

    Parameters
    ----------
    tol_km : float
        Acceptance radius [km].  A homed ray is accepted when its landing
        point is within *tol_km* of the target.  Default ``10.0``.
    elev_min_deg : float
        Lower bound of the elevation fan sweep [°].  Default ``-30.0``.
    elev_max_deg : float
        Upper bound of the elevation fan sweep [°].  Default ``89.0``.
    elev_step_deg : float
        Coarse elevation step for the fan sweep [°].  Default ``2.0``.
    az_min_deg : float
        *(3-D only)* Lower azimuth bound [°].  Default ``0.0``.
    az_max_deg : float
        *(3-D only)* Upper azimuth bound [°].  Default ``360.0``.
    az_step_deg : float
        *(3-D only)* Coarse azimuth step [°].  Default ``5.0``.
    fine_points : int
        Number of interpolation points used when searching for spline roots.
        Increase for densely packed multipath modes.  Default ``2000``.
    max_roots : int
        Safety cap on total returned ray paths per frequency.  Default ``10``.
    max_roots_per_az : int
        *(3-D only)* Safety cap per azimuth slice.  Default ``5``.
    mode : str
        Polarisation mode (``'O'`` or ``'X'``).  Default ``'O'``.
    """

    tol_km: float = 10.0
    elev_min_deg: float = -30.0
    elev_max_deg: float = 89.0
    elev_step_deg: float = 2.0
    az_min_deg: float = 0.0
    az_max_deg: float = 360.0
    az_step_deg: float = 5.0
    fine_points: int = 2000
    max_roots: int = 10
    max_roots_per_az: int = 5
    mode: str = "O"


@dataclass(frozen=True)
class HomingResult:
    """
    Immutable record for a single homed ray.

    Common fields (2-D and 3-D)
    ---------------------------
    freq_hz : float
        Operating frequency [Hz].
    elevation_deg : float
        Homed launch elevation [°].
    group_path_km : float
        Integrated group path length [km].
    group_delay_sec : float
        Two-way group delay [s].
    virtual_height_km : float
        Virtual height ``h' = c · τ / 2`` [km].
    status : str
        Ray termination status (``'ground'``, ``'domain'``, …).
    mode : str
        Polarisation mode.

    2-D only
    --------
    ground_range_km : float
        Actual landing ground range [km].
    miss_km : float
        Signed miss-distance from target [km].
    x_km, z_km : ndarray
        Ray path in Cartesian (range, altitude) coordinates.

    3-D only
    --------
    azimuth_deg : float
        Homed launch azimuth [°].
    landing_lat, landing_lon : float
        Geographic coordinates of the landing point [°].
    dist_to_target_km : float
        Great-circle distance from landing to target [km].
    """

    freq_hz: float
    elevation_deg: float
    group_path_km: float
    group_delay_sec: float
    virtual_height_km: float
    status: str
    mode: str
    # 2-D fields
    ground_range_km: float = float("nan")
    miss_km: float = float("nan")
    x_km: np.ndarray | None = None
    z_km: np.ndarray | None = None
    # 3-D fields
    azimuth_deg: float = float("nan")
    landing_lat: float = float("nan")
    landing_lon: float = float("nan")
    dist_to_target_km: float = float("nan")
    lat_deg: np.ndarray | None = None
    lon_deg: np.ndarray | None = None
    # extra path arrays stored as a dict to keep the frozen dataclass simple
    extra: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation (excludes ``extra``)."""
        return {
            k: v for k, v in self.__dict__.items()
            if k != "extra" and not isinstance(v, np.ndarray)
        }


# ─────────────────────────────────────────────────────────────────────────── #
#  2-D Homing                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class Homing2D:
    """
    Homing-in solver for a 2-D ionospheric tracer (:class:`~hfpytrace.model.rt2d.RT2D`).

    The solver finds all elevation angles whose ray path lands within
    :attr:`~HomingConfig.tol_km` of a target ground range ``x_target_km``
    (default 0 = NVIS / vertical incidence).

    Parameters
    ----------
    rt2d : RT2D
        Initialised 2-D tracer with a loaded ionospheric profile.
    config : HomingConfig, optional
        Sweep and tolerance settings.  A default :class:`HomingConfig` is
        used when omitted.
    trace_fn : callable, optional
        Override the trace callable.  Defaults to ``rt2d.trace``.  Must
        accept ``freq_hz``, ``elevation_deg``, and ``mode`` keyword
        arguments and return a :class:`~types.SimpleNamespace` with at
        least ``status``, ``ground_range_km``, ``group_path_km``,
        ``group_delay_sec``, ``x_km``, and ``z_km``.
    trace_kw : dict, optional
        Extra keyword arguments forwarded to *trace_fn* on every call.

    Examples
    --------
    NVIS ionogram synthesis::

        from hfpytrace.model.rt2d import RT2D, RT2DProfile
        from hfpytrace.homing import Homing2D, HomingConfig
        import numpy as np

        profile = RT2DProfile.from_cfg(cfg, time=event_time, fetch_iri=True)
        model   = RT2D(profile=profile)
        homing  = Homing2D(model, config=HomingConfig(tol_km=10.0, elev_step_deg=2.0))

        freqs = np.arange(2e6, 12e6, 0.02e6)
        iono  = homing.synthesize_ionogram(freqs)   # shape (N, 5)

    Oblique homing to 800 km::

        rays = homing.home(freq_hz=7e6, x_target_km=800.0, tol_km=20.0)
        for r in rays:
            print(f"el={r.elevation_deg:.1f}°  h'={r.virtual_height_km:.0f} km")
    """

    def __init__(
        self,
        rt2d,
        config: HomingConfig | None = None,
        trace_fn: Callable | None = None,
        trace_kw: dict | None = None,
    ) -> None:
        self._rt2d = rt2d
        self.config = config or HomingConfig()
        self._trace_fn: Callable = trace_fn or rt2d.trace
        self._trace_kw: dict = trace_kw or {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def home(
        self,
        freq_hz: float,
        *,
        x_target_km: float = 0.0,
        tol_km: float | None = None,
        elev_min_deg: float | None = None,
        elev_max_deg: float | None = None,
        elev_step_deg: float | None = None,
        mode: str | None = None,
    ) -> list[HomingResult]:
        """
        Find all elevation angles that home to *x_target_km* ground range.

        Per-call keyword arguments override the corresponding
        :class:`HomingConfig` values without mutating the stored config.

        Parameters
        ----------
        freq_hz : float
            Operating frequency [Hz].
        x_target_km : float
            Target ground range [km].  ``0`` gives NVIS / vertical incidence.
        tol_km : float, optional
            Override :attr:`HomingConfig.tol_km` for this call.
        elev_min/max/step_deg : float, optional
            Override sweep bounds / step for this call.
        mode : str, optional
            Override polarisation mode for this call.

        Returns
        -------
        list[HomingResult]
            One entry per distinct ray path that lands within *tol_km* of
            the target.  Empty list when no return exists (above MUF, etc.).
        """
        cfg = self.config
        _tol = tol_km if tol_km is not None else cfg.tol_km
        _emin = elev_min_deg if elev_min_deg is not None else cfg.elev_min_deg
        _emax = elev_max_deg if elev_max_deg is not None else cfg.elev_max_deg
        _estep = elev_step_deg if elev_step_deg is not None else cfg.elev_step_deg
        _mode = mode or cfg.mode

        return self._home_impl(
            freq_hz=freq_hz,
            x_target_km=x_target_km,
            tol_km=_tol,
            elev_min_deg=_emin,
            elev_max_deg=_emax,
            elev_step_deg=_estep,
            mode=_mode,
        )

    def synthesize_ionogram(
        self,
        freqs_hz: Sequence[float] | np.ndarray,
        *,
        x_target_km: float = 0.0,
        tol_km: float | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        """
        Build a synthetic ionogram by running :meth:`home` at each frequency.

        Parameters
        ----------
        freqs_hz : array-like
            Operating frequencies [Hz].
        x_target_km : float
            Target ground range [km].
        tol_km : float, optional
            Override tolerance for this call.
        mode : str, optional
            Override polarisation mode.

        Returns
        -------
        np.ndarray, shape (N_pixels, 5)
            Columns: ``[freq_hz, virtual_height_km, elevation_deg,
            ground_range_km, miss_km]``.
            Each row is one ionogram pixel (one homed ray at one frequency).
        """
        rows: list[tuple] = []
        for f in np.asarray(freqs_hz, dtype=float):
            for r in self.home(f, x_target_km=x_target_km, tol_km=tol_km, mode=mode):
                rows.append((f, r.virtual_height_km, r.elevation_deg,
                             r.ground_range_km, r.miss_km))
        return np.array(rows, dtype=float) if rows else np.empty((0, 5))

    # ------------------------------------------------------------------ #
    #  Internal implementation                                            #
    # ------------------------------------------------------------------ #

    def _do_trace(self, freq_hz: float, elevation_deg: float, mode: str) -> SimpleNamespace:
        return self._trace_fn(
            freq_hz=freq_hz,
            elevation_deg=float(elevation_deg),
            mode=mode,
            **self._trace_kw,
        )

    def _miss(self, ray: SimpleNamespace, x_target_km: float) -> float:
        """Signed miss-distance [km]: + overshot, − undershot, NaN if no ground return."""
        if ray.status != "ground" or not np.isfinite(ray.ground_range_km):
            return float("nan")
        return float(ray.ground_range_km) - x_target_km

    def _home_impl(
        self,
        freq_hz: float,
        x_target_km: float,
        tol_km: float,
        elev_min_deg: float,
        elev_max_deg: float,
        elev_step_deg: float,
        mode: str,
    ) -> list[HomingResult]:
        cfg = self.config
        elevations = np.arange(elev_min_deg, elev_max_deg, elev_step_deg, dtype=float)
        D = np.full(len(elevations), float("nan"))

        logger.debug(
            "Homing2D fan: {:.3f} MHz, target={} km, tol=±{} km, {} elevations",
            freq_hz / 1e6, x_target_km, tol_km, len(elevations),
        )

        # ── Step 1: coarse fan sweep ──────────────────────────────────────
        for i, el in enumerate(elevations):
            ray = self._do_trace(freq_hz, el, mode)
            D[i] = self._miss(ray, x_target_km)

        valid = np.isfinite(D)
        if valid.sum() < 4:
            logger.warning(
                "Homing2D: < 4 valid ground returns at {:.3f} MHz", freq_hz / 1e6
            )
            return []

        el_v, D_v = elevations[valid], D[valid]

        # ── Step 2: spline + root finding ────────────────────────────────
        cs = CubicSpline(el_v, D_v)
        el_fine = np.linspace(el_v[0], el_v[-1], cfg.fine_points)
        D_fine = cs(el_fine)

        root_elevs: list[float] = []
        for idx in np.where(np.diff(np.sign(D_fine)))[0][: cfg.max_roots * 2]:
            try:
                root_elevs.append(
                    brentq(cs, el_fine[idx], el_fine[idx + 1], xtol=0.05)
                )
            except ValueError:
                pass

        # Also accept coarse hits within tol that produced no clean crossing
        for el, d in zip(el_v, D_v):
            if abs(d) <= tol_km and not any(abs(el - r) < elev_step_deg for r in root_elevs):
                root_elevs.append(float(el))

        # ── Step 3: precise re-trace ──────────────────────────────────────
        results: list[HomingResult] = []
        for el_root in root_elevs[: cfg.max_roots]:
            ray = self._do_trace(freq_hz, el_root, mode)
            miss = self._miss(ray, x_target_km)
            if math.isfinite(miss) and abs(miss) > tol_km:
                continue

            vh_km = C_KM_S * ray.group_delay_sec / 2.0
            result = HomingResult(
                freq_hz=freq_hz,
                elevation_deg=float(el_root),
                group_path_km=float(ray.group_path_km),
                group_delay_sec=float(ray.group_delay_sec),
                virtual_height_km=vh_km,
                status=ray.status,
                mode=mode,
                ground_range_km=float(ray.ground_range_km),
                miss_km=float(miss) if math.isfinite(miss) else float("nan"),
                x_km=np.asarray(ray.x_km),
                z_km=np.asarray(ray.z_km),
            )
            results.append(result)
            logger.info(
                "Homing2D: el={:.2f}°  range={:.1f} km (miss={:.1f} km)  h'={:.1f} km",
                el_root, result.ground_range_km, result.miss_km, vh_km,
            )

        return results


# ─────────────────────────────────────────────────────────────────────────── #
#  3-D Homing                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class Homing3D:
    """
    Homing-in solver for a 3-D ionospheric tracer (:class:`~hfpytrace.model.rt3d.RT3D`).

    For each azimuth in the fan, the solver sweeps elevation angles and
    finds where the great-circle distance from the landing point to the
    target falls within :attr:`~HomingConfig.tol_km`.  This naturally
    handles multipath, loop-like paths, and off-great-circle propagation.

    Parameters
    ----------
    rt3d : RT3D
        Initialised 3-D tracer with a loaded ionospheric profile.
    tx_lat : float
        Transmitter latitude [°].  Required for Cartesian → lat/lon
        conversion when the tracer runs in Cartesian mode.
    tx_lon : float
        Transmitter longitude [°].
    config : HomingConfig, optional
        Sweep and tolerance settings.
    coordinate_system : str
        Passed to :meth:`RT3D.oblique_trace`.  ``'spherical'`` (default)
        gives ``lat_deg``/``lon_deg`` outputs directly; ``'cartesian'``
        uses flat-Earth conversion via *tx_lat*/*tx_lon*.
    solver : str
        Passed to :meth:`RT3D.oblique_trace`.  ``'gradient'`` (default)
        or ``'hamiltonian'``.
    trace_kw : dict, optional
        Extra keyword arguments forwarded to :meth:`RT3D.oblique_trace`
        on every call.

    Examples
    --------
    Oblique link homing from (40°N, 95°W) to (45°N, 90°W) within 25 km::

        from hfpytrace.model.rt3d import RT3D, RT3DProfile
        from hfpytrace.homing import Homing3D, HomingConfig

        profile = RT3DProfile.from_cfg(cfg, time=event_time, fetch_iri=True)
        model   = RT3D(profile=profile)
        homing  = Homing3D(model, tx_lat=40.0, tx_lon=-95.0,
                           config=HomingConfig(tol_km=25.0, az_step_deg=5.0))

        rays = homing.home(freq_hz=5e6, target_lat=45.0, target_lon=-90.0)
        for r in rays:
            print(f"az={r.azimuth_deg:.0f}° el={r.elevation_deg:.1f}°  "
                  f"h'={r.virtual_height_km:.0f} km  dist={r.dist_to_target_km:.1f} km")

    3-D ionogram synthesis::

        import numpy as np
        freqs = np.arange(2e6, 10e6, 0.05e6)
        iono  = homing.synthesize_ionogram(freqs, target_lat=45.0, target_lon=-90.0)
    """

    def __init__(
        self,
        rt3d,
        *,
        tx_lat: float,
        tx_lon: float,
        config: HomingConfig | None = None,
        coordinate_system: str = "spherical",
        solver: str = "gradient",
        trace_kw: dict | None = None,
    ) -> None:
        self._rt3d = rt3d
        self.tx_lat = float(tx_lat)
        self.tx_lon = float(tx_lon)
        self.config = config or HomingConfig()
        self._coord = coordinate_system
        self._solver = solver
        self._trace_kw: dict = trace_kw or {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def home(
        self,
        freq_hz: float,
        *,
        target_lat: float,
        target_lon: float,
        tol_km: float | None = None,
        az_min_deg: float | None = None,
        az_max_deg: float | None = None,
        az_step_deg: float | None = None,
        elev_min_deg: float | None = None,
        elev_max_deg: float | None = None,
        elev_step_deg: float | None = None,
        mode: str | None = None,
    ) -> list[HomingResult]:
        """
        Find all (azimuth, elevation) pairs that home to (*target_lat*, *target_lon*).

        Parameters
        ----------
        freq_hz : float
            Operating frequency [Hz].
        target_lat : float
            Target landing latitude [°].
        target_lon : float
            Target landing longitude [°].
        tol_km : float, optional
            Acceptance radius [km]; overrides :attr:`HomingConfig.tol_km`.
        az_min/max/step_deg : float, optional
            Override azimuth fan bounds / step.
        elev_min/max/step_deg : float, optional
            Override elevation sweep bounds / step.
        mode : str, optional
            Override polarisation mode.

        Returns
        -------
        list[HomingResult]
            One entry per homed ray path, sorted by ascending azimuth.
        """
        cfg = self.config
        _tol = tol_km if tol_km is not None else cfg.tol_km
        _azmin = az_min_deg if az_min_deg is not None else cfg.az_min_deg
        _azmax = az_max_deg if az_max_deg is not None else cfg.az_max_deg
        _azstep = az_step_deg if az_step_deg is not None else cfg.az_step_deg
        _emin = elev_min_deg if elev_min_deg is not None else cfg.elev_min_deg
        _emax = elev_max_deg if elev_max_deg is not None else cfg.elev_max_deg
        _estep = elev_step_deg if elev_step_deg is not None else cfg.elev_step_deg
        _mode = mode or cfg.mode

        return self._home_impl(
            freq_hz=freq_hz,
            target_lat=target_lat,
            target_lon=target_lon,
            tol_km=_tol,
            az_min_deg=_azmin,
            az_max_deg=_azmax,
            az_step_deg=_azstep,
            elev_min_deg=_emin,
            elev_max_deg=_emax,
            elev_step_deg=_estep,
            mode=_mode,
        )

    def synthesize_ionogram(
        self,
        freqs_hz: Sequence[float] | np.ndarray,
        *,
        target_lat: float,
        target_lon: float,
        tol_km: float | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        """
        Build a synthetic 3-D ionogram by running :meth:`home` at each frequency.

        Parameters
        ----------
        freqs_hz : array-like
            Operating frequencies [Hz].
        target_lat, target_lon : float
            Target landing point [°].
        tol_km : float, optional
            Override acceptance radius.
        mode : str, optional
            Override polarisation mode.

        Returns
        -------
        np.ndarray, shape (N_pixels, 6)
            Columns: ``[freq_hz, virtual_height_km, azimuth_deg,
            elevation_deg, landing_lat, landing_lon]``.
        """
        rows: list[tuple] = []
        for f in np.asarray(freqs_hz, dtype=float):
            for r in self.home(
                f,
                target_lat=target_lat,
                target_lon=target_lon,
                tol_km=tol_km,
                mode=mode,
            ):
                rows.append((f, r.virtual_height_km, r.azimuth_deg,
                             r.elevation_deg, r.landing_lat, r.landing_lon))
        return np.array(rows, dtype=float) if rows else np.empty((0, 6))

    # ------------------------------------------------------------------ #
    #  Internal implementation                                            #
    # ------------------------------------------------------------------ #

    def _do_trace(
        self, freq_hz: float, elevation_deg: float, azimuth_deg: float, mode: str
    ) -> SimpleNamespace:
        return self._rt3d.oblique_trace(
            freq_hz=freq_hz,
            elevation_deg=float(elevation_deg),
            azimuth_deg=float(azimuth_deg),
            mode=mode,
            coordinate_system=self._coord,
            solver=self._solver,
            **self._trace_kw,
        )

    def _dist_to_target(
        self, ray: SimpleNamespace, target_lat: float, target_lon: float
    ) -> float:
        """Great-circle distance [km] from landing to target; NaN on failure."""
        land_lat, land_lon = _landing_latlon(ray, self.tx_lat, self.tx_lon)
        if land_lat is None:
            return float("nan")
        return _haversine_km(land_lat, land_lon, target_lat, target_lon)

    def _home_impl(
        self,
        freq_hz: float,
        target_lat: float,
        target_lon: float,
        tol_km: float,
        az_min_deg: float,
        az_max_deg: float,
        az_step_deg: float,
        elev_min_deg: float,
        elev_max_deg: float,
        elev_step_deg: float,
        mode: str,
    ) -> list[HomingResult]:
        cfg = self.config
        azimuths = np.arange(az_min_deg, az_max_deg, az_step_deg, dtype=float)
        elevations = np.arange(elev_min_deg, elev_max_deg, elev_step_deg, dtype=float)

        logger.debug(
            "Homing3D fan: {:.3f} MHz, target=({:.2f}°,{:.2f}°), "
            "tol={} km, {} az × {} el",
            freq_hz / 1e6, target_lat, target_lon, tol_km,
            len(azimuths), len(elevations),
        )

        # ── Steps 1 & 2: per-azimuth elevation sweep + root-find ──────────
        candidates: list[tuple[float, float]] = []   # (azimuth_deg, elev_deg)

        for az in azimuths:
            D = np.full(len(elevations), float("nan"))
            for j, el in enumerate(elevations):
                ray = self._do_trace(freq_hz, el, az, mode)
                D[j] = self._dist_to_target(ray, target_lat, target_lon)

            valid = np.isfinite(D)
            if valid.sum() < 4 or D[valid].min() > tol_km * 3:
                continue

            el_v, D_v = elevations[valid], D[valid]
            cs = CubicSpline(el_v, D_v)
            el_fine = np.linspace(el_v[0], el_v[-1], cfg.fine_points)
            D_fine = cs(el_fine)

            # Find elevations that cross the tol_km circle boundary
            root_elevs_az: list[float] = []
            for idx in np.where(np.diff(np.sign(D_fine - tol_km)))[0][
                : cfg.max_roots_per_az * 2
            ]:
                try:
                    root_elevs_az.append(
                        brentq(
                            lambda e: cs(e) - tol_km,
                            el_fine[idx], el_fine[idx + 1],
                            xtol=0.05,
                        )
                    )
                except ValueError:
                    pass

            # Always include the best-fit elevation (minimum distance)
            best_el = float(el_fine[np.argmin(D_fine)])
            if D_fine.min() <= tol_km:
                root_elevs_az.append(best_el)

            # Deduplicate within one elevation step
            seen: list[float] = []
            for el_r in sorted(set(root_elevs_az)):
                if not any(abs(el_r - s) < elev_step_deg for s in seen):
                    seen.append(el_r)

            candidates.extend((az, el_r) for el_r in seen[: cfg.max_roots_per_az])

        logger.debug(
            "Homing3D: {} candidate (az, el) pairs before re-trace", len(candidates)
        )

        # ── Step 3: precise re-trace and acceptance filter ─────────────────
        results: list[HomingResult] = []
        for az, el in candidates:
            ray = self._do_trace(freq_hz, el, az, mode)
            dist = self._dist_to_target(ray, target_lat, target_lon)
            if not math.isfinite(dist) or dist > tol_km:
                continue

            land_lat, land_lon = _landing_latlon(ray, self.tx_lat, self.tx_lon)
            vh_km = C_KM_S * ray.group_delay_sec / 2.0

            extra: dict = {}
            for attr in ("lat_deg", "lon_deg", "x_km", "y_km", "z_km", "r_km"):
                if hasattr(ray, attr) and getattr(ray, attr) is not None:
                    extra[attr] = np.asarray(getattr(ray, attr))

            result = HomingResult(
                freq_hz=freq_hz,
                elevation_deg=float(el),
                group_path_km=float(ray.group_path_km),
                group_delay_sec=float(ray.group_delay_sec),
                virtual_height_km=vh_km,
                status=ray.status,
                mode=mode,
                azimuth_deg=float(az),
                landing_lat=float(land_lat) if land_lat is not None else float("nan"),
                landing_lon=float(land_lon) if land_lon is not None else float("nan"),
                dist_to_target_km=float(dist),
                lat_deg=extra.get("lat_deg"),
                lon_deg=extra.get("lon_deg"),
                extra=extra,
            )
            results.append(result)
            logger.info(
                "Homing3D: az={:.1f}° el={:.2f}°  "
                "land=({:.3f}°,{:.3f}°)  dist={:.1f} km  h'={:.1f} km",
                az, el, result.landing_lat, result.landing_lon,
                dist, vh_km,
            )

        return sorted(results, key=lambda r: r.azimuth_deg)
