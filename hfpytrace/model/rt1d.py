"""Lean 1D profile model for density/collision/geomag inputs."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from loguru import logger
from scipy import constants

from hfpytrace.collision import NRLMSISE2D
from hfpytrace.density.iri import IRI2d
from hfpytrace.geomag import build_geomag_grid
from hfpytrace.model.dispersion import AppletonHartreeDispersion, SenWyllerDispersion


@dataclass
class RT1DProfile:
    """
    Single-point altitude profile for ray/ionosphere workflows.

    Notes
    -----
    - `alt_km` is required and must be strictly increasing.
    - Electron density is stored in both m^-3 and cm^-3 (when available).
    - MSIS and geomagnetic outputs are attached as namespaces.
    """

    alt_km: np.ndarray
    lat: float
    lon: float
    time: dt.datetime
    ne_m3: np.ndarray | None = None
    ne_cm3: np.ndarray | None = None
    source: str = "iri"
    msise: SimpleNamespace | None = None
    geomag: SimpleNamespace | None = None

    def __post_init__(self) -> None:
        self.alt_km = np.asarray(self.alt_km, dtype=float).ravel()
        self.lat = float(self.lat)
        self.lon = float(self.lon)
        if not isinstance(self.time, dt.datetime):
            self.time = dt.datetime.fromisoformat(str(self.time))
        self.validate()

    def validate(self) -> None:
        if self.alt_km.ndim != 1 or self.alt_km.size < 2:
            raise ValueError("alt_km must be a 1D array with at least 2 points")
        if not np.all(np.diff(self.alt_km) > 0):
            raise ValueError("alt_km must be strictly increasing")
        if not np.isfinite(self.lat) or not np.isfinite(self.lon):
            raise ValueError("lat/lon must be finite")

        if self.ne_m3 is not None:
            self.ne_m3 = np.asarray(self.ne_m3, dtype=float).ravel()
            if self.ne_m3.shape != self.alt_km.shape:
                raise ValueError("ne_m3 must match alt_km shape")
            if np.any(self.ne_m3 < 0):
                raise ValueError("ne_m3 must be non-negative")
            self.ne_cm3 = self.ne_m3 * 1e-6
        elif self.ne_cm3 is not None:
            self.ne_cm3 = np.asarray(self.ne_cm3, dtype=float).ravel()
            if self.ne_cm3.shape != self.alt_km.shape:
                raise ValueError("ne_cm3 must match alt_km shape")
            if np.any(self.ne_cm3 < 0):
                raise ValueError("ne_cm3 must be non-negative")
            self.ne_m3 = self.ne_cm3 * 1e6

        if self.msise is not None:
            for k in ("N2", "O2", "O", "H", "He", "Tn"):
                if not hasattr(self.msise, k):
                    raise ValueError(f"msise missing required field: {k}")
                arr = np.asarray(getattr(self.msise, k), dtype=float).ravel()
                if arr.shape != self.alt_km.shape:
                    raise ValueError(f"msise.{k} must match alt_km shape")
                setattr(self.msise, k, arr)

        if self.geomag is not None:
            for k in ("Bx", "By", "Bz", "bmag_t", "inc_deg", "psi_deg"):
                if not hasattr(self.geomag, k):
                    raise ValueError(f"geomag missing required field: {k}")
                arr = np.asarray(getattr(self.geomag, k), dtype=float).ravel()
                if arr.shape != self.alt_km.shape:
                    raise ValueError(f"geomag.{k} must match alt_km shape")
                setattr(self.geomag, k, arr)

    @classmethod
    def from_cfg(
        cls,
        cfg,
        time: dt.datetime | None = None,
        lat: float | None = None,
        lon: float | None = None,
        alt_km: np.ndarray | None = None,
        fetch_iri: bool = True,
        fetch_msise: bool = True,
        fetch_geomag: bool = True,
        workers: int = 1,
    ) -> "RT1DProfile":
        logger.info("RT1DProfile.from_cfg: build profile from config")
        t = time if time is not None else dt.datetime.fromisoformat(str(cfg.event))
        if lat is None or lon is None:
            if hasattr(cfg, "origin"):
                lat = float(cfg.origin.lat)
                lon = float(cfg.origin.lon)
            elif hasattr(cfg, "route") and hasattr(cfg.route, "start"):
                lat = float(cfg.route.start.lat)
                lon = float(cfg.route.start.lon)
            else:
                raise ValueError("Unable to infer lat/lon from cfg. Provide lat/lon.")

        if alt_km is None:
            h0 = float(getattr(cfg, "start_height_km"))
            h1 = float(getattr(cfg, "end_height_km"))
            dh = float(getattr(cfg, "height_incriment_km", 1.0))
            alt_km = np.arange(h0, h1, dh, dtype=float)

        prof = cls(alt_km=alt_km, lat=float(lat), lon=float(lon), time=t)
        logger.info(
            "RT1DProfile created: lat={:.3f}, lon={:.3f}, alt_points={}",
            prof.lat,
            prof.lon,
            prof.alt_km.size,
        )
        if fetch_iri:
            prof.fetch_iri(cfg)
        if fetch_msise:
            prof.fetch_msise(workers=workers)
        if fetch_geomag:
            gm_cfg = getattr(cfg, "geomag_grid", SimpleNamespace(coord_input="GEO"))
            prof.fetch_geomag(
                coord_input=str(getattr(gm_cfg, "coord_input", "GEO")),
                coeff_dir=getattr(gm_cfg, "coeff_dir", None),
            )
        prof.validate()
        return prof

    def set_electron_density(
        self,
        ne_m3: np.ndarray | None = None,
        ne_cm3: np.ndarray | None = None,
        source: str = "iri",
    ) -> None:
        if (ne_m3 is None) == (ne_cm3 is None):
            raise ValueError("Provide exactly one of ne_m3 or ne_cm3")
        self.source = str(source)
        if ne_m3 is not None:
            self.ne_m3 = np.asarray(ne_m3, dtype=float).ravel()
            self.ne_cm3 = self.ne_m3 * 1e-6
        else:
            self.ne_cm3 = np.asarray(ne_cm3, dtype=float).ravel()
            self.ne_m3 = self.ne_cm3 * 1e6
        self.validate()

    def fetch_iri(self, cfg) -> np.ndarray:
        logger.info(
            "Fetching IRI profile: lat={:.3f}, lon={:.3f}, alt_points={}",
            self.lat,
            self.lon,
            self.alt_km.size,
        )
        model = IRI2d(cfg, self.time)
        ne_cm3, _ = model.fetch_dataset(
            self.time,
            lats=np.array([self.lat], dtype=float),
            lons=np.array([self.lon], dtype=float),
            alts=self.alt_km,
            workers=1,
        )
        self.ne_cm3 = np.asarray(ne_cm3[:, 0], dtype=float)
        self.ne_m3 = self.ne_cm3 * 1e6
        self.source = "iri"
        self.validate()
        logger.info(
            "IRI profile fetched: ne range [{:.3e}, {:.3e}] m^-3",
            float(np.nanmin(self.ne_m3)),
            float(np.nanmax(self.ne_m3)),
        )
        return self.ne_m3

    def fetch_msise(
        self,
        workers: int = 1,
        update_spaceweather: bool = False,
        suppress_spaceweather_warning: bool = True,
    ) -> SimpleNamespace:
        logger.info(
            "Fetching NRLMSISE profile: lat={:.3f}, lon={:.3f}, alt_points={}, workers={}",
            self.lat,
            self.lon,
            self.alt_km.size,
            int(workers),
        )
        ms = NRLMSISE2D(
            date=self.time,
            lats=np.array([self.lat], dtype=float),
            lons=np.array([self.lon], dtype=float),
            heights_km=self.alt_km,
            workers=int(workers),
            update_spaceweather=bool(update_spaceweather),
            suppress_spaceweather_warning=bool(suppress_spaceweather_warning),
        ).msise
        self.msise = SimpleNamespace(
            N2=np.asarray(ms["N2"][:, 0], dtype=float),
            O2=np.asarray(ms["O2"][:, 0], dtype=float),
            O=np.asarray(ms["O"][:, 0], dtype=float),
            H=np.asarray(ms["H"][:, 0], dtype=float),
            He=np.asarray(ms["He"][:, 0], dtype=float),
            Tn=np.asarray(ms["Tn"][:, 0], dtype=float),
            t_nn=np.asarray(ms["t_nn"][:, 0], dtype=float),
        )
        self.validate()
        logger.info("NRLMSISE profile fetched successfully.")
        return self.msise

    def fetch_geomag(
        self,
        coord_input: str = "GEO",
        coeff_dir: str | None = None,
    ) -> SimpleNamespace:
        logger.info(
            "Fetching geomag profile: lat={:.3f}, lon={:.3f}, alt_points={}, coord_input={}",
            self.lat,
            self.lon,
            self.alt_km.size,
            coord_input,
        )
        cdir = None
        if isinstance(coeff_dir, str) and coeff_dir.strip():
            cdir = coeff_dir
        gm = build_geomag_grid(
            lats=np.array([self.lat], dtype=float),
            lons=np.array([self.lon], dtype=float),
            alts_km=self.alt_km,
            time=self.time,
            coord_input=coord_input,
            coeff_dir=cdir,
        )
        self.geomag = SimpleNamespace(
            Bx=np.asarray(gm.Bx[0, 0, :], dtype=float),
            By=np.asarray(gm.By[0, 0, :], dtype=float),
            Bz=np.asarray(gm.Bz[0, 0, :], dtype=float),
            bmag_t=np.asarray(gm.bmag_t[0, 0, :], dtype=float),
            inc_deg=np.asarray(gm.inc_deg[0, 0, :], dtype=float),
            dec_deg=np.asarray(gm.dec_deg[0, 0, :], dtype=float),
            psi_deg=np.asarray(gm.psi_deg[0, 0, :], dtype=float),
        )
        self.validate()
        logger.info("Geomag profile fetched successfully.")
        return self.geomag

    @staticmethod
    def den_to_plasma_freq_hz(ne_m3: np.ndarray | float) -> np.ndarray:
        ne = np.asarray(ne_m3, dtype=float)
        if np.any(ne < 0):
            raise ValueError("Electron density must be non-negative")
        omega_p = np.sqrt(ne * constants.e**2 / (constants.epsilon_0 * constants.m_e))
        return omega_p / (2.0 * np.pi)

    @staticmethod
    def plasma_freq_to_den(freq_hz: np.ndarray | float) -> np.ndarray:
        f = np.asarray(freq_hz, dtype=float)
        if np.any(f < 0):
            raise ValueError("Plasma frequency must be non-negative")
        return (
            ((2.0 * np.pi * f) ** 2)
            * constants.epsilon_0
            * constants.m_e
            / (constants.e**2)
        )

    @staticmethod
    def inclintation2vertical(
        inclination_deg: np.ndarray | float,
    ) -> np.ndarray:
        return 90.0 - np.abs(np.asarray(inclination_deg, dtype=float))

    @staticmethod
    def inclination_to_vertical_angle(
        inclination_deg: np.ndarray | float,
    ) -> np.ndarray:
        return RT1DProfile.inclintation2vertical(inclination_deg)

    @staticmethod
    def vertical_to_magnetic_angle(inclination_deg: np.ndarray | float) -> np.ndarray:
        return RT1DProfile.inclintation2vertical(inclination_deg)


class RT1D:
    """
    1D model entry-point that owns a single :class:`RT1DProfile`.

    This initializer is intentionally flexible so callers can construct the
    profile from:

    1. a pre-built ``RT1DProfile`` object, or
    2. a TRACE cfg object (``config1D``-style namespace), or
    3. explicit scalar/array user inputs.

    Parameters
    ----------
    profile : RT1DProfile, optional
        Existing profile instance. When provided, this takes precedence and is
        validated directly.
    cfg : object, optional
        Config namespace used by :meth:`RT1DProfile.from_cfg`. Can also be used
        only for defaults (event/lat/lon/heights) when explicit values are
        partially provided.
    time : datetime | str, optional
        Profile timestamp. If omitted and ``cfg`` has ``event``, cfg event is
        used. Otherwise current UTC time is used.
    lat, lon : float, optional
        Profile location. If omitted, inferred from ``cfg.origin`` or
        ``cfg.route.start``.
    alt_km : array-like, optional
        Altitude grid [km]. If omitted and ``cfg`` is provided, built from
        ``start_height_km/end_height_km/height_incriment_km``.
    ne_m3, ne_cm3 : array-like, optional
        User-provided electron density. Provide exactly one if overriding model
        fetch.
    source : str, optional
        Density source label when user density is supplied.
    fetch_iri, fetch_msise, fetch_geomag : bool, optional
        If True, populate those profile components during initialization.
    workers : int, optional
        Worker hint passed to MSIS/cfg-based constructor where supported.
    coord_input, coeff_dir : str, optional
        Geomagnetic options passed to ``fetch_geomag`` when requested and not
        provided by cfg.

    Notes
    -----
    - This class currently provides initialization/validation orchestration.
    - Frequency conversions remain exposed as static compatibility methods.
    """

    def __init__(
        self,
        profile: RT1DProfile | None = None,
        cfg=None,
        time: dt.datetime | str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        alt_km: np.ndarray | None = None,
        ne_m3: np.ndarray | None = None,
        ne_cm3: np.ndarray | None = None,
        source: str = "iri",
        fetch_iri: bool = False,
        fetch_msise: bool = False,
        fetch_geomag: bool = False,
        workers: int = 1,
        coord_input: str = "GEO",
        coeff_dir: str | None = None,
    ):
        logger.info("Initializing RT1D model")
        self.cfg = cfg
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

        # Path 1: caller gives a full profile instance.
        if profile is not None:
            if not isinstance(profile, RT1DProfile):
                raise TypeError("profile must be an RT1DProfile instance")
            if any(v is not None for v in (cfg, lat, lon, alt_km, ne_m3, ne_cm3)):
                raise ValueError(
                    "When `profile` is provided, do not also pass cfg/lat/lon/alt/density inputs."
                )
            self.profile = profile
            self.profile.validate()
            logger.info("RT1D initialized from existing RT1DProfile")
            return

        # Path 2: build directly from cfg when available and no explicit overrides.
        has_explicit_density = (ne_m3 is not None) or (ne_cm3 is not None)
        if (
            cfg is not None
            and lat is None
            and lon is None
            and alt_km is None
            and not has_explicit_density
        ):
            t_cfg = time if time is not None else None
            self.profile = RT1DProfile.from_cfg(
                cfg=cfg,
                time=t_cfg,
                fetch_iri=bool(fetch_iri),
                fetch_msise=bool(fetch_msise),
                fetch_geomag=bool(fetch_geomag),
                workers=self.workers,
            )
            logger.info("RT1D initialized from config-driven profile")
            return

        # Path 3: explicit user assembly, with cfg optionally supplying defaults.
        t = self._resolve_time(time=time, cfg=cfg)
        lat_f, lon_f = self._resolve_location(lat=lat, lon=lon, cfg=cfg)
        alt_arr = self._resolve_altitudes(alt_km=alt_km, cfg=cfg)

        self.profile = RT1DProfile(
            alt_km=alt_arr,
            lat=lat_f,
            lon=lon_f,
            time=t,
        )

        if has_explicit_density:
            self.profile.set_electron_density(
                ne_m3=ne_m3,
                ne_cm3=ne_cm3,
                source=source,
            )
        elif fetch_iri:
            if cfg is None:
                raise ValueError("fetch_iri=True requires cfg for IRI parameters")
            self.profile.fetch_iri(cfg)

        if fetch_msise:
            self.profile.fetch_msise(workers=self.workers)

        if fetch_geomag:
            cdir = coeff_dir
            if cfg is not None and hasattr(cfg, "geomag_grid"):
                coord_input = str(getattr(cfg.geomag_grid, "coord_input", coord_input))
                cdir = getattr(cfg.geomag_grid, "coeff_dir", cdir)
            self.profile.fetch_geomag(coord_input=coord_input, coeff_dir=cdir)

        self.profile.validate()
        logger.info(
            "RT1D ready: lat={:.3f}, lon={:.3f}, alt_points={}, source={}",
            self.profile.lat,
            self.profile.lon,
            self.profile.alt_km.size,
            self.profile.source,
        )

    @staticmethod
    def _resolve_time(time: dt.datetime | str | None, cfg=None) -> dt.datetime:
        """Resolve profile time from explicit input, cfg event, or UTC now."""
        if time is not None:
            return (
                time
                if isinstance(time, dt.datetime)
                else dt.datetime.fromisoformat(str(time))
            )
        if cfg is not None and hasattr(cfg, "event"):
            return dt.datetime.fromisoformat(str(cfg.event))
        return dt.datetime.utcnow()

    @staticmethod
    def _resolve_location(
        lat: float | None, lon: float | None, cfg=None
    ) -> tuple[float, float]:
        """Resolve lat/lon from explicit values or cfg defaults."""
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        if (lat is None) ^ (lon is None):
            raise ValueError("Provide both lat and lon, or neither")
        if cfg is not None:
            if hasattr(cfg, "origin"):
                return float(cfg.origin.lat), float(cfg.origin.lon)
            if hasattr(cfg, "route") and hasattr(cfg.route, "start"):
                return float(cfg.route.start.lat), float(cfg.route.start.lon)
        raise ValueError(
            "Unable to resolve lat/lon. Provide lat/lon or cfg with origin/route.start."
        )

    @staticmethod
    def _resolve_altitudes(alt_km: np.ndarray | None, cfg=None) -> np.ndarray:
        """Resolve altitude grid from explicit input or cfg height fields."""
        if alt_km is not None:
            arr = np.asarray(alt_km, dtype=float).ravel()
            if arr.size < 2:
                raise ValueError("alt_km must contain at least 2 points")
            return arr
        if cfg is not None:
            h0 = float(getattr(cfg, "start_height_km"))
            h1 = float(getattr(cfg, "end_height_km"))
            dh = float(getattr(cfg, "height_incriment_km", 1.0))
            return np.arange(h0, h1, dh, dtype=float)
        raise ValueError("Unable to resolve altitude grid. Provide alt_km or cfg.")

    def _refractive_index_profile(
        self,
        frequency_hz: float,
        mode: str = "O",
        formulation: str = "appleton",
        collision_hz: np.ndarray | float | None = None,
        b_t: np.ndarray | float | None = None,
        theta_deg: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """
        Build a 1D refractive-index profile n(z) on ``self.profile.alt_km``.

        Parameters
        ----------
        frequency_hz : float
            Wave frequency in Hz.
        mode : str, optional
            Mode selector for the selected formulation.
        formulation : {"appleton", "senwyller"}, optional
            Dispersion relation backend.
        collision_hz, b_t, theta_deg : array-like | scalar, optional
            Optional overrides. If omitted, best-available values are pulled
            from ``self.profile``:
            - collision: 0 (collisionless)
            - b_t: geomag ``bmag_t`` if available else 0
            - theta_deg: geomag ``psi_deg`` if available else 0
        """
        if self.profile.ne_m3 is None:
            raise ValueError("Profile must include electron density (ne_m3).")

        alt = self.profile.alt_km
        ne = self.profile.ne_m3
        nu = (
            np.zeros_like(alt, dtype=float)
            if collision_hz is None
            else np.asarray(collision_hz, dtype=float)
        )
        if b_t is None:
            b = (
                np.asarray(self.profile.geomag.bmag_t, dtype=float)
                if self.profile.geomag is not None
                and hasattr(self.profile.geomag, "bmag_t")
                else np.zeros_like(alt, dtype=float)
            )
        else:
            b = np.asarray(b_t, dtype=float)

        if theta_deg is None:
            psi = (
                np.asarray(self.profile.geomag.psi_deg, dtype=float)
                if self.profile.geomag is not None
                and hasattr(self.profile.geomag, "psi_deg")
                else np.zeros_like(alt, dtype=float)
            )
        else:
            psi = np.asarray(theta_deg, dtype=float)

        form = str(formulation).strip().lower()
        if form == "appleton":
            model = AppletonHartreeDispersion(
                frequency_hz=frequency_hz,
                ne_m3=ne,
                collision_hz=nu,
                b_t=b,
                theta_deg=psi,
            )
        elif form in {"senwyller", "sen-wyller"}:
            model = SenWyllerDispersion(
                frequency_hz=frequency_hz,
                ne_m3=ne,
                collision_hz=nu,
                b_t=b,
                theta_deg=psi,
            )
        else:
            raise ValueError("formulation must be 'appleton' or 'senwyller'")

        n = np.real(model.refractive_index(mode=mode))
        n = np.asarray(n, dtype=float).ravel()
        # guard tiny negatives from numerical noise
        n = np.where(np.isfinite(n), np.clip(n, 0.0, None), np.nan)
        if n.shape != alt.shape:
            raise ValueError("Refractive index profile shape mismatch.")
        return n

    @staticmethod
    def _smooth_nonuniform_grid(
        start: float, end: float, n_points: int, sharpness: float
    ) -> np.ndarray:
        """
        Build a smooth nonuniform coordinate in [start, end] with denser
        sampling near ``end``.
        """
        if int(n_points) < 3:
            raise ValueError("n_points must be >= 3")
        u = np.linspace(0.0, 1.0, int(n_points))
        flipped_u = 1.0 - u
        denom = np.exp(float(sharpness)) - 1.0
        if np.isclose(denom, 0.0):
            return np.linspace(float(start), float(end), int(n_points))
        factor = (np.exp(float(sharpness) * flipped_u) - 1.0) / denom
        return 1.0 - (float(start) + (float(end) - float(start)) * factor)

    def NVIS_tracer(
        self,
        freq_mhz: np.ndarray | float,
        mode: str = "O",
        formulation: str = "appleton",
        collision_hz: np.ndarray | float | None = None,
        b_t: np.ndarray | float | None = None,
        theta_deg: np.ndarray | float | None = None,
        n_floor: float = 1e-8,
        use_nonuniform_grid: bool = True,
        nonuniform_points: int = 240,
        nonuniform_sharpness: float = 10.0,
    ) -> SimpleNamespace:
        """
        Vertical-forward-operator style NVIS tracer for a 1D profile.

        Parameters
        ----------
        freq_mhz : array-like or float
            Sounding frequencies in MHz.
        mode : str, optional
            Dispersion mode selector. Supported values are inherited directly
            from the selected dispersion formulation in ``dispersion.py``:
            - Appleton-Hartree: ``N/NO/ISO``, ``O``, ``X``, ``R``, ``L``
            - Sen-Wyller: ``N/NO/ISO``, ``O``, ``X``, ``R``, ``L``
        formulation : {"appleton", "senwyller"}, optional
            Dispersion backend.
        collision_hz, b_t, theta_deg : array-like or scalar, optional
            Overrides for collision frequency, magnetic field, and angle.
        n_floor : float, optional
            Minimum refractive index used to identify valid propagation layers.
        use_nonuniform_grid : bool, optional
            If True, remap each frequency profile onto a stretched vertical
            grid with denser sampling near the turning altitude.
        nonuniform_points : int, optional
            Number of regridded altitude points used when
            ``use_nonuniform_grid=True``.
        nonuniform_sharpness : float, optional
            Stretching strength for nonuniform grid. Larger values concentrate
            more points near the turning altitude.

        Returns
        -------
        SimpleNamespace
            - ``freq_mhz`` : frequency array [MHz]
            - ``vh_km`` : virtual-height estimate [km]
            - ``turning_height_km`` : turning heights [km]
            - ``n_profile`` : refractive-index profiles [nfreq, nz]
            - ``reason`` : per-frequency status strings

        Notes
        -----
        This method intentionally mirrors a vertical forward operator:
        it integrates an approximate group index ``mu' ~= 1 / n`` from the
        bottom altitude up to the turning point for each frequency.
        """
        z = np.asarray(self.profile.alt_km, dtype=float)
        f_mhz = np.atleast_1d(np.asarray(freq_mhz, dtype=float))
        if np.any(f_mhz <= 0.0):
            raise ValueError("All frequencies must be > 0 MHz.")
        mode = str(mode).upper()
        logger.info(
            "NVIS tracer start: mode={}, formulation={}, freq_points={}, nonuniform_grid={}",
            mode,
            formulation,
            f_mhz.size,
            bool(use_nonuniform_grid),
        )

        n_all = np.full((f_mhz.size, z.size), np.nan, dtype=float)
        vh = np.full(f_mhz.size, np.nan, dtype=float)
        zt = np.full(f_mhz.size, np.nan, dtype=float)
        reason = np.full(f_mhz.size, "no_propagation", dtype=object)

        z0 = float(np.min(z))
        for i, fm in enumerate(f_mhz):
            n = self._refractive_index_profile(
                frequency_hz=float(fm) * 1e6,
                mode=mode,
                formulation=formulation,
                collision_hz=collision_hz,
                b_t=b_t,
                theta_deg=theta_deg,
            )
            n_all[i, :] = n

            mask = np.isfinite(n) & (n > float(n_floor))
            if not np.any(mask):
                reason[i] = "no_propagation"
                continue

            # Use only the first contiguous propagation segment from the bottom.
            # This avoids integrating through disconnected valid layers above a
            # cutoff, which can create non-physical huge virtual heights.
            i_start = int(np.argmax(mask))
            if i_start >= (z.size - 1):
                reason[i] = "no_propagation"
                continue

            i_stop = i_start
            while i_stop < z.size and mask[i_stop]:
                i_stop += 1
            i_top = i_stop - 1
            if (i_top - i_start + 1) < 2:
                reason[i] = "no_propagation"
                continue

            z_up = z[i_start : i_top + 1]
            n_up = n[i_start : i_top + 1]
            if use_nonuniform_grid and z_up.size >= 3:
                m = self._smooth_nonuniform_grid(
                    0.0,
                    1.0,
                    int(nonuniform_points),
                    float(nonuniform_sharpness),
                )
                z_new = z_up[0] + m * (z_up[-1] - z_up[0])
                n_new = np.interp(z_new, z_up, n_up)
                z_up = z_new
                n_up = n_new

            mu_p = 1.0 / np.clip(n_up, float(n_floor), None)
            dz = np.diff(z_up)
            mu_mid = 0.5 * (mu_p[:-1] + mu_p[1:])
            iono_h = np.sum(mu_mid * dz) if dz.size > 0 else 0.0

            zt[i] = float(z_up[-1])
            vh[i] = float(z0 + iono_h)
            reason[i] = "turning" if i_stop < z.size else "top_of_profile"

        return SimpleNamespace(
            freq_mhz=f_mhz,
            vh_km=vh,
            turning_height_km=zt,
            n_profile=n_all,
            reason=reason,
        )

    # Compatibility static helpers.
    den_to_plasma_freq_hz = staticmethod(RT1DProfile.den_to_plasma_freq_hz)
    plasma_freq_to_den = staticmethod(RT1DProfile.plasma_freq_to_den)
    vertical_to_magnetic_angle = staticmethod(RT1DProfile.vertical_to_magnetic_angle)
    inclination_to_vertical_angle = staticmethod(
        RT1DProfile.inclination_to_vertical_angle
    )
    inclintation2vertical = staticmethod(RT1DProfile.inclintation2vertical)
