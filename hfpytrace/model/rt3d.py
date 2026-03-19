"""3D profile-first scaffolding for future ray tracing workflows."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from loguru import logger
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from hfpytrace.collision import NRLMSISE3D
from hfpytrace.density.iri import IRI3d
from hfpytrace.geomag import build_geomag_grid
from hfpytrace.model.dispersion import AppletonHartreeDispersion, SenWyllerDispersion

C_KM_S = 299792.458
N_FLOOR = 1e-6


@dataclass
class RT3DProfile:
    """
    3D gridded ionosphere/background container.

    Axis convention:
    - ``lats``: latitude axis, shape ``(nlat,)``
    - ``lons``: longitude axis, shape ``(nlon,)``
    - ``alts_km``: altitude axis, shape ``(nalt,)``
    - 3D fields use shape ``(nlat, nlon, nalt)``
    """

    lats: np.ndarray
    lons: np.ndarray
    alts_km: np.ndarray
    time: dt.datetime
    ne_m3: np.ndarray | None = None
    ne_cm3: np.ndarray | None = None
    source: str = "iri"
    msise: SimpleNamespace | None = None
    geomag: SimpleNamespace | None = None
    collision: object | None = None  # ComputeCollision instance after compute_collision()

    def __post_init__(self) -> None:
        self.lats = np.asarray(self.lats, dtype=float).ravel()
        self.lons = np.asarray(self.lons, dtype=float).ravel()
        self.alts_km = np.asarray(self.alts_km, dtype=float).ravel()
        if not isinstance(self.time, dt.datetime):
            self.time = dt.datetime.fromisoformat(str(self.time))
        self.validate()

    def validate(self) -> None:
        if self.lats.size < 2 or self.lons.size < 2 or self.alts_km.size < 2:
            raise ValueError("lats, lons, alts_km must each contain at least 2 points")
        if not np.all(np.diff(self.lats) > 0):
            raise ValueError("lats must be strictly increasing")
        if not np.all(np.diff(self.lons) > 0):
            raise ValueError("lons must be strictly increasing")
        if not np.all(np.diff(self.alts_km) > 0):
            raise ValueError("alts_km must be strictly increasing")

        shape = (self.lats.size, self.lons.size, self.alts_km.size)
        if self.ne_m3 is not None:
            ne = np.asarray(self.ne_m3, dtype=float)
            if ne.shape != shape:
                raise ValueError(f"ne_m3 must have shape {shape}")
            if np.any(ne < 0):
                raise ValueError("ne_m3 must be non-negative")
            self.ne_m3 = ne
            self.ne_cm3 = ne * 1e-6
        elif self.ne_cm3 is not None:
            ne = np.asarray(self.ne_cm3, dtype=float)
            if ne.shape != shape:
                raise ValueError(f"ne_cm3 must have shape {shape}")
            if np.any(ne < 0):
                raise ValueError("ne_cm3 must be non-negative")
            self.ne_cm3 = ne
            self.ne_m3 = ne * 1e6

        if self.msise is not None:
            for k in ("N2", "O2", "O", "H", "He", "Tn"):
                if not hasattr(self.msise, k):
                    raise ValueError(f"msise missing required field: {k}")
                arr = np.asarray(getattr(self.msise, k), dtype=float)
                if arr.shape != shape:
                    raise ValueError(f"msise.{k} must have shape {shape}")
                setattr(self.msise, k, arr)

        if self.geomag is not None:
            for k in ("Bx", "By", "Bz", "bmag_t", "inc_deg", "psi_deg"):
                if not hasattr(self.geomag, k):
                    raise ValueError(f"geomag missing required field: {k}")
                arr = np.asarray(getattr(self.geomag, k), dtype=float)
                if arr.shape != shape:
                    raise ValueError(f"geomag.{k} must have shape {shape}")
                setattr(self.geomag, k, arr)

    @staticmethod
    def _axis_from_cfg(start: float, step: float, count: int, name: str) -> np.ndarray:
        n = int(count)
        if n < 2:
            raise ValueError(f"{name} count must be >= 2")
        return float(start) + float(step) * np.arange(n, dtype=float)

    @classmethod
    def from_cfg(
        cls,
        cfg,
        time: dt.datetime | None = None,
        lats: np.ndarray | None = None,
        lons: np.ndarray | None = None,
        alts_km: np.ndarray | None = None,
        fetch_iri: bool = True,
        fetch_msise: bool = False,
        fetch_geomag: bool = False,
        workers: int = 1,
    ) -> "RT3DProfile":
        """
        Build a profile from explicit axes or config-driven 3D grid settings.

        Grid source priority:
        1) explicit lats/lons/alts_km
        2) ``cfg.iono_grid`` fields
        3) fallback from global 2D-style height settings and coarse CONUS-like lat/lon grid
        """
        t = time if time is not None else dt.datetime.fromisoformat(str(cfg.event))

        if lats is None or lons is None or alts_km is None:
            if hasattr(cfg, "iono_grid"):
                ig = cfg.iono_grid
                lats = cls._axis_from_cfg(
                    start=float(ig.lat_start),
                    step=float(ig.lat_step),
                    count=int(ig.num_lats),
                    name="lat",
                )
                lons = cls._axis_from_cfg(
                    start=float(ig.lon_start),
                    step=float(ig.lon_step),
                    count=int(ig.num_lons),
                    name="lon",
                )
                alts_km = cls._axis_from_cfg(
                    start=float(ig.height_start_km),
                    step=float(ig.height_step_km),
                    count=int(ig.num_heights),
                    name="height",
                )
            else:
                logger.warning(
                    "cfg.iono_grid not found; using fallback lat/lon axes and cfg height settings."
                )
                lats = np.linspace(24.0, 50.0, 53)
                lons = np.linspace(-125.0, -66.0, 119)
                h0 = float(getattr(cfg, "start_height_km", 100.0))
                h1 = float(getattr(cfg, "end_height_km", 500.0))
                dh = float(getattr(cfg, "height_incriment_km", 5.0))
                alts_km = np.arange(h0, h1, dh, dtype=float)

        p = cls(
            lats=np.asarray(lats, dtype=float).ravel(),
            lons=np.asarray(lons, dtype=float).ravel(),
            alts_km=np.asarray(alts_km, dtype=float).ravel(),
            time=t,
        )
        logger.info(
            "RT3DProfile created: nlat={}, nlon={}, nalt={}",
            p.lats.size,
            p.lons.size,
            p.alts_km.size,
        )
        if fetch_iri:
            p.fetch_iri(cfg=cfg, workers=int(workers))
        if fetch_msise:
            p.fetch_msise(workers=int(workers))
        if fetch_geomag:
            gm_cfg = getattr(cfg, "geomag_grid", SimpleNamespace(coord_input="GEO"))
            p.fetch_geomag(
                coord_input=str(getattr(gm_cfg, "coord_input", "GEO")),
                coeff_dir=getattr(gm_cfg, "coeff_dir", None),
            )
        p.validate()
        return p

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
            self.ne_m3 = np.asarray(ne_m3, dtype=float)
            self.ne_cm3 = self.ne_m3 * 1e-6
        else:
            self.ne_cm3 = np.asarray(ne_cm3, dtype=float)
            self.ne_m3 = self.ne_cm3 * 1e6
        self.validate()

    def force_zero_density_below(self, min_alt_km: float) -> int:
        """Set all density values to zero for ``alt < min_alt_km``."""
        if self.ne_m3 is None or self.ne_cm3 is None:
            raise ValueError(
                "Electron density is not initialized; call fetch_iri() or set_electron_density() first."
            )
        below = np.asarray(self.alts_km, dtype=float) < float(min_alt_km)
        n_rows = int(np.count_nonzero(below))
        if n_rows == 0:
            return 0
        self.ne_m3[:, :, below] = 0.0
        self.ne_cm3[:, :, below] = 0.0
        self.validate()
        return n_rows

    def fetch_iri(self, cfg, workers: int = 1) -> np.ndarray:
        logger.info(
            "Fetching 3D IRI profile: nlat={}, nlon={}, nalt={}",
            self.lats.size,
            self.lons.size,
            self.alts_km.size,
        )
        model = IRI3d(cfg, self.time)
        ne_cm3, _ = model.fetch_dataset(
            time=self.time,
            lats=self.lats,
            lons=self.lons,
            alts=self.alts_km,
            workers=int(workers),
        )
        self.ne_cm3 = np.asarray(ne_cm3, dtype=float)
        self.ne_m3 = self.ne_cm3 * 1e6
        self.source = "iri"
        self.validate()
        return self.ne_m3

    def fetch_msise(
        self,
        workers: int = 1,
        update_spaceweather: bool = False,
        suppress_spaceweather_warning: bool = True,
    ) -> SimpleNamespace:
        logger.info(
            "Fetching 3D NRLMSISE profile: nlat={}, nlon={}, nalt={}, workers={}",
            self.lats.size,
            self.lons.size,
            self.alts_km.size,
            int(workers),
        )
        ms = NRLMSISE3D(
            date=self.time,
            lats=self.lats,
            lons=self.lons,
            heights_km=self.alts_km,
            workers=int(workers),
            update_spaceweather=bool(update_spaceweather),
            suppress_spaceweather_warning=bool(suppress_spaceweather_warning),
        ).msise
        self.msise = SimpleNamespace(
            N2=np.asarray(ms["N2"], dtype=float),
            O2=np.asarray(ms["O2"], dtype=float),
            O=np.asarray(ms["O"], dtype=float),
            H=np.asarray(ms["H"], dtype=float),
            He=np.asarray(ms["He"], dtype=float),
            Tn=np.asarray(ms["Tn"], dtype=float),
            t_nn=np.asarray(ms["t_nn"], dtype=float),
        )
        self.validate()
        return self.msise

    def fetch_geomag(
        self,
        coord_input: str = "GEO",
        coeff_dir: str | None = None,
    ) -> SimpleNamespace:
        logger.info(
            "Fetching 3D geomag profile: nlat={}, nlon={}, nalt={}",
            self.lats.size,
            self.lons.size,
            self.alts_km.size,
        )
        gm = build_geomag_grid(
            lats=self.lats,
            lons=self.lons,
            alts_km=self.alts_km,
            time=self.time,
            coord_input=coord_input,
            coeff_dir=coeff_dir,
        )
        self.geomag = SimpleNamespace(
            Bx=np.asarray(gm.Bx, dtype=float),
            By=np.asarray(gm.By, dtype=float),
            Bz=np.asarray(gm.Bz, dtype=float),
            bmag_t=np.asarray(gm.bmag_t, dtype=float),
            inc_deg=np.asarray(gm.inc_deg, dtype=float),
            dec_deg=np.asarray(gm.dec_deg, dtype=float),
            psi_deg=np.asarray(gm.psi_deg, dtype=float),
            lat_geo=np.asarray(gm.lat_geo, dtype=float),
            lon_geo=np.asarray(gm.lon_geo, dtype=float),
            qd=gm.qd,
            apex=gm.apex,
        )
        self.validate()
        return self.geomag

    def compute_collision(
        self,
        Te: np.ndarray | float | None = None,
        Ti: np.ndarray | float | None = None,
        edens: np.ndarray | None = None,
        O2p: np.ndarray | None = None,
        Op: np.ndarray | None = None,
    ) -> object:
        """
        Compute collision frequencies using the already-fetched MSIS neutral data.

        Requires ``self.msise`` (call ``fetch_msise()`` first) and electron
        density (call ``fetch_iri()`` or ``set_electron_density()`` first).

        Parameters
        ----------
        Te, Ti : array-like or float, optional
            Electron/ion temperature [K], shape (nlat, nlon, nalt) or broadcastable.
            Defaults to MSIS neutral temperature Tn.
        edens : array-like, optional
            Electron density [cm^-3], shape (nlat, nlon, nalt).
            Defaults to ``self.ne_cm3``.
        O2p, Op : array-like, optional
            O2+ and O+ densities [cm^-3]. Defaults to 10%/90% of edens.

        Returns
        -------
        ComputeCollision
            Also stored on ``self.collision`` for retrieval via
            ``collision_type`` in :class:`RT3D`.

        Notes
        -----
        Supported ``collision_type`` keys:

        +-------------+--------------------------------------------------+
        | Key         | Model                                            |
        +=============+==================================================+
        | ``"FT"``    | Friedrich-Tonker (ν_ft, a=1.0)                   |
        | ``"FT_cc"`` | Friedrich-Tonker (ν_av_cc, a=2.5)                |
        | ``"FT_mb"`` | Friedrich-Tonker (ν_av_mb, a=1.5)                |
        | ``"SN_en"`` | Schunk-Nagy electron-neutral total               |
        | ``"SN_ei"`` | Schunk-Nagy electron-ion total                   |
        | ``"SN"``    | Schunk-Nagy full (en + ei)                       |
        | ``"atm"``   | Atmospheric ion-neutral approximation            |
        +-------------+--------------------------------------------------+
        """
        from hfpytrace.collision import ComputeCollision

        if self.msise is None:
            raise ValueError(
                "MSIS neutral data not available. Call fetch_msise() first."
            )
        if self.ne_cm3 is None:
            raise ValueError(
                "Electron density not set. "
                "Call fetch_iri() or set_electron_density() first."
            )

        Tn = np.asarray(self.msise.Tn, dtype=float)   # shape (nlat, nlon, nalt)
        ne = np.asarray(self.ne_cm3, dtype=float)

        Te_use = np.asarray(Te, dtype=float) if Te is not None else Tn.copy()
        Ti_use = np.asarray(Ti, dtype=float) if Ti is not None else Tn.copy()
        edens_use = np.asarray(edens, dtype=float) if edens is not None else ne.copy()
        Op_use = np.asarray(Op, dtype=float) if Op is not None else 0.9 * ne
        O2p_use = np.asarray(O2p, dtype=float) if O2p is not None else 0.1 * ne

        cc = ComputeCollision(
            Te=Te_use,
            Ti=Ti_use,
            Tn=Tn,
            edens=edens_use,
            O2p=O2p_use,
            Op=Op_use,
            N2=np.asarray(self.msise.N2, dtype=float),
            O2=np.asarray(self.msise.O2, dtype=float),
            O=np.asarray(self.msise.O, dtype=float),
            H=np.asarray(self.msise.H, dtype=float),
            He=np.asarray(self.msise.He, dtype=float),
            date=self.time,
        )
        self.collision = cc
        logger.info(
            "3D collision computed: nu_ft=[{:.3e},{:.3e}] Hz, nu_sn=[{:.3e},{:.3e}] Hz",
            float(np.nanmin(cc.collision.nu_ft)),
            float(np.nanmax(cc.collision.nu_ft)),
            float(np.nanmin(cc.collision.nu_sn.total)),
            float(np.nanmax(cc.collision.nu_sn.total)),
        )
        return cc


class RT3D:
    """
    Minimal RT3D container for downstream 3D tracing implementations.

    This class currently focuses on profile management and data integrity checks.
    """

    _VALID_COLLISION_TYPES: frozenset[str] = frozenset(
        {"FT", "FT_CC", "FT_MB", "SN_EN", "SN_EI", "SN", "ATM"}
    )

    @staticmethod
    def _extract_collision_hz(cc, collision_type: str) -> np.ndarray:
        """
        Extract a collision-frequency array from a ``ComputeCollision`` object.

        The returned array has the same shape as the 3D profile fields
        ``(nlat, nlon, nalt)`` and can be sliced/interpolated for ray tracing.

        Parameters
        ----------
        cc : ComputeCollision
        collision_type : str
            One of ``"FT"``, ``"FT_cc"``, ``"FT_mb"``, ``"SN_en"``,
            ``"SN_ei"``, ``"SN"``, ``"atm"`` (case-insensitive).
        """
        ct = str(collision_type).strip().upper()
        _map = {
            "FT":    lambda c: np.asarray(c.collision.nu_ft,          dtype=float),
            "FT_CC": lambda c: np.asarray(c.collision.nu_av_cc,       dtype=float),
            "FT_MB": lambda c: np.asarray(c.collision.nu_av_mb,       dtype=float),
            "SN_EN": lambda c: np.asarray(c.collision.nu_sn.en.total, dtype=float),
            "SN_EI": lambda c: np.asarray(c.collision.nu_sn.ei.total, dtype=float),
            "SN":    lambda c: np.asarray(c.collision.nu_sn.total,    dtype=float),
            "ATM":   lambda c: np.asarray(
                         c.atmospheric_ion_neutral_collision_frequency(), dtype=float
                     ),
        }
        if ct not in _map:
            raise ValueError(
                f"Unknown collision_type '{collision_type}'. "
                f"Valid options: {sorted(_map.keys())}"
            )
        return _map[ct](cc)

    def fetch_collision(
        self,
        Te: np.ndarray | float | None = None,
        Ti: np.ndarray | float | None = None,
        edens: np.ndarray | None = None,
        O2p: np.ndarray | None = None,
        Op: np.ndarray | None = None,
    ) -> object:
        """
        Compute and attach collision frequencies to the profile.

        Convenience wrapper around :meth:`RT3DProfile.compute_collision`.
        Requires that ``fetch_msise()`` has been called on the profile.

        Returns
        -------
        ComputeCollision
        """
        return self.profile.compute_collision(
            Te=Te, Ti=Ti, edens=edens, O2p=O2p, Op=Op,
        )

    def __init__(
        self,
        *,
        profile: RT3DProfile | None = None,
        cfg=None,
        time: dt.datetime | str | None = None,
        lats: np.ndarray | None = None,
        lons: np.ndarray | None = None,
        alts_km: np.ndarray | None = None,
        ne_m3: np.ndarray | None = None,
        ne_cm3: np.ndarray | None = None,
        source: str = "iri",
        fetch_iri: bool = False,
        fetch_msise: bool = False,
        fetch_geomag: bool = False,
        workers: int = 1,
    ):
        if profile is not None:
            if not isinstance(profile, RT3DProfile):
                raise TypeError("profile must be an RT3DProfile")
            profile.validate()
            self.profile = profile
        else:
            if cfg is None:
                raise ValueError("Provide profile or cfg for RT3D initialization")
            t = time if time is not None else dt.datetime.fromisoformat(str(cfg.event))
            self.profile = RT3DProfile.from_cfg(
                cfg=cfg,
                time=t,
                lats=lats,
                lons=lons,
                alts_km=alts_km,
                fetch_iri=bool(fetch_iri),
                fetch_msise=bool(fetch_msise),
                fetch_geomag=bool(fetch_geomag),
                workers=int(workers),
            )

        if (ne_m3 is not None) or (ne_cm3 is not None):
            self.profile.set_electron_density(ne_m3=ne_m3, ne_cm3=ne_cm3, source=source)
        self.profile.validate()
        if self.profile.ne_m3 is None:
            logger.warning(
                "RT3D initialized without electron density; set ne_m3/ne_cm3 or fetch_iri=True."
            )
        logger.info(
            "RT3D initialized: nlat={}, nlon={}, nalt={}, source={}",
            self.profile.lats.size,
            self.profile.lons.size,
            self.profile.alts_km.size,
            self.profile.source,
        )
        # Cache for expensive refractive-index/interpolator construction.
        self._interp_cache_key = None
        self._interp_cache_out = None

    @property
    def lats(self) -> np.ndarray:
        return self.profile.lats

    @property
    def lons(self) -> np.ndarray:
        return self.profile.lons

    @property
    def alts_km(self) -> np.ndarray:
        return self.profile.alts_km

    @property
    def ne_m3(self) -> np.ndarray | None:
        return self.profile.ne_m3

    @staticmethod
    def _resolve_dispersion_model_name(formulation: str) -> str:
        name = str(formulation).strip().lower().replace("_", "-")
        alias_map = {
            "appleton": "appleton-hartree",
            "appleton-hartree": "appleton-hartree",
            "ah": "appleton-hartree",
            "senwyller": "sen-wyller",
            "sen-wyller": "sen-wyller",
            "sw": "sen-wyller",
        }
        if name not in alias_map:
            raise ValueError(
                "Unsupported dispersion model. Use 'appleton-hartree' or 'sen-wyller'."
            )
        return alias_map[name]

    def _build_local_xy_axes(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        lat0 = float(np.mean(self.lats))
        lon0 = float(np.mean(self.lons))
        km_per_deg_lat = 111.32
        km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(lat0))
        x_km = (np.asarray(self.lons, dtype=float) - lon0) * km_per_deg_lon
        y_km = (np.asarray(self.lats, dtype=float) - lat0) * km_per_deg_lat
        return x_km, y_km, lat0, lon0

    def build_refractive_index_interpolators(
        self,
        freq_hz: float,
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        collision_hz: np.ndarray | float | None = None,
        mode: str = "O",
        formulation: str = "appleton-hartree",
    ) -> SimpleNamespace:
        cache_key = (
            float(freq_hz),
            str(mode),
            str(formulation),
            id(b_abs_t) if b_abs_t is not None else None,
            id(b_psi_deg) if b_psi_deg is not None else None,
            id(collision_hz) if collision_hz is not None else None,
            id(self.profile.ne_m3),
        )
        if self._interp_cache_key == cache_key and self._interp_cache_out is not None:
            return self._interp_cache_out

        if self.ne_m3 is None:
            raise ValueError("RT3D profile has no electron density.")
        ne = np.asarray(self.ne_m3, dtype=float)
        shape = ne.shape
        model_name = self._resolve_dispersion_model_name(formulation)

        if b_abs_t is None:
            if self.profile.geomag is not None:
                b_t = np.asarray(self.profile.geomag.bmag_t, dtype=float)
            else:
                b_t = np.zeros(shape, dtype=float)
        else:
            b_t = np.asarray(b_abs_t, dtype=float)
            if b_t.ndim == 0:
                b_t = np.full(shape, float(b_t), dtype=float)
        if b_psi_deg is None:
            if self.profile.geomag is not None:
                theta = np.asarray(self.profile.geomag.psi_deg, dtype=float)
            else:
                theta = np.zeros(shape, dtype=float)
        else:
            theta = np.asarray(b_psi_deg, dtype=float)
            if theta.ndim == 0:
                theta = np.full(shape, float(theta), dtype=float)
        if collision_hz is None:
            nu = np.zeros(shape, dtype=float)
        else:
            nu = np.asarray(collision_hz, dtype=float)
            if nu.ndim == 0:
                nu = np.full(shape, float(nu), dtype=float)
            if nu.shape != shape:
                raise ValueError(f"collision_hz must have shape {shape}")

        if model_name == "appleton-hartree":
            disp = AppletonHartreeDispersion(
                frequency_hz=float(freq_hz),
                ne_m3=ne,
                collision_hz=nu,
                b_t=b_t,
                theta_deg=theta,
            )
        else:
            disp = SenWyllerDispersion(
                frequency_hz=float(freq_hz),
                ne_m3=ne,
                collision_hz=nu,
                b_t=b_t,
                theta_deg=theta,
            )

        n_complex = disp.refractive_index(mode=mode)
        n = np.real(n_complex)
        # Harden against singular/invalid dispersion outputs.
        n = np.where(np.isfinite(n), np.clip(n, 0.0, None), np.nan)
        if np.any(~np.isfinite(n)):
            n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.clip(n, N_FLOOR, None)
        mup = 1.0 / np.clip(n, N_FLOOR, None)

        # 3D interpolators on physical local axes (z, y, x).
        x_km, y_km, lat_ref_deg, lon_ref_deg = self._build_local_xy_axes()
        z_km = np.asarray(self.alts_km, dtype=float)
        n_zyx = np.transpose(n, (2, 0, 1))
        mup_zyx = np.transpose(mup, (2, 0, 1))
        dn_dz, dn_dy, dn_dx = np.gradient(n_zyx, z_km, y_km, x_km, edge_order=2)

        self._x_km = x_km
        self._y_km = y_km
        self._z_km = z_km
        self._lat_ref_deg = lat_ref_deg
        self._lon_ref_deg = lon_ref_deg

        self._n_interp_zyx = RegularGridInterpolator(
            (z_km, y_km, x_km), n_zyx, bounds_error=False, fill_value=np.nan
        )
        self._mup_interp_zyx = RegularGridInterpolator(
            (z_km, y_km, x_km), mup_zyx, bounds_error=False, fill_value=np.nan
        )
        self._dn_dx_interp_zyx = RegularGridInterpolator(
            (z_km, y_km, x_km), dn_dx, bounds_error=False, fill_value=np.nan
        )
        self._dn_dy_interp_zyx = RegularGridInterpolator(
            (z_km, y_km, x_km), dn_dy, bounds_error=False, fill_value=np.nan
        )
        self._dn_dz_interp_zyx = RegularGridInterpolator(
            (z_km, y_km, x_km), dn_dz, bounds_error=False, fill_value=np.nan
        )

        # Also keep altitude-lat-lon interpolators for spherical mode.
        self._n_interp_altll = RegularGridInterpolator(
            (self.alts_km, self.lats, self.lons),
            np.transpose(n, (2, 0, 1)),
            bounds_error=False,
            fill_value=np.nan,
        )
        self._mup_interp_altll = RegularGridInterpolator(
            (self.alts_km, self.lats, self.lons),
            np.transpose(mup, (2, 0, 1)),
            bounds_error=False,
            fill_value=np.nan,
        )
        logger.info(
            "RT3D refractive index interpolators ready: freq={} Hz, mode={}, model={}",
            float(freq_hz),
            mode,
            model_name,
        )
        out = SimpleNamespace(n=n, mup=mup)
        self._interp_cache_key = cache_key
        self._interp_cache_out = out
        return out

    def _eval_n_grad_cart(
        self, x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.atleast_1d(np.asarray(x_km, dtype=float))
        y = np.atleast_1d(np.asarray(y_km, dtype=float))
        z = np.atleast_1d(np.asarray(z_km, dtype=float))
        x, y, z = np.broadcast_arrays(x, y, z)
        if hasattr(self, "_x_km") and hasattr(self, "_y_km") and hasattr(self, "_z_km"):
            eps = 1e-6
            x = np.clip(x, float(self._x_km[0]) + eps, float(self._x_km[-1]) - eps)
            y = np.clip(y, float(self._y_km[0]) + eps, float(self._y_km[-1]) - eps)
            z = np.clip(z, float(self._z_km[0]) + eps, float(self._z_km[-1]) - eps)
        pts = np.column_stack([z.ravel(), y.ravel(), x.ravel()])
        n = self._n_interp_zyx(pts).reshape(x.shape)
        dnx = self._dn_dx_interp_zyx(pts).reshape(x.shape)
        dny = self._dn_dy_interp_zyx(pts).reshape(x.shape)
        dnz = self._dn_dz_interp_zyx(pts).reshape(x.shape)
        return n, dnx, dny, dnz

    def _eval_mup_cart(
        self, x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray
    ) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x_km, dtype=float))
        y = np.atleast_1d(np.asarray(y_km, dtype=float))
        z = np.atleast_1d(np.asarray(z_km, dtype=float))
        x, y, z = np.broadcast_arrays(x, y, z)
        if hasattr(self, "_x_km") and hasattr(self, "_y_km") and hasattr(self, "_z_km"):
            eps = 1e-6
            x = np.clip(x, float(self._x_km[0]) + eps, float(self._x_km[-1]) - eps)
            y = np.clip(y, float(self._y_km[0]) + eps, float(self._y_km[-1]) - eps)
            z = np.clip(z, float(self._z_km[0]) + eps, float(self._z_km[-1]) - eps)
        pts = np.column_stack([z.ravel(), y.ravel(), x.ravel()])
        return self._mup_interp_zyx(pts).reshape(x.shape)

    @staticmethod
    def _rhs_cart(_s: float, y: np.ndarray, n_grad_fn) -> np.ndarray:
        x, yy, z, vx, vy, vz = y
        n, dnx, dny, dnz = n_grad_fn(np.array([x]), np.array([yy]), np.array([z]))
        n = float(n[0])
        dnx = float(dnx[0])
        dny = float(dny[0])
        dnz = float(dnz[0])
        if not np.isfinite(n) or n <= 0.0:
            return np.zeros(6, dtype=float)
        dot = dnx * vx + dny * vy + dnz * vz
        dvx = (dnx - dot * vx) / n
        dvy = (dny - dot * vy) / n
        dvz = (dnz - dot * vz) / n
        return np.array([vx, vy, vz, dvx, dvy, dvz], dtype=float)

    def trace_cartesian_gradient(
        self,
        freq_hz: float,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        x0_km: float = 0.0,
        y0_km: float = 0.0,
        z0_km: float = 0.0,
        s_max_km: float = 6000.0,
        mode: str = "O",
        collision_hz: np.ndarray | float | None = None,
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        formulation: str = "appleton-hartree",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_km: float | None = None,
    ) -> SimpleNamespace:
        self.build_refractive_index_interpolators(
            freq_hz=freq_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            collision_hz=collision_hz,
            mode=mode,
            formulation=formulation,
        )
        elev = np.deg2rad(float(elevation_deg))
        az = np.deg2rad(float(azimuth_deg))
        vx0 = np.cos(elev) * np.cos(az)
        vy0 = np.cos(elev) * np.sin(az)
        vz0 = np.sin(elev)
        vnorm = np.linalg.norm([vx0, vy0, vz0])
        vx0, vy0, vz0 = vx0 / vnorm, vy0 / vnorm, vz0 / vnorm
        yinit = np.array([x0_km, y0_km, z0_km, vx0, vy0, vz0], dtype=float)

        zmin, zmax = float(self._z_km[0]), float(self._z_km[-1])
        ymin, ymax = float(self._y_km[0]), float(self._y_km[-1])
        xmin, xmax = float(self._x_km[0]), float(self._x_km[-1])

        def ev_zmin(_s, y):
            return y[2] - zmin - 1e-3

        def ev_zmax(_s, y):
            return zmax - y[2]

        def ev_ymin(_s, y):
            return y[1] - ymin

        def ev_ymax(_s, y):
            return ymax - y[1]

        def ev_xmin(_s, y):
            return y[0] - xmin

        def ev_xmax(_s, y):
            return xmax - y[0]

        for ev in (ev_zmin, ev_zmax, ev_ymin, ev_ymax, ev_xmin, ev_xmax):
            ev.terminal = True
            ev.direction = -1.0

        sol = solve_ivp(
            lambda s, y: self._rhs_cart(s, y, self._eval_n_grad_cart),
            (0.0, float(s_max_km)),
            yinit,
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step_km) if max_step_km is not None else np.inf,
            events=[ev_zmin, ev_zmax, ev_ymin, ev_ymax, ev_xmin, ev_xmax],
        )
        x = sol.y[0, :]
        yy = sol.y[1, :]
        z = sol.y[2, :]
        ds = np.sqrt(np.diff(x) ** 2 + np.diff(yy) ** 2 + np.diff(z) ** 2)
        group_path_km = float(np.nansum(ds))
        xm = 0.5 * (x[:-1] + x[1:])
        ym = 0.5 * (yy[:-1] + yy[1:])
        zm = 0.5 * (z[:-1] + z[1:])
        mup = np.asarray(self._eval_mup_cart(xm, ym, zm), dtype=float)
        valid = np.isfinite(mup)
        group_delay_sec = float(np.nansum((mup[valid] / C_KM_S) * ds[valid]))
        status = "length"
        if sol.status == 1:
            status = "ground" if len(sol.t_events[0]) > 0 else "domain"
        elif sol.status == -1:
            status = "failure"
        return SimpleNamespace(
            x_km=x,
            y_km=yy,
            z_km=z,
            vx=sol.y[3, :],
            vy=sol.y[4, :],
            vz=sol.y[5, :],
            t=sol.t,
            status=status,
            reason=status,
            group_path_km=group_path_km,
            group_delay_sec=group_delay_sec,
            mode=mode,
            coordinate_system="cartesian",
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            azimuth_deg=float(azimuth_deg),
        )

    def trace_cartesian_hamiltonian(
        self,
        freq_hz: float,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        x0_km: float = 0.0,
        y0_km: float = 0.0,
        z0_km: float = 0.0,
        s_max_km: float = 6000.0,
        mode: str = "O",
        collision_hz: np.ndarray | float | None = None,
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        formulation: str = "appleton-hartree",
        h0_km: float = 2.0,
        h_min_km: float = 0.25,
        h_max_km: float = 8.0,
        max_step_km: float | None = None,
        local_err_tol_km: float = 5e-3,
        max_steps: int = 20000,
        boundary_eps_km: float = 1e-3,
    ) -> SimpleNamespace:
        """
        Adaptive 3D Cartesian Hamiltonian solver for isotropic n(x).

        Hamiltonian:
            H(x, p) = 0.5 (|p|^2 - n(x)^2) = 0
        Equations:
            dx/dtau = p
            dp/dtau = 0.5 * grad(n^2) = n * grad(n)
        """
        self.build_refractive_index_interpolators(
            freq_hz=freq_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            collision_hz=collision_hz,
            mode=mode,
            formulation=formulation,
        )

        zmin, zmax = float(self._z_km[0]), float(self._z_km[-1])
        ymin, ymax = float(self._y_km[0]), float(self._y_km[-1])
        xmin, xmax = float(self._x_km[0]), float(self._x_km[-1])
        beps = max(0.0, float(boundary_eps_km))

        def _inside_domain(x: float, yv: float, z: float, margin: float = 0.0) -> bool:
            return (
                (xmin + margin) <= float(x) <= (xmax - margin)
                and (ymin + margin) <= float(yv) <= (ymax - margin)
                and (zmin + margin) <= float(z) <= (zmax - margin)
            )

        def _deriv(yvec: np.ndarray):
            x, yy, z, px, py, pz = yvec
            n, dnx, dny, dnz = self._eval_n_grad_cart(
                np.array([x]), np.array([yy]), np.array([z])
            )
            n = float(n[0])
            dnx = float(dnx[0])
            dny = float(dny[0])
            dnz = float(dnz[0])
            if (not np.isfinite(n)) or (n <= 0.0):
                return None
            dxdt = np.array([px, py, pz], dtype=float)
            dpdt = np.array([n * dnx, n * dny, n * dnz], dtype=float)
            return np.concatenate([dxdt, dpdt])

        def _rk4(yvec: np.ndarray, h: float):
            k1 = _deriv(yvec)
            if k1 is None:
                return None
            k2 = _deriv(yvec + 0.5 * h * k1)
            if k2 is None:
                return None
            k3 = _deriv(yvec + 0.5 * h * k2)
            if k3 is None:
                return None
            k4 = _deriv(yvec + h * k3)
            if k4 is None:
                return None
            return yvec + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Initial canonical momentum p0 = n0 * ray_direction_unit
        elev = np.deg2rad(float(elevation_deg))
        az = np.deg2rad(float(azimuth_deg))
        ux = np.cos(elev) * np.cos(az)
        uy = np.cos(elev) * np.sin(az)
        uz = np.sin(elev)
        u = np.array([ux, uy, uz], dtype=float)
        un = np.linalg.norm(u)
        if un <= 0:
            raise ValueError("Invalid launch direction norm")
        u /= un
        n0_arr = self._eval_n_grad_cart(
            np.array([x0_km]), np.array([y0_km]), np.array([z0_km])
        )[0]
        n0 = float(n0_arr[0])
        if (not np.isfinite(n0)) or (n0 <= 0.0):
            raise ValueError("Launch point refractive index is invalid")
        p0 = n0 * u
        y = np.array([x0_km, y0_km, z0_km, p0[0], p0[1], p0[2]], dtype=float)

        if max_step_km is not None:
            h_max_km = min(float(h_max_km), float(max_step_km))
        h = float(np.clip(h0_km, h_min_km, h_max_km))
        tau = 0.0
        tau_max = float(s_max_km)
        xs = [float(y[0])]
        ys = [float(y[1])]
        zs = [float(y[2])]
        pxs = [float(y[3])]
        pys = [float(y[4])]
        pzs = [float(y[5])]
        status = "length"

        for _ in range(int(max_steps)):
            if tau >= tau_max:
                status = "length"
                break
            # Boundary handling: allow starts on boundaries if the ray points inward.
            if y[2] <= zmin + 1e-3 and y[5] < 0:
                status = "ground"
                break
            if (
                (y[2] >= zmax and y[5] > 0)
                or (y[1] <= ymin and y[4] < 0)
                or (y[1] >= ymax and y[4] > 0)
                or (y[0] <= xmin and y[3] < 0)
                or (y[0] >= xmax and y[3] > 0)
            ):
                status = "domain"
                break

            h_try = min(h, tau_max - tau)
            y_big = _rk4(y, h_try)
            if y_big is None:
                if h_try > h_min_km:
                    h = max(h_min_km, 0.5 * h_try)
                    continue
                status = "failure"
                break
            y_half = _rk4(y, 0.5 * h_try)
            if y_half is None:
                if h_try > h_min_km:
                    h = max(h_min_km, 0.5 * h_try)
                    continue
                status = "failure"
                break
            y_half2 = _rk4(y_half, 0.5 * h_try)
            if y_half2 is None:
                if h_try > h_min_km:
                    h = max(h_min_km, 0.5 * h_try)
                    continue
                status = "failure"
                break

            # If the candidate step exits the safe interpolation interior,
            # reduce step first; if already at minimum step, terminate as domain.
            if not _inside_domain(y_half2[0], y_half2[1], y_half2[2], margin=beps):
                if h_try > h_min_km:
                    h = max(h_min_km, 0.5 * h_try)
                    continue
                status = "domain" if y_half2[2] > zmin + beps else "ground"
                break

            err = np.linalg.norm(y_half2[:3] - y_big[:3], ord=2)
            if np.isfinite(err) and err > local_err_tol_km and h_try > h_min_km:
                h = max(h_min_km, 0.5 * h_try)
                continue

            # Accept higher-accuracy step (step-doubling result)
            y = y_half2
            # Project momentum back to |p| = n to reduce drift.
            n_now = self._eval_n_grad_cart(
                np.array([y[0]]), np.array([y[1]]), np.array([y[2]])
            )[0][0]
            pnorm = np.linalg.norm(y[3:6])
            if np.isfinite(n_now) and n_now > 0 and pnorm > 0:
                y[3:6] = y[3:6] * (float(n_now) / float(pnorm))

            tau += h_try
            xs.append(float(y[0]))
            ys.append(float(y[1]))
            zs.append(float(y[2]))
            pxs.append(float(y[3]))
            pys.append(float(y[4]))
            pzs.append(float(y[5]))

            if np.isfinite(err) and err < 0.25 * local_err_tol_km:
                h = min(h_max_km, 2.0 * h_try)
            else:
                h = h_try
        else:
            status = "failure"

        x = np.asarray(xs, dtype=float)
        yy = np.asarray(ys, dtype=float)
        z = np.asarray(zs, dtype=float)
        px = np.asarray(pxs, dtype=float)
        py = np.asarray(pys, dtype=float)
        pz = np.asarray(pzs, dtype=float)

        ds = np.sqrt(np.diff(x) ** 2 + np.diff(yy) ** 2 + np.diff(z) ** 2)
        group_path_km = float(np.nansum(ds))
        xm = 0.5 * (x[:-1] + x[1:])
        ym = 0.5 * (yy[:-1] + yy[1:])
        zm = 0.5 * (z[:-1] + z[1:])
        mup = np.asarray(self._eval_mup_cart(xm, ym, zm), dtype=float)
        valid = np.isfinite(mup)
        group_delay_sec = float(np.nansum((mup[valid] / C_KM_S) * ds[valid]))

        # Direction estimates from canonical momentum.
        pnorm = np.sqrt(px**2 + py**2 + pz**2)
        vx = np.divide(px, pnorm, out=np.zeros_like(px), where=pnorm > 0)
        vy = np.divide(py, pnorm, out=np.zeros_like(py), where=pnorm > 0)
        vz = np.divide(pz, pnorm, out=np.zeros_like(pz), where=pnorm > 0)

        return SimpleNamespace(
            x_km=x,
            y_km=yy,
            z_km=z,
            vx=vx,
            vy=vy,
            vz=vz,
            px=px,
            py=py,
            pz=pz,
            t=np.linspace(0.0, tau, x.size),
            status=status,
            reason=status,
            group_path_km=group_path_km,
            group_delay_sec=group_delay_sec,
            mode=mode,
            coordinate_system="cartesian",
            solver="hamiltonian",
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            azimuth_deg=float(azimuth_deg),
        )

    def _eval_n_sph(
        self, alt_km: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray
    ) -> np.ndarray:
        alt = np.atleast_1d(np.asarray(alt_km, dtype=float))
        lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
        lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
        alt, lat, lon = np.broadcast_arrays(alt, lat, lon)
        pts = np.column_stack([alt.ravel(), lat.ravel(), lon.ravel()])
        return self._n_interp_altll(pts).reshape(alt.shape)

    def _eval_mup_sph(
        self, alt_km: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray
    ) -> np.ndarray:
        alt = np.atleast_1d(np.asarray(alt_km, dtype=float))
        lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
        lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
        alt, lat, lon = np.broadcast_arrays(alt, lat, lon)
        pts = np.column_stack([alt.ravel(), lat.ravel(), lon.ravel()])
        return self._mup_interp_altll(pts).reshape(alt.shape)

    def _eval_n_grad_sph(
        self,
        r_km: np.ndarray,
        lat_rad: np.ndarray,
        lon_rad: np.ndarray,
        r_earth_km: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = np.atleast_1d(np.asarray(r_km, dtype=float))
        lat = np.atleast_1d(np.asarray(lat_rad, dtype=float))
        lon = np.atleast_1d(np.asarray(lon_rad, dtype=float))
        r, lat, lon = np.broadcast_arrays(r, lat, lon)
        alt = r - float(r_earth_km)
        lat_deg = np.rad2deg(lat)
        lon_deg = np.rad2deg(lon)
        n = self._eval_n_sph(alt, lat_deg, lon_deg)

        dr = 1.0
        da = np.deg2rad(0.02)
        dl = np.deg2rad(0.02)
        n_rp = self._eval_n_sph(alt + dr, lat_deg, lon_deg)
        n_rm = self._eval_n_sph(alt - dr, lat_deg, lon_deg)
        n_ap = self._eval_n_sph(alt, np.rad2deg(lat + da), lon_deg)
        n_am = self._eval_n_sph(alt, np.rad2deg(lat - da), lon_deg)
        n_lp = self._eval_n_sph(alt, lat_deg, np.rad2deg(lon + dl))
        n_lm = self._eval_n_sph(alt, lat_deg, np.rad2deg(lon - dl))

        dn_dr = (n_rp - n_rm) / (2.0 * dr)
        dn_dlat = (n_ap - n_am) / (2.0 * da)
        dn_dlon = (n_lp - n_lm) / (2.0 * dl)
        return n, dn_dr, dn_dlat, dn_dlon

    @staticmethod
    def _rhs_sph(_s: float, y: np.ndarray, n_grad_fn) -> np.ndarray:
        r, lat, lon, vr, vlat, vlon = y
        n, dn_dr, dn_dlat, dn_dlon = n_grad_fn(
            np.array([r]), np.array([lat]), np.array([lon])
        )
        n = float(n[0])
        dn_dr = float(dn_dr[0])
        dn_dlat = float(dn_dlat[0])
        dn_dlon = float(dn_dlon[0])
        cl = max(np.cos(lat), 1e-6)
        if not np.isfinite(n) or n <= 0.0 or r <= 0.0:
            return np.zeros(6, dtype=float)
        grad_dot_v = dn_dr * vr + (dn_dlat / r) * vlat + (dn_dlon / (r * cl)) * vlon
        drds = vr
        dlatds = vlat / r
        dlonds = vlon / (r * cl)
        dvr = (dn_dr - grad_dot_v * vr) / n + (vlat * vlat + vlon * vlon) / r
        dvlat = (
            ((dn_dlat / r) - grad_dot_v * vlat) / n
            - (vr * vlat) / r
            + (vlon * vlon * np.tan(lat)) / r
        )
        dvlon = (
            ((dn_dlon / (r * cl)) - grad_dot_v * vlon) / n
            - (vr * vlon) / r
            - (vlat * vlon * np.tan(lat)) / r
        )
        return np.array([drds, dlatds, dlonds, dvr, dvlat, dvlon], dtype=float)

    def trace_spherical_gradient(
        self,
        freq_hz: float,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        x0_km: float = 0.0,
        y0_km: float = 0.0,
        z0_km: float = 0.0,
        s_max_km: float = 7000.0,
        mode: str = "O",
        collision_hz: np.ndarray | float | None = None,
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        formulation: str = "appleton-hartree",
        r_earth_km: float = 6371.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_km: float | None = None,
    ) -> SimpleNamespace:
        self.build_refractive_index_interpolators(
            freq_hz=freq_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            collision_hz=collision_hz,
            mode=mode,
            formulation=formulation,
        )
        km_per_deg_lat = 111.32
        km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(self._lat_ref_deg))
        lat0_deg = self._lat_ref_deg + float(y0_km) / km_per_deg_lat
        lon0_deg = self._lon_ref_deg + float(x0_km) / max(km_per_deg_lon, 1e-6)
        r0 = float(r_earth_km) + float(z0_km)
        lat0 = np.deg2rad(lat0_deg)
        lon0 = np.deg2rad(lon0_deg)

        elev = np.deg2rad(float(elevation_deg))
        az = np.deg2rad(float(azimuth_deg))
        vr0 = np.sin(elev)
        vh = np.cos(elev)
        vlat0 = vh * np.cos(az)
        vlon0 = vh * np.sin(az)
        vnorm = np.linalg.norm([vr0, vlat0, vlon0])
        vr0, vlat0, vlon0 = vr0 / vnorm, vlat0 / vnorm, vlon0 / vnorm
        yinit = np.array([r0, lat0, lon0, vr0, vlat0, vlon0], dtype=float)

        rmin = float(r_earth_km) + float(self.alts_km[0])
        rmax = float(r_earth_km) + float(self.alts_km[-1])
        lat_min = np.deg2rad(float(self.lats[0]))
        lat_max = np.deg2rad(float(self.lats[-1]))
        lon_min = np.deg2rad(float(self.lons[0]))
        lon_max = np.deg2rad(float(self.lons[-1]))

        def ev_rmin(_s, y):
            return y[0] - rmin - 1e-3

        def ev_rmax(_s, y):
            return rmax - y[0]

        def ev_latmin(_s, y):
            return y[1] - lat_min

        def ev_latmax(_s, y):
            return lat_max - y[1]

        def ev_lonmin(_s, y):
            return y[2] - lon_min

        def ev_lonmax(_s, y):
            return lon_max - y[2]

        for ev in (ev_rmin, ev_rmax, ev_latmin, ev_latmax, ev_lonmin, ev_lonmax):
            ev.terminal = True
            ev.direction = -1.0

        sol = solve_ivp(
            lambda s, y: self._rhs_sph(
                s,
                y,
                lambda r, lat, lon: self._eval_n_grad_sph(
                    r_km=r, lat_rad=lat, lon_rad=lon, r_earth_km=float(r_earth_km)
                ),
            ),
            (0.0, float(s_max_km)),
            yinit,
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step_km) if max_step_km is not None else np.inf,
            events=[ev_rmin, ev_rmax, ev_latmin, ev_latmax, ev_lonmin, ev_lonmax],
        )

        r = sol.y[0, :]
        lat = sol.y[1, :]
        lon = sol.y[2, :]
        z = r - float(r_earth_km)
        lat_deg = np.rad2deg(lat)
        lon_deg = np.rad2deg(lon)
        x = (lon_deg - self._lon_ref_deg) * km_per_deg_lon
        yy = (lat_deg - self._lat_ref_deg) * km_per_deg_lat
        ds = np.sqrt(
            np.diff(r) ** 2
            + (0.5 * (r[:-1] + r[1:]) * np.diff(lat)) ** 2
            + (
                0.5
                * (r[:-1] + r[1:])
                * np.cos(0.5 * (lat[:-1] + lat[1:]))
                * np.diff(lon)
            )
            ** 2
        )
        group_path_km = float(np.nansum(ds))
        alt_mid = 0.5 * (z[:-1] + z[1:])
        lat_mid = 0.5 * (lat_deg[:-1] + lat_deg[1:])
        lon_mid = 0.5 * (lon_deg[:-1] + lon_deg[1:])
        mup = np.asarray(self._eval_mup_sph(alt_mid, lat_mid, lon_mid), dtype=float)
        valid = np.isfinite(mup)
        group_delay_sec = float(np.nansum((mup[valid] / C_KM_S) * ds[valid]))
        status = "length"
        if sol.status == 1:
            status = "ground" if len(sol.t_events[0]) > 0 else "domain"
        elif sol.status == -1:
            status = "failure"
        return SimpleNamespace(
            x_km=x,
            y_km=yy,
            z_km=z,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            r_km=r,
            vr=sol.y[3, :],
            vlat=sol.y[4, :],
            vlon=sol.y[5, :],
            t=sol.t,
            status=status,
            reason=status,
            group_path_km=group_path_km,
            group_delay_sec=group_delay_sec,
            mode=mode,
            coordinate_system="spherical",
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            azimuth_deg=float(azimuth_deg),
        )

    def oblique_trace(
        self,
        freq_hz: float,
        elevation_deg: float,
        *,
        coordinate_system: str = "cartesian",
        solver: str = "gradient",
        nhops: int = 1,
        **kwargs,
    ) -> SimpleNamespace:
        """Unified 3-D oblique ray-trace entry point.

        Parameters
        ----------
        coordinate_system : ``"cartesian"`` or ``"spherical"``.
        solver : ``"gradient"`` (default) or ``"hamiltonian"`` (cartesian only).
        nhops : int, optional
            Number of ionospheric hops (default 1).  For nhops > 1 each
            ground hit is reflected specularly (vz → −vz for cartesian,
            vr → −vr for spherical) and the ODE restarts from the domain
            left edge with x/y shifted by the accumulated ground-hit
            offset (horizontal-homogeneity assumption).  All hop segments
            are concatenated in the returned namespace.
        """
        nhops = max(1, int(nhops))
        coord = str(coordinate_system).strip().lower()
        solv  = str(solver).strip().lower()

        # ── select tracer ──────────────────────────────────────────────────
        if coord in {"cartesian", "cart", "xyz"}:
            _tracer  = (self.trace_cartesian_hamiltonian
                        if solv in {"hamiltonian", "ham"}
                        else self.trace_cartesian_gradient)
            is_sph   = False
        elif coord in {"spherical", "sph", "rll"}:
            if solv in {"hamiltonian", "ham"}:
                logger.warning(
                    "Hamiltonian spherical solver not implemented; using gradient."
                )
            _tracer  = self.trace_spherical_gradient
            is_sph   = True
        else:
            raise ValueError("coordinate_system must be 'cartesian' or 'spherical'")

        # ── single-hop fast path ───────────────────────────────────────────
        if nhops == 1:
            out = _tracer(freq_hz=freq_hz, elevation_deg=elevation_deg, **kwargs)
            out.nhops_completed = 1
            return out

        # ── multi-hop path ─────────────────────────────────────────────────
        # Extract first-hop origin (consumed here; subsequent hops restart
        # at domain origin with coordinate shift applied to output).
        x0 = float(kwargs.pop("x0_km", 0.0))
        y0 = float(kwargs.pop("y0_km", 0.0))
        z0 = float(kwargs.pop("z0_km", float(self.alts_km[0])))

        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        total_gpath  = 0.0
        total_gdelay = 0.0
        z_apex_best  = -np.inf
        last: SimpleNamespace | None = None
        hops_done = 0
        elev = float(elevation_deg)
        az   = float(kwargs.get("azimuth_deg", 0.0))

        # Accumulated physical offset for subsequent hops
        x_accum, y_accum = x0, y0

        for _hop in range(nhops):
            x_ode = 0.0 if _hop > 0 else x0
            y_ode = 0.0 if _hop > 0 else y0
            x_sft = x_accum if _hop > 0 else 0.0
            y_sft = y_accum if _hop > 0 else 0.0

            ray = _tracer(
                freq_hz=freq_hz,
                elevation_deg=elev,
                x0_km=x_ode,
                y0_km=y_ode,
                z0_km=z0,
                **kwargs,
            )
            x_arr = np.asarray(ray.x_km, dtype=float) + x_sft
            y_arr = np.asarray(ray.y_km, dtype=float) + y_sft
            z_arr = np.asarray(ray.z_km, dtype=float)
            all_x.append(x_arr)
            all_y.append(y_arr)
            all_z.append(z_arr)
            total_gpath  += float(ray.group_path_km)
            total_gdelay += float(getattr(ray, "group_delay_sec", 0.0))
            if z_arr.size:
                z_apex_best = max(z_apex_best, float(np.nanmax(z_arr)))
            last = ray
            hops_done += 1

            if ray.status != "ground" or x_arr.size == 0:
                break  # did not reach ground — no further hops

            # ── specular ground reflection ─────────────────────────────────
            if is_sph:
                # state [r, lat, lon, vr, vlat, vlon]; vr²+vlat²+vlon²=1
                vr_last   = float(ray.vr[-1])    # < 0 at descent
                vlat_last = float(ray.vlat[-1])
                vlon_last = float(ray.vlon[-1])
                elev = float(np.degrees(
                    np.arctan2(abs(vr_last),
                               np.sqrt(vlat_last**2 + vlon_last**2))
                ))
                az   = float(np.degrees(np.arctan2(vlon_last, vlat_last)))
            else:
                # state [x, y, z, vx, vy, vz]; vx²+vy²+vz²=1
                vx_last = float(ray.vx[-1])
                vy_last = float(ray.vy[-1])
                vz_last = float(ray.vz[-1])    # < 0 at descent
                elev = float(np.degrees(
                    np.arctan2(abs(vz_last),
                               np.sqrt(vx_last**2 + vy_last**2))
                ))
                az   = float(np.degrees(np.arctan2(vy_last, vx_last)))

            # Update azimuth in kwargs so the next tracer call uses it
            kwargs["azimuth_deg"] = az

            # Physical ground-hit position becomes the shift for the next hop
            x_accum = float(x_arr[-1])
            y_accum = float(y_arr[-1])
            z0 = float(self.alts_km[0])   # restart at ground level

        # ── concatenate all hop segments ───────────────────────────────────
        x_cat = np.concatenate(all_x) if all_x else np.array([], dtype=float)
        y_cat = np.concatenate(all_y) if all_y else np.array([], dtype=float)
        z_cat = np.concatenate(all_z) if all_z else np.array([], dtype=float)
        final_status = last.status if last is not None else "failure"

        out = SimpleNamespace(
            x_km=x_cat,
            y_km=y_cat,
            z_km=z_cat,
            status=final_status,
            reason=final_status,
            group_path_km=total_gpath,
            group_delay_sec=total_gdelay,
            z_apex_km=float(z_apex_best) if z_apex_best > -np.inf else np.nan,
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            azimuth_deg=float(kwargs.get("azimuth_deg", az)),
            mode=getattr(last, "mode", None),
            coordinate_system=coord,
            solver=getattr(last, "solver", "gradient"),
            nhops_completed=hops_done,
        )
        # Carry terminal velocity components from the final hop
        if last is not None:
            if is_sph:
                out.vr   = last.vr
                out.vlat = last.vlat
                out.vlon = last.vlon
            else:
                out.vx = last.vx
                out.vy = last.vy
                out.vz = last.vz
        return out
