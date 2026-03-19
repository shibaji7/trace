"""2D HF ray tools for profile assembly and oblique ray integration."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from loguru import logger
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from hfpytrace.collision import NRLMSISE2D
from hfpytrace.density.iri import IRI2d
from hfpytrace.geomag import build_geomag_grid
from hfpytrace.model.dispersion import AppletonHartreeDispersion, SenWyllerDispersion
from hfpytrace.model.rt1d import RT1DProfile
from hfpytrace.utils import build_heights_from_cfg, build_route_from_cfg

C_KM_S = 299792.458


@dataclass
class RT2DProfile:
    """
    Container for a 2D ionospheric slice sampled on altitude and path axes.

    Conventions:
    - vertical axis: ``alt_km`` (length ``nz``)
    - route axis: ``x_km`` / ``lats`` / ``lons`` (length ``nx``)
    - any 2D physical field uses shape ``(nz, nx)``
    """

    alt_km: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    time: dt.datetime
    x_km: np.ndarray | None = None
    ne_m3: np.ndarray | None = None
    ne_cm3: np.ndarray | None = None
    source: str = "iri"
    msise: SimpleNamespace | None = None
    geomag: SimpleNamespace | None = None
    collision: object | None = None  # ComputeCollision instance after compute_collision()

    def __post_init__(self) -> None:
        """Normalize input arrays and run structural validation."""
        self.alt_km = np.asarray(self.alt_km, dtype=float).ravel()
        self.lats = np.asarray(self.lats, dtype=float).ravel()
        self.lons = np.asarray(self.lons, dtype=float).ravel()
        if not isinstance(self.time, dt.datetime):
            self.time = dt.datetime.fromisoformat(str(self.time))
        if self.x_km is None:
            self.x_km = np.arange(self.lats.size, dtype=float)
        else:
            self.x_km = np.asarray(self.x_km, dtype=float).ravel()
        self.validate()

    def validate(self) -> None:
        """Validate grid monotonicity and attached field shapes."""
        if self.alt_km.size < 2:
            raise ValueError("alt_km must contain at least 2 points")
        if self.lats.size < 2 or self.lons.size < 2:
            raise ValueError("lats and lons must contain at least 2 points")
        if self.lats.shape != self.lons.shape:
            raise ValueError("lats and lons must have the same shape")
        if self.x_km.shape != self.lats.shape:
            raise ValueError("x_km must match lats/lons shape")
        if not np.all(np.diff(self.alt_km) > 0):
            raise ValueError("alt_km must be strictly increasing")
        if not np.all(np.diff(self.x_km) > 0):
            raise ValueError("x_km must be strictly increasing")

        nz = self.alt_km.size
        nx = self.lats.size
        shape = (nz, nx)

        if self.ne_m3 is not None:
            ne = np.asarray(self.ne_m3, dtype=float)
            if ne.shape != shape:
                raise ValueError("ne_m3 must have shape (len(alt_km), len(lats))")
            if np.any(ne < 0):
                raise ValueError("ne_m3 must be non-negative")
            self.ne_m3 = ne
            self.ne_cm3 = ne * 1e-6
        elif self.ne_cm3 is not None:
            ne = np.asarray(self.ne_cm3, dtype=float)
            if ne.shape != shape:
                raise ValueError("ne_cm3 must have shape (len(alt_km), len(lats))")
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

    @classmethod
    def from_cfg(
        cls,
        cfg,
        time: dt.datetime | None = None,
        lats: np.ndarray | None = None,
        lons: np.ndarray | None = None,
        alt_km: np.ndarray | None = None,
        fetch_iri: bool = True,
        fetch_msise: bool = False,
        fetch_geomag: bool = False,
        workers: int = 1,
    ) -> "RT2DProfile":
        """Build an RT2DProfile from route/height settings in config."""
        logger.info("RT2DProfile.from_cfg: build route profile from config")
        t = time if time is not None else dt.datetime.fromisoformat(str(cfg.event))

        if lats is None or lons is None:
            n_range = int(getattr(cfg, "number_of_ground_step_km", 200))
            lats, lons, _, route_km = build_route_from_cfg(cfg, n_range)
            x_km = np.linspace(0.0, float(route_km), int(n_range))
        else:
            lats = np.asarray(lats, dtype=float).ravel()
            lons = np.asarray(lons, dtype=float).ravel()
            if lats.shape != lons.shape:
                raise ValueError("lats and lons must have same shape")
            x_km = np.arange(lats.size, dtype=float)

        if alt_km is None:
            alt_km = build_heights_from_cfg(cfg)
        else:
            alt_km = np.asarray(alt_km, dtype=float).ravel()

        prof = cls(alt_km=alt_km, lats=lats, lons=lons, x_km=x_km, time=t)
        if fetch_iri:
            prof.fetch_iri(cfg, workers=workers)
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
        """Attach user-supplied electron density and keep unit pairs synced."""
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
        """
        Force electron density to zero below a specified altitude threshold.

        Parameters
        ----------
        min_alt_km:
            Altitude floor in km. Any row with ``alt_km < min_alt_km`` is zeroed.

        Returns
        -------
        int
            Number of altitude rows modified.
        """
        if self.ne_m3 is None or self.ne_cm3 is None:
            raise ValueError(
                "Electron density is not initialized; call fetch_iri() or set_electron_density() first."
            )
        below = np.asarray(self.alt_km, dtype=float) < float(min_alt_km)
        n_rows = int(np.count_nonzero(below))
        if n_rows == 0:
            return 0
        self.ne_m3[below, :] = 0.0
        self.ne_cm3[below, :] = 0.0
        self.validate()
        return n_rows

    def fetch_iri(self, cfg, workers: int = 1) -> np.ndarray:
        """Populate electron density from IRI2d on the profile grid."""
        logger.info(
            "Fetching 2D IRI profile: path_points={}, alt_points={}",
            self.lats.size,
            self.alt_km.size,
        )
        model = IRI2d(cfg, self.time)
        ne_cm3, _ = model.fetch_dataset(
            self.time,
            lats=self.lats,
            lons=self.lons,
            alts=self.alt_km,
            workers=int(workers),
        )
        self.ne_cm3 = np.asarray(ne_cm3, dtype=float)
        self.ne_m3 = self.ne_cm3 * 1e6
        self.source = "iri"
        self.validate()
        logger.info("2D IRI profile fetched: shape={}", self.ne_m3.shape)
        return self.ne_m3

    def fetch_msise(
        self,
        workers: int = 1,
        update_spaceweather: bool = False,
        suppress_spaceweather_warning: bool = True,
    ) -> SimpleNamespace:
        """Populate neutral species and neutral temperature from NRLMSISE."""
        logger.info(
            "Fetching 2D NRLMSISE profile: path_points={}, alt_points={}, workers={}",
            self.lats.size,
            self.alt_km.size,
            int(workers),
        )
        ms = NRLMSISE2D(
            date=self.time,
            lats=self.lats,
            lons=self.lons,
            heights_km=self.alt_km,
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
        logger.info("2D NRLMSISE profile fetched.")
        return self.msise

    def fetch_geomag(
        self,
        coord_input: str = "GEO",
        coeff_dir: str | None = None,
    ) -> SimpleNamespace:
        """Populate geomagnetic vectors and angles along the 2D route grid."""
        logger.info(
            "Fetching 2D geomag profile on route: path_points={}, alt_points={}",
            self.lats.size,
            self.alt_km.size,
        )
        nx = self.lats.size
        nz = self.alt_km.size
        Bx = np.zeros((nz, nx), dtype=float)
        By = np.zeros((nz, nx), dtype=float)
        Bz = np.zeros((nz, nx), dtype=float)
        bmag_t = np.zeros((nz, nx), dtype=float)
        inc_deg = np.zeros((nz, nx), dtype=float)
        dec_deg = np.zeros((nz, nx), dtype=float)
        psi_deg = np.zeros((nz, nx), dtype=float)

        cdir = None
        if isinstance(coeff_dir, str) and coeff_dir.strip():
            cdir = coeff_dir

        for i in range(nx):
            gm = build_geomag_grid(
                lats=np.array([self.lats[i]], dtype=float),
                lons=np.array([self.lons[i]], dtype=float),
                alts_km=self.alt_km,
                time=self.time,
                coord_input=coord_input,
                coeff_dir=cdir,
            )
            Bx[:, i] = gm.Bx[0, 0, :]
            By[:, i] = gm.By[0, 0, :]
            Bz[:, i] = gm.Bz[0, 0, :]
            bmag_t[:, i] = gm.bmag_t[0, 0, :]
            inc_deg[:, i] = gm.inc_deg[0, 0, :]
            dec_deg[:, i] = gm.dec_deg[0, 0, :]
            psi_deg[:, i] = gm.psi_deg[0, 0, :]

        self.geomag = SimpleNamespace(
            Bx=Bx,
            By=By,
            Bz=Bz,
            bmag_t=bmag_t,
            inc_deg=inc_deg,
            dec_deg=dec_deg,
            psi_deg=psi_deg,
        )
        self.validate()
        logger.info("2D geomag profile fetched.")
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
            Electron/ion temperature [K], shape (nz, nx) or broadcastable.
            Defaults to MSIS neutral temperature Tn.
        edens : array-like, optional
            Electron density [cm^-3], shape (nz, nx). Defaults to ``self.ne_cm3``.
        O2p, Op : array-like, optional
            O2+ and O+ densities [cm^-3]. Defaults to 10%/90% of edens.

        Returns
        -------
        ComputeCollision
            Also stored on ``self.collision`` for retrieval via
            ``collision_type`` in :meth:`RT2D.build_refractive_index_interpolators`.

        Notes
        -----
        Supported ``collision_type`` keys for :meth:`RT2D.oblique_trace`:

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

        Tn = np.asarray(self.msise.Tn, dtype=float)   # shape (nz, nx)
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
            "2D collision computed: nu_ft=[{:.3e},{:.3e}] Hz, nu_sn=[{:.3e},{:.3e}] Hz",
            float(np.nanmin(cc.collision.nu_ft)),
            float(np.nanmax(cc.collision.nu_ft)),
            float(np.nanmin(cc.collision.nu_sn.total)),
            float(np.nanmax(cc.collision.nu_sn.total)),
        )
        return cc


@dataclass
class RT2DConfig:
    """Integration controls for :class:`RT2D`."""

    x0_km: float = 0.0
    z0_km: float = 0.0
    ds_km: float = 1.0
    max_steps: int = 8000
    x_min_km: float | None = None
    x_max_km: float | None = None
    z_min_km: float = 0.0
    z_max_km: float | None = None
    n2_floor: float = 1e-5
    keep_every: int = 1


class RT2D:
    """
    Main 2D tracing interface for profile-based and legacy workflows.

    Accepted initialization patterns:
    1) direct grids: ``RT2D(x_km, z_km, ne_m3)``
    2) profile object: ``RT2D(profile=RT2DProfile(...))``
    3) config-driven: ``RT2D(cfg=..., fetch_iri=True, ...)``
    """

    def __init__(
        self,
        x_km: np.ndarray | None = None,
        z_km: np.ndarray | None = None,
        ne_m3: np.ndarray | None = None,
        *,
        profile: RT2DProfile | None = None,
        cfg=None,
        time: dt.datetime | str | None = None,
        lats: np.ndarray | None = None,
        lons: np.ndarray | None = None,
        alt_km: np.ndarray | None = None,
        ne_cm3: np.ndarray | None = None,
        source: str = "iri",
        fetch_iri: bool = False,
        fetch_msise: bool = False,
        fetch_geomag: bool = False,
        workers: int = 1,
    ):
        """Create an RT2D solver instance and prepare interpolation kernels."""
        self.profile: RT2DProfile | None = None

        if profile is not None:
            if not isinstance(profile, RT2DProfile):
                raise TypeError("profile must be an RT2DProfile")
            profile.validate()
            self.profile = profile
            if self.profile.ne_m3 is None:
                raise ValueError("RT2DProfile must include ne_m3 to initialize RT2D")
            self.x_km = np.asarray(self.profile.x_km, dtype=float)
            self.z_km = np.asarray(self.profile.alt_km, dtype=float)
            self.ne_m3 = np.asarray(self.profile.ne_m3, dtype=float)
        elif x_km is not None and z_km is not None and ne_m3 is not None:
            # Legacy path
            self.x_km = np.asarray(x_km, dtype=float).ravel()
            self.z_km = np.asarray(z_km, dtype=float).ravel()
            self.ne_m3 = np.asarray(ne_m3, dtype=float)
        else:
            # Config/explicit profile assembly path
            if cfg is None:
                raise ValueError(
                    "Provide profile, or legacy arrays (x_km/z_km/ne_m3), or cfg-based inputs."
                )
            t = time if time is not None else dt.datetime.fromisoformat(str(cfg.event))
            self.profile = RT2DProfile.from_cfg(
                cfg=cfg,
                time=t,
                lats=lats,
                lons=lons,
                alt_km=alt_km,
                fetch_iri=bool(fetch_iri),
                fetch_msise=bool(fetch_msise),
                fetch_geomag=bool(fetch_geomag),
                workers=int(workers),
            )
            if (ne_m3 is not None) or (ne_cm3 is not None):
                self.profile.set_electron_density(
                    ne_m3=ne_m3,
                    ne_cm3=ne_cm3,
                    source=source,
                )
            if self.profile.ne_m3 is None:
                raise ValueError(
                    "No electron density available. Set ne_m3/ne_cm3 or fetch_iri=True."
                )
            self.x_km = np.asarray(self.profile.x_km, dtype=float)
            self.z_km = np.asarray(self.profile.alt_km, dtype=float)
            self.ne_m3 = np.asarray(self.profile.ne_m3, dtype=float)

        if self.ne_m3.shape != (self.z_km.size, self.x_km.size):
            raise ValueError("ne_m3 shape must be (len(z_km), len(x_km))")
        if not np.all(np.diff(self.x_km) > 0) or not np.all(np.diff(self.z_km) > 0):
            raise ValueError("x_km and z_km must be strictly increasing")

        self._ne_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            self.ne_m3,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._dx = float(np.mean(np.diff(self.x_km)))
        self._dz = float(np.mean(np.diff(self.z_km)))
        self._n_interp = None
        self._dn_dx_interp = None
        self._dn_dz_interp = None
        self._mup_interp = None
        logger.info(
            "RT2D initialized: nx={}, nz={}, ne_shape={}",
            self.x_km.size,
            self.z_km.size,
            self.ne_m3.shape,
        )

    _VALID_COLLISION_TYPES: frozenset[str] = frozenset(
        {"FT", "FT_CC", "FT_MB", "SN_EN", "SN_EI", "SN", "ATM"}
    )

    @staticmethod
    def _extract_collision_hz(cc, collision_type: str) -> np.ndarray:
        """
        Extract a collision-frequency array from a ``ComputeCollision`` object.

        The returned array has the same shape as the profile fields (nz, nx)
        and can be passed directly to the dispersion models.

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

        Convenience wrapper around :meth:`RT2DProfile.compute_collision`.
        Requires ``fetch_msise()`` to have been called on the profile first.

        After calling this, pass ``collision_type`` to
        :meth:`oblique_trace` to select which model to use, e.g.::

            rt = RT2D(profile=prof)
            rt.fetch_collision()
            ray = rt.oblique_trace(freq_hz=10.5e6, elevation_deg=25,
                                   collision_type="SN")

        Returns
        -------
        ComputeCollision
        """
        if self.profile is None:
            raise ValueError("RT2D has no attached profile. Cannot compute collision.")
        return self.profile.compute_collision(
            Te=Te, Ti=Ti, edens=edens, O2p=O2p, Op=Op,
        )

    def _inside(self, x: float, z: float, cfg: RT2DConfig) -> bool:
        """Check whether a point lies inside active tracing bounds."""
        x_min = self.x_km[0] if cfg.x_min_km is None else cfg.x_min_km
        x_max = self.x_km[-1] if cfg.x_max_km is None else cfg.x_max_km
        z_max = self.z_km[-1] if cfg.z_max_km is None else cfg.z_max_km
        return (x_min <= x <= x_max) and (cfg.z_min_km <= z <= z_max)

    def _sample_ne(self, x: float, z: float) -> float:
        """Sample electron density at a single Cartesian point."""
        return float(self._ne_interp(np.array([[z, x]], dtype=float))[0])

    def _n_and_grad_fd(
        self, freq_hz: float, x: float, z: float
    ) -> tuple[float, float, float]:
        """Finite-difference estimate of refractive index and spatial gradients."""
        ne = self._sample_ne(x, z)
        fp = RT1DProfile.den_to_plasma_freq_hz(np.maximum(ne, 0.0))
        n2 = max(1.0 - (fp / float(freq_hz)) ** 2, 1e-12)
        n = float(np.sqrt(n2))

        hx = max(self._dx, 1e-3)
        hz = max(self._dz, 1e-3)
        ne_xp = self._sample_ne(x + hx, z)
        ne_xm = self._sample_ne(x - hx, z)
        ne_zp = self._sample_ne(x, z + hz)
        ne_zm = self._sample_ne(x, z - hz)

        n2_xp = max(
            1.0
            - (RT1DProfile.den_to_plasma_freq_hz(np.maximum(ne_xp, 0.0)) / freq_hz)
            ** 2,
            1e-12,
        )
        n2_xm = max(
            1.0
            - (RT1DProfile.den_to_plasma_freq_hz(np.maximum(ne_xm, 0.0)) / freq_hz)
            ** 2,
            1e-12,
        )
        n2_zp = max(
            1.0
            - (RT1DProfile.den_to_plasma_freq_hz(np.maximum(ne_zp, 0.0)) / freq_hz)
            ** 2,
            1e-12,
        )
        n2_zm = max(
            1.0
            - (RT1DProfile.den_to_plasma_freq_hz(np.maximum(ne_zm, 0.0)) / freq_hz)
            ** 2,
            1e-12,
        )

        if n <= 0:
            return n, 0.0, 0.0
        dn_dx = 0.25 * (n2_xp - n2_xm) / (hx * n)
        dn_dz = 0.25 * (n2_zp - n2_zm) / (hz * n)
        return n, float(dn_dx), float(dn_dz)

    @staticmethod
    def _resolve_dispersion_model_name(formulation: str) -> str:
        """
        Map external model labels to internal canonical names.

        Canonical names:
        - ``appleton-hartree``
        - ``sen-wyller``
        """
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

    def build_refractive_index_interpolators(
        self,
        freq_hz: float,
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        mode: str = "O",
        formulation: str = "appleton",
        collision_hz: np.ndarray | float | None = None,
        collision_type: str | None = None,
    ) -> SimpleNamespace:
        """
        Build refractive-index and gradient interpolators on the RT2D grid.

        The returned namespace includes:
        - ``n``: phase refractive index
        - ``mup``: group refractive index proxy (1/n)
        - ``dn_dx``, ``dn_dz``: Cartesian gradients of ``n``

        Parameters
        ----------
        collision_hz : array-like or float, optional
            Collision frequency [Hz], shape (nz, nx) or scalar. Mutually
            exclusive with ``collision_type``.
        collision_type : str, optional
            Named collision model key. Requires ``fetch_collision()`` to have
            been called first. Mutually exclusive with ``collision_hz``.
            Valid values: ``"FT"``, ``"FT_cc"``, ``"FT_mb"``, ``"SN_en"``,
            ``"SN_ei"``, ``"SN"``, ``"atm"`` (case-insensitive).
        """
        if collision_type is not None and collision_hz is not None:
            raise ValueError(
                "Provide at most one of 'collision_hz' or 'collision_type', not both."
            )
        if collision_type is not None:
            if self.profile is None or self.profile.collision is None:
                raise ValueError(
                    f"collision_type='{collision_type}' requires pre-computed collision. "
                    "Call rt.fetch_collision() first."
                )
            collision_hz = self._extract_collision_hz(
                self.profile.collision, collision_type
            )
            logger.info(
                "RT2D interpolators: using collision_type='{}', nu=[{:.3e},{:.3e}] Hz",
                collision_type,
                float(np.nanmin(collision_hz)),
                float(np.nanmax(collision_hz)),
            )

        model_name = self._resolve_dispersion_model_name(formulation)
        logger.info(
            "Building RT2D refractive index interpolators: freq={} Hz, mode={}, model={}, collision={}",
            float(freq_hz),
            mode,
            model_name,
            collision_type if collision_type is not None else ("custom" if collision_hz is not None else "none"),
        )
        if b_abs_t is None:
            b_abs_t_arr = np.zeros_like(self.ne_m3)
        else:
            b_abs_t_arr = np.asarray(b_abs_t, dtype=float)
            if b_abs_t_arr.ndim == 0:
                b_abs_t_arr = np.full_like(self.ne_m3, float(b_abs_t_arr))
        if b_psi_deg is None:
            b_psi_arr = np.zeros_like(self.ne_m3)
        else:
            b_psi_arr = np.asarray(b_psi_deg, dtype=float)
            if b_psi_arr.ndim == 0:
                b_psi_arr = np.full_like(self.ne_m3, float(b_psi_arr))

        nu_arr = (
            np.zeros_like(self.ne_m3)
            if collision_hz is None
            else (
                np.full_like(self.ne_m3, float(collision_hz))
                if np.asarray(collision_hz).ndim == 0
                else np.asarray(collision_hz, dtype=float)
            )
        )

        if model_name == "appleton-hartree":
            model = AppletonHartreeDispersion(
                frequency_hz=float(freq_hz),
                ne_m3=self.ne_m3,
                collision_hz=nu_arr,
                b_t=b_abs_t_arr,
                theta_deg=b_psi_arr,
            )
        elif model_name == "sen-wyller":
            model = SenWyllerDispersion(
                frequency_hz=float(freq_hz),
                ne_m3=self.ne_m3,
                collision_hz=nu_arr,
                b_t=b_abs_t_arr,
                theta_deg=b_psi_arr,
            )

        n_complex = model.refractive_index(mode=mode)
        n = np.real(n_complex)
        n = np.where(np.isfinite(n), np.clip(n, 0.0, None), np.nan)
        mup = 1.0 / np.clip(n, 1e-8, None)
        dn_dz, dn_dx = np.gradient(n, self.z_km, self.x_km)

        self._n_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            n,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._dn_dx_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            dn_dx,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._dn_dz_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            dn_dz,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        self._mup_interp = RegularGridInterpolator(
            (self.z_km, self.x_km),
            mup,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        logger.debug("RT2D refractive index interpolators ready")
        return SimpleNamespace(n=n, mup=mup, dn_dx=dn_dx, dn_dz=dn_dz)

    def _eval_n_grad(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate phase index and gradients at Cartesian sample points."""
        if self._n_interp is None:
            raise RuntimeError("Call build_refractive_index_interpolators() first")
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        x_arr, z_arr = np.broadcast_arrays(x_arr, z_arr)
        pts = np.column_stack([z_arr.ravel(), x_arr.ravel()])
        n = self._n_interp(pts).reshape(x_arr.shape)
        dnx = self._dn_dx_interp(pts).reshape(x_arr.shape)
        dnz = self._dn_dz_interp(pts).reshape(x_arr.shape)
        return n, dnx, dnz

    def _eval_mup(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate group-index proxy at Cartesian sample points."""
        if self._mup_interp is None:
            raise RuntimeError("Call build_refractive_index_interpolators() first")
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        x_arr, z_arr = np.broadcast_arrays(x_arr, z_arr)
        pts = np.column_stack([z_arr.ravel(), x_arr.ravel()])
        return self._mup_interp(pts).reshape(x_arr.shape)

    def _eval_n_grad_rphi(
        self,
        phi: np.ndarray,
        r: np.ndarray,
        r_earth_km: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate n, dn/dr, dn/dphi by mapping spherical points to x-z grid."""
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        phi_arr, r_arr = np.broadcast_arrays(phi_arr, r_arr)
        x = phi_arr * float(r_earth_km)
        z = r_arr - float(r_earth_km)
        n, dn_dx, dn_dz = self._eval_n_grad(x=x, z=z)
        dn_dr = dn_dz
        dn_dphi = dn_dx * float(r_earth_km)
        return n, dn_dr, dn_dphi

    @staticmethod
    def _ray_rhs(_s: float, y: np.ndarray, n_grad_fn) -> np.ndarray:
        """RHS for Cartesian ray ODE integrated over path length."""
        x, z, vx, vz = y
        n, dnx, dnz = n_grad_fn(np.array([x]), np.array([z]))
        n = float(n[0])
        dnx = float(dnx[0])
        dnz = float(dnz[0])
        if not np.isfinite(n) or n <= 0:
            return np.zeros(4)
        dot = dnx * vx + dnz * vz
        dvx = (dnx - dot * vx) / n
        dvz = (dnz - dot * vz) / n
        return np.array([vx, vz, dvx, dvz], dtype=float)

    @staticmethod
    def _ray_rhs_spherical(_s: float, y: np.ndarray, n_grad_rphi_fn) -> np.ndarray:
        """RHS for spherical (r, phi) ray ODE integrated over path length."""
        r, phi, v_r, v_phi = y
        n, dn_dr, dn_dphi = n_grad_rphi_fn(np.array([phi]), np.array([r]))
        n = float(n[0])
        dn_dr = float(dn_dr[0])
        dn_dphi = float(dn_dphi[0])
        if not np.isfinite(n) or n <= 0.0 or r <= 0.0:
            return np.zeros(4, dtype=float)
        grad_dot_v = dn_dr * v_r + (dn_dphi / r) * v_phi
        drds = v_r
        dphids = v_phi / r
        dv_r = (dn_dr - grad_dot_v * v_r) / n + (v_phi * v_phi) / r
        dv_phi = ((dn_dphi / r) - grad_dot_v * v_phi) / n - (v_r * v_phi) / r
        return np.array([drds, dphids, dv_r, dv_phi], dtype=float)

    def trace_cartesian_gradient(
        self,
        freq_hz: float,
        elevation_deg: float,
        x0_km: float = 0.0,
        z0_km: float = 0.0,
        s_max_km: float = 5000.0,
        mode: str = "O",
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        formulation: str = "appleton",
        collision_hz: np.ndarray | float | None = None,
        collision_type: str | None = None,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_km: float | None = None,
    ) -> SimpleNamespace:
        """
        Integrate one oblique ray in Cartesian coordinates using gradient ODEs.

        Returns path coordinates, ray direction history, and derived path metrics.

        Parameters
        ----------
        collision_hz : array-like or float, optional
            Collision frequency [Hz], shape (nz, nx) or scalar. Mutually
            exclusive with ``collision_type``.
        collision_type : str, optional
            Named collision model. Requires ``fetch_collision()`` first.
            Valid values: ``"FT"``, ``"FT_cc"``, ``"FT_mb"``, ``"SN_en"``,
            ``"SN_ei"``, ``"SN"``, ``"atm"`` (case-insensitive).
        """
        logger.info(
            "RT2D gradient trace start: freq={} Hz, elev={} deg, mode={}, collision={}",
            float(freq_hz),
            float(elevation_deg),
            mode,
            collision_type if collision_type is not None else ("custom" if collision_hz is not None else "none"),
        )
        self.build_refractive_index_interpolators(
            freq_hz=freq_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            mode=mode,
            formulation=formulation,
            collision_hz=collision_hz,
            collision_type=collision_type,
        )

        elev = np.deg2rad(float(elevation_deg))
        y0 = np.array(
            [float(x0_km), float(z0_km), np.cos(elev), np.sin(elev)], dtype=float
        )
        y0[2:] /= max(np.linalg.norm(y0[2:]), 1e-12)

        z_ground = float(self.z_km[0])
        z_top = float(self.z_km[-1])
        x_left = float(self.x_km[0])
        x_right = float(self.x_km[-1])

        def ev_ground(_s, y):
            return y[1] - z_ground - 1e-3

        def ev_top(_s, y):
            return z_top - y[1]

        def ev_left(_s, y):
            return y[0] - x_left

        def ev_right(_s, y):
            return x_right - y[0]

        for ev in (ev_ground, ev_top, ev_left, ev_right):
            ev.terminal = True
            ev.direction = -1.0

        sol = solve_ivp(
            lambda s, y: self._ray_rhs(s, y, self._eval_n_grad),
            (0.0, float(s_max_km)),
            y0,
            method="RK45",
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step_km) if max_step_km is not None else np.inf,
            events=[ev_ground, ev_top, ev_left, ev_right],
        )
        x = sol.y[0, :]
        z = sol.y[1, :]
        dx = np.diff(x)
        dz = np.diff(z)
        ds = np.hypot(dx, dz)
        group_path_km = float(np.nansum(ds))
        x_mid = 0.5 * (x[:-1] + x[1:])
        z_mid = 0.5 * (z[:-1] + z[1:])
        mup_mid = np.asarray(self._eval_mup(x_mid, z_mid), dtype=float)
        valid = np.isfinite(mup_mid)
        group_delay_sec = float(np.nansum((mup_mid[valid] / C_KM_S) * ds[valid]))
        status = "length"
        if sol.status == 1:
            status = "ground" if len(sol.t_events[0]) > 0 else "domain"
        elif sol.status == -1:
            status = "failure"
        out = SimpleNamespace(
            x_km=x,
            z_km=z,
            vx=sol.y[2, :],
            vz=sol.y[3, :],
            t=sol.t,
            status=status,
            reason=status,
            group_path_km=group_path_km,
            group_delay_sec=group_delay_sec,
            ground_range_km=float(x[-1]) if status == "ground" else np.nan,
            x_apex_km=float(x[np.nanargmax(z)]) if z.size else np.nan,
            z_apex_km=float(np.nanmax(z)) if z.size else np.nan,
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            mode=mode,
        )
        logger.info("RT2D gradient trace complete: status={}", out.status)
        return out

    def trace_spherical_gradient(
        self,
        freq_hz: float,
        elevation_deg: float,
        x0_km: float = 0.0,
        z0_km: float = 0.0,
        s_max_km: float = 5000.0,
        mode: str = "O",
        b_abs_t: np.ndarray | float | None = None,
        b_psi_deg: np.ndarray | float | None = None,
        formulation: str = "appleton",
        collision_hz: np.ndarray | float | None = None,
        collision_type: str | None = None,
        r_earth_km: float = 6371.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_step_km: float | None = None,
    ) -> SimpleNamespace:
        """
        Integrate one oblique ray in spherical geometry over the same n-field.

        The n-field is sampled from the internal x-z grid; geometry terms are
        handled in (r, phi) coordinates for better long-range behavior.

        Parameters
        ----------
        collision_hz : array-like or float, optional
            Collision frequency [Hz]. Mutually exclusive with ``collision_type``.
        collision_type : str, optional
            Named collision model key. Requires ``fetch_collision()`` first.
        """
        logger.info(
            "RT2D spherical gradient trace start: freq={} Hz, elev={} deg, mode={}, collision={}",
            float(freq_hz),
            float(elevation_deg),
            mode,
            collision_type if collision_type is not None else ("custom" if collision_hz is not None else "none"),
        )
        self.build_refractive_index_interpolators(
            freq_hz=freq_hz,
            b_abs_t=b_abs_t,
            b_psi_deg=b_psi_deg,
            mode=mode,
            formulation=formulation,
            collision_hz=collision_hz,
            collision_type=collision_type,
        )

        elev = np.deg2rad(float(elevation_deg))
        r0 = float(r_earth_km) + float(z0_km)
        phi0 = float(x0_km) / float(r_earth_km)
        y0 = np.array([r0, phi0, np.sin(elev), np.cos(elev)], dtype=float)
        y0[2:] /= max(np.linalg.norm(y0[2:]), 1e-12)

        r_ground = float(r_earth_km) + float(self.z_km[0])
        r_top = float(r_earth_km) + float(self.z_km[-1])
        phi_left = float(self.x_km[0]) / float(r_earth_km)
        phi_right = float(self.x_km[-1]) / float(r_earth_km)

        def ev_ground(_s, y):
            return y[0] - r_ground - 1e-3

        def ev_top(_s, y):
            return r_top - y[0]

        def ev_left(_s, y):
            return y[1] - phi_left

        def ev_right(_s, y):
            return phi_right - y[1]

        for ev in (ev_ground, ev_top, ev_left, ev_right):
            ev.terminal = True
            ev.direction = -1.0

        sol = solve_ivp(
            lambda s, y: self._ray_rhs_spherical(
                s,
                y,
                lambda phi, r: self._eval_n_grad_rphi(
                    phi=phi,
                    r=r,
                    r_earth_km=float(r_earth_km),
                ),
            ),
            (0.0, float(s_max_km)),
            y0,
            method="RK45",
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step_km) if max_step_km is not None else np.inf,
            events=[ev_ground, ev_top, ev_left, ev_right],
        )

        r = sol.y[0, :]
        phi = sol.y[1, :]
        x = float(r_earth_km) * phi
        z = r - float(r_earth_km)

        dr = np.diff(r)
        dphi = np.diff(phi)
        r_mid = 0.5 * (r[:-1] + r[1:])
        ds = np.sqrt(dr * dr + (r_mid * dphi) * (r_mid * dphi))
        group_path_km = float(np.nansum(ds))
        x_mid = 0.5 * (x[:-1] + x[1:])
        z_mid = 0.5 * (z[:-1] + z[1:])
        mup_mid = np.asarray(self._eval_mup(x_mid, z_mid), dtype=float)
        valid = np.isfinite(mup_mid)
        group_delay_sec = float(np.nansum((mup_mid[valid] / C_KM_S) * ds[valid]))

        status = "length"
        if sol.status == 1:
            status = "ground" if len(sol.t_events[0]) > 0 else "domain"
        elif sol.status == -1:
            status = "failure"

        out = SimpleNamespace(
            x_km=x,
            z_km=z,
            r_km=r,
            phi_rad=phi,
            v_r=sol.y[2, :],
            v_phi=sol.y[3, :],
            t=sol.t,
            status=status,
            reason=status,
            group_path_km=group_path_km,
            group_delay_sec=group_delay_sec,
            ground_range_km=float(x[-1]) if status == "ground" else np.nan,
            x_apex_km=float(x[np.nanargmax(z)]) if z.size else np.nan,
            z_apex_km=float(np.nanmax(z)) if z.size else np.nan,
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            mode=mode,
            coordinate_system="spherical",
        )
        logger.info("RT2D spherical gradient trace complete: status={}", out.status)
        return out

    def oblique_trace(
        self,
        freq_hz: float,
        elevation_deg: float,
        *,
        coordinate_system: str = "cartesian",
        nhops: int = 1,
        **kwargs,
    ) -> SimpleNamespace:
        """
        Unified gradient tracer entry point for oblique propagation.

        Parameters
        ----------
        coordinate_system:
            ``"cartesian"`` or ``"spherical"`` (aliases accepted).
        nhops : int, optional
            Number of ionospheric hops to attempt (default 1).  For
            nhops > 1 each time a hop reaches the ground the ray
            direction is reflected (vz → −vz) and a new ODE restarts
            from that ground-hit point.  All segments are concatenated
            in the returned namespace.
        """
        nhops = max(1, int(nhops))
        coord = str(coordinate_system).strip().lower()
        if coord in {"cartesian", "cart", "xy"}:
            _tracer = self.trace_cartesian_gradient
        elif coord in {"spherical", "sph", "rphi"}:
            _tracer = self.trace_spherical_gradient
        else:
            raise ValueError("coordinate_system must be 'cartesian' or 'spherical'")

        # ── single-hop fast path (no overhead) ────────────────────────────
        if nhops == 1:
            out = _tracer(freq_hz=freq_hz, elevation_deg=elevation_deg, **kwargs)
            out.coordinate_system = coord
            out.nhops_completed = 1
            return out

        # ── multi-hop path ─────────────────────────────────────────────────
        z_ground    = float(self.z_km[0])
        r_earth_km  = float(kwargs.get("r_earth_km", 6371.0))
        x_start     = float(kwargs.pop("x0_km", 0.0))
        z_start     = float(kwargs.pop("z0_km", z_ground))
        is_sph      = coord in {"spherical", "sph", "rphi"}

        all_x: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        total_gpath  = 0.0
        total_gdelay = 0.0
        z_apex_best  = -np.inf
        last: SimpleNamespace | None = None
        hops_done = 0
        elev = float(elevation_deg)
        # x_accum: running total of physical x displacement across all hops.
        # For hops 2+ we reset the ODE to x=0 inside the domain (horizontal
        # homogeneity) so the full n-field grid is available.  The output
        # x coordinates are then shifted back to physical positions.
        x_accum = x_start

        for _hop in range(nhops):
            # hops 2+ restart at the domain left edge; x output is shifted
            x_ode_start = 0.0 if _hop > 0 else x_start
            x_shift     = x_accum if _hop > 0 else 0.0

            ray = _tracer(
                freq_hz=freq_hz,
                elevation_deg=elev,
                x0_km=x_ode_start,
                z0_km=z_start,
                **kwargs,
            )
            x_arr = np.asarray(ray.x_km, dtype=float) + x_shift
            z_arr = np.asarray(ray.z_km, dtype=float)
            all_x.append(x_arr)
            all_z.append(z_arr)
            total_gpath  += float(ray.group_path_km)
            total_gdelay += float(getattr(ray, "group_delay_sec", 0.0))
            if z_arr.size:
                z_apex_best = max(z_apex_best, float(np.nanmax(z_arr)))
            last = ray
            hops_done += 1

            if ray.status != "ground" or x_arr.size == 0:
                break  # ray did not reach ground — no further hops

            # ── reflect at ground: compute new elevation from terminal velocity
            if is_sph:
                # Spherical state: [r, phi, v_r, v_phi].
                # Initial condition sets v_r=sin(elev), v_phi=cos(elev) so
                # v_r² + v_phi² = 1 (Euclidean normalisation, not spherical
                # metric).  Therefore elevation = arctan2(v_r, v_phi) — no
                # factor of r needed.
                v_r_last   = float(ray.v_r[-1])    # < 0 at ground impact
                v_phi_last = float(ray.v_phi[-1])
                elev = float(np.degrees(
                    np.arctan2(abs(v_r_last), abs(v_phi_last))
                ))
            else:
                # cartesian state: [x, z, vx, vz]  (vz < 0 at descent)
                vx_last = float(ray.vx[-1])
                vz_last = float(ray.vz[-1])
                elev = float(np.degrees(np.arctan2(abs(vz_last), abs(vx_last))))

            x_accum = float(x_arr[-1])   # physical accumulated x at ground hit
            z_start = z_ground

        x_cat = np.concatenate(all_x) if all_x else np.array([], dtype=float)
        z_cat = np.concatenate(all_z) if all_z else np.array([], dtype=float)
        final_status = last.status if last is not None else "failure"
        _empty = np.array([], dtype=float)

        out = SimpleNamespace(
            x_km=x_cat,
            z_km=z_cat,
            status=final_status,
            reason=final_status,
            group_path_km=total_gpath,
            group_delay_sec=total_gdelay,
            ground_range_km=float(x_cat[-1]) if final_status == "ground" else np.nan,
            z_apex_km=float(z_apex_best) if z_apex_best > -np.inf else np.nan,
            x_apex_km=(
                float(x_cat[int(np.nanargmax(z_cat))]) if z_cat.size else np.nan
            ),
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            mode=getattr(last, "mode", None),
            coordinate_system=coord,
            nhops_completed=hops_done,
        )
        # carry terminal velocity attributes from the final hop
        if last is not None:
            if is_sph:
                out.v_r   = last.v_r
                out.v_phi = last.v_phi
            else:
                out.vx = last.vx
                out.vz = last.vz
        return out

    def trace(
        self,
        freq_hz: float,
        elevation_deg: float,
        cfg: RT2DConfig | None = None,
    ) -> SimpleNamespace:
        """
        Legacy finite-difference tracer retained for backward compatibility.
        """
        logger.info(
            "RT2D FD trace start: freq={} Hz, elev={} deg",
            float(freq_hz),
            float(elevation_deg),
        )
        cfg = cfg or RT2DConfig()
        r = np.array([float(cfg.x0_km), float(cfg.z0_km)], dtype=float)
        el = np.deg2rad(float(elevation_deg))
        t = np.array([np.cos(el), np.sin(el)], dtype=float)

        xs: list[float] = []
        zs: list[float] = []
        ns: list[float] = []
        reason = "max_steps"

        for i in range(int(cfg.max_steps)):
            if not self._inside(r[0], r[1], cfg):
                reason = "out_of_bounds"
                break

            n, gx, gz = self._n_and_grad_fd(float(freq_hz), r[0], r[1])
            if not np.isfinite(n):
                reason = "nan_field"
                break
            if n * n < float(cfg.n2_floor):
                reason = "evanescent"
                break

            if i % max(1, int(cfg.keep_every)) == 0:
                xs.append(float(r[0]))
                zs.append(float(r[1]))
                ns.append(float(n))

            g = np.array([gx, gz], dtype=float)
            a = (g - np.dot(g, t) * t) / max(n, 1e-9)
            t = t + a * float(cfg.ds_km)
            t_norm = np.linalg.norm(t)
            if t_norm <= 0:
                reason = "invalid_direction"
                break
            t /= t_norm
            r = r + t * float(cfg.ds_km)

        out = SimpleNamespace(
            x_km=np.asarray(xs, dtype=float),
            z_km=np.asarray(zs, dtype=float),
            n=np.asarray(ns, dtype=float),
            freq_hz=float(freq_hz),
            elevation_deg=float(elevation_deg),
            reason=reason,
            status=reason,
        )
        logger.info("RT2D FD trace complete: status={}", out.status)
        return out

    def trace_fan(
        self,
        freqs_hz: np.ndarray,
        elevations_deg: np.ndarray,
        cfg: RT2DConfig | None = None,
    ) -> list[SimpleNamespace]:
        """Run a finite-difference fan sweep over frequencies and elevations."""
        out: list[SimpleNamespace] = []
        for f in np.asarray(freqs_hz, dtype=float):
            for el in np.asarray(elevations_deg, dtype=float):
                out.append(
                    self.trace(freq_hz=float(f), elevation_deg=float(el), cfg=cfg)
                )
        return out
