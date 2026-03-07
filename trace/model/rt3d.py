"""3D profile-first scaffolding for future ray tracing workflows."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from loguru import logger

from trace.collision import NRLMSISE3D
from trace.density.iri import IRI3d
from trace.geomag import build_geomag_grid


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


class RT3D:
    """
    Minimal RT3D container for downstream 3D tracing implementations.

    This class currently focuses on profile management and data integrity checks.
    """

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
            self.profile.set_electron_density(
                ne_m3=ne_m3, ne_cm3=ne_cm3, source=source
            )
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

