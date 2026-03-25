"""IRI (International Reference Ionosphere) electron density models.

Provides ``IRI2d`` and ``IRI3d`` wrappers around the PyIRI library for
computing IRI-2016 electron density profiles along 2D great-circle routes
and on 3D lat/lon/altitude grids respectively.

Requires
--------
PyIRI : optional dependency (install with ``pip install PyIRI``).

Classes
-------
IRI2d
    Electron density sampled along a 2D route (lat, lon, alt) profile.
IRI3d
    Electron density on a 3D geographic grid (nlat × nlon × nalt).

Notes
-----
Both classes are driven by a config namespace (``cfg``) that is typically
loaded from ``config*.json`` via ``hfpytrace.utils.load_config_*``.  Key
config fields are ``iri_param.f107``, ``iri_param.foF2_coeff`` (``"CCIR"``
or ``"URSI"``), ``iri_param.hmF2_model``, and ``iri_param.coord``.
"""

import datetime as dt

import numpy as np
from loguru import logger
from scipy.io import loadmat, savemat


def _import_pyiri_sh():
    # PyIRI expects scipy.special.assoc_legendre_p in some versions.
    # Provide a fallback via lpmv when missing.
    try:
        import scipy.special as ss

        if not hasattr(ss, "assoc_legendre_p"):
            logger.warning(
                "scipy.special.assoc_legendre_p missing; enabling lpmv-based compatibility shim for PyIRI."
            )

            def _assoc_legendre_p(n, m, z):
                n_arr = np.atleast_1d(np.asarray(n, dtype=int))
                m_arr = np.atleast_1d(np.asarray(m, dtype=int))
                z_arr = np.asarray(z, dtype=float)
                out = np.empty((m_arr.size, n_arr.size) + z_arr.shape, dtype=float)
                for i, mi in enumerate(m_arr):
                    for j, nj in enumerate(n_arr):
                        out[i, j, ...] = ss.lpmv(int(mi), int(nj), z_arr)
                return out

            ss.assoc_legendre_p = _assoc_legendre_p
    except Exception:
        pass

    try:
        from PyIRI import sh_library as sh
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyIRI is required for hfpytrace.density.iri. Install `PyIRI` and retry."
        ) from exc
    return sh


def _dens_to_points_by_alt(den_raw, npts: int, nalt: int) -> np.ndarray | None:
    den = np.asarray(den_raw, dtype=float)
    if den.size == 0:
        return None
    s = np.squeeze(den)
    if s.ndim == 1:
        if s.size == nalt and npts == 1:
            return s.reshape(1, nalt)
        if s.size == npts and nalt == 1:
            return s.reshape(npts, 1)
        return None
    if s.ndim == 2:
        if s.shape == (npts, nalt):
            return s
        if s.shape == (nalt, npts):
            return s.T
        return None
    if s.size == (npts * nalt):
        return s.reshape(npts, nalt)
    return None


def _pyiri_profiles_vectorized(
    time: dt.datetime,
    alts: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    f107: float,
    foF2_coeff: str,
    hmF2_model: str,
    coord: str,
) -> np.ndarray:
    """
    Vectorized PyIRI call for multiple (lat, lon) points.
    Returns density in m^-3, shape (npts, nalt).
    Falls back to per-point calls if output shape is ambiguous.
    """
    sh = _import_pyiri_sh()
    ts = time if isinstance(time, dt.datetime) else dt.datetime.fromisoformat(str(time))
    ut_hours = (
        float(ts.hour)
        + float(ts.minute) / 60.0
        + (float(ts.second) + float(ts.microsecond) * 1e-6) / 3600.0
    )
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    alts = np.asarray(alts, dtype=float).ravel()
    if lats.size != lons.size:
        raise ValueError("lats and lons must have same size")
    npts = int(lats.size)
    nalt = int(alts.size)

    out = sh.IRI_density_1day(
        int(ts.year),
        int(ts.month),
        int(ts.day),
        np.array([ut_hours], dtype=float),
        lons,
        lats,
        alts,
        float(f107),
        coeff_dir=None,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord,
    )
    den = _dens_to_points_by_alt(out[-1], npts=npts, nalt=nalt)
    if den is not None:
        return den

    logger.warning(
        "PyIRI returned an unexpected array shape in vectorized mode; "
        "falling back to per-point profile evaluation."
    )
    out_pts = np.zeros((npts, nalt), dtype=float)
    for i in range(npts):
        out_i = sh.IRI_density_1day(
            int(ts.year),
            int(ts.month),
            int(ts.day),
            np.array([ut_hours], dtype=float),
            np.array([float(lons[i])], dtype=float),
            np.array([float(lats[i])], dtype=float),
            alts,
            float(f107),
            coeff_dir=None,
            foF2_coeff=foF2_coeff,
            hmF2_model=hmF2_model,
            coord=coord,
        )
        den_i = np.asarray(out_i[-1], dtype=float).squeeze()
        den_i = np.ravel(den_i)
        if den_i.size != nalt:
            z_model = np.linspace(
                float(alts[0]), float(alts[-1]), den_i.size, dtype=float
            )
            den_i = np.interp(alts, z_model, den_i)
        out_pts[i, :] = den_i
    return out_pts


def _iri_profiles_points_chunked(
    time: dt.datetime,
    alts: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    f107: float,
    foF2_coeff: str,
    hmF2_model: str,
    coord: str,
) -> np.ndarray:
    """
    Vectorized profiles for many points.
    Returns cm^-3 with shape (nalt, npts).
    """
    alts = np.asarray(alts, dtype=float)
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    den_m3 = _pyiri_profiles_vectorized(
        time=time,
        alts=alts,
        lats=lats,
        lons=lons,
        f107=f107,
        foF2_coeff=foF2_coeff,
        hmF2_model=hmF2_model,
        coord=coord,
    )  # (npts, nalt)
    return den_m3.T * 1e-6  # (nalt, npts) in cm^-3


class IRI2d(object):
    """IRI electron density sampled along a 2D great-circle route.

    Uses PyIRI to evaluate IRI-2016 profiles at a sequence of (lat, lon, alt)
    points that together form a slant ionospheric cross-section.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.  Required sub-namespace ``cfg.iri_param`` with
        fields ``f107`` (float), ``foF2_coeff`` (str), ``hmF2_model`` (str),
        and ``coord`` (str, e.g. ``"GEO"``).
    event : datetime.datetime
        Reference UTC time for the IRI model.

    Attributes
    ----------
    param : np.ndarray, shape (nalt, npts)
        Electron density in cm⁻³ after :meth:`fetch_dataset`.
    alts, lats, lons : np.ndarray
        Altitude and route point arrays stored by :meth:`fetch_dataset`.
    """

    def __init__(self, cfg, event: dt.datetime):
        self.cfg = cfg
        self.event = event
        ip = getattr(self.cfg, "iri_param", None)
        self.f107 = float(getattr(ip, "f107", 150.0))
        self.foF2_coeff = str(getattr(ip, "foF2_coeff", "CCIR"))
        self.hmF2_model = str(getattr(ip, "hmF2_model", "SHU2015"))
        self.coord = str(getattr(ip, "coord", "GEO"))
        return

    def fetch_dataset(
        self,
        time: dt.datetime,
        lats,
        lons,
        alts,
        workers: int = 1,
        to_file: str = None,
    ):
        self.lats = np.asarray(lats, dtype=float)
        self.alts = np.asarray(alts, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.time = time
        if self.lats.shape != self.lons.shape:
            raise ValueError("lats and lons must have same shape")
        if self.lats.ndim != 1 or self.alts.ndim != 1:
            raise ValueError("lats/lons and alts must be 1D arrays")
        if self.alts.size < 2:
            raise ValueError("alts must contain at least 2 points")
        if int(workers) != 1:
            logger.warning(
                "IRI2d workers argument is ignored; PyIRI runs in serial/vectorized mode."
            )

        self.param = _iri_profiles_points_chunked(
            time=self.time,
            alts=self.alts,
            lats=self.lats,
            lons=self.lons,
            f107=self.f107,
            foF2_coeff=self.foF2_coeff,
            hmF2_model=self.hmF2_model,
            coord=self.coord,
        )
        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts

    def load_from_file(self, to_file: str):
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param


class IRI3d(object):
    """IRI electron density on a 3D geographic grid.

    Evaluates IRI-2016 profiles at every ``(lat, lon)`` point in a meshgrid
    and assembles the results into a ``(nlat, nlon, nalt)`` Ne array suitable
    for use as a PHaRLAP ``iono_en_grid``.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config with ``cfg.iri_param`` (same fields as :class:`IRI2d`).
    event : datetime.datetime
        Reference UTC time for the IRI model.

    Attributes
    ----------
    ne_grid : np.ndarray, shape (nlat, nlon, nalt)
        Electron density in cm⁻³ after :meth:`fetch_dataset`.
    """

    def __init__(self, cfg, event: dt.datetime):
        self.cfg = cfg
        self.event = event
        ip = getattr(self.cfg, "iri_param", None)
        self.f107 = float(getattr(ip, "f107", 150.0))
        self.foF2_coeff = str(getattr(ip, "foF2_coeff", "CCIR"))
        self.hmF2_model = str(getattr(ip, "hmF2_model", "SHU2015"))
        self.coord = str(getattr(ip, "coord", "GEO"))
        return

    def fetch_dataset(
        self,
        time: dt.datetime,
        lats,
        lons,
        alts,
        workers: int = 1,
        to_file: str = None,
    ):
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.alts = np.asarray(alts, dtype=float)
        self.time = time

        if self.lats.ndim != 1 or self.lons.ndim != 1 or self.alts.ndim != 1:
            raise ValueError("lats, lons, alts must be 1D arrays")
        if self.alts.size < 2:
            raise ValueError("alts must have at least 2 points")
        if int(workers) != 1:
            logger.warning(
                "IRI3d workers argument is ignored; PyIRI runs in serial/vectorized mode."
            )

        # Full-grid vectorization: one PyIRI evaluation across all (lat, lon) points.
        lat2d, lon2d = np.meshgrid(self.lats, self.lons, indexing="ij")
        lat_flat = lat2d.ravel()
        lon_flat = lon2d.ravel()
        den_h_pts = _iri_profiles_points_chunked(
            time=self.time,
            alts=self.alts,
            lats=lat_flat,
            lons=lon_flat,
            f107=self.f107,
            foF2_coeff=self.foF2_coeff,
            hmF2_model=self.hmF2_model,
            coord=self.coord,
        )  # (nalt, nlat*nlon) cm^-3
        self.param = den_h_pts.reshape(
            self.alts.size, self.lats.size, self.lons.size
        ).transpose(
            1, 2, 0
        )  # (nlat, nlon, nalt)

        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts
