from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass
class GeomagGrid:
    """
    Geomagnetic grid container.

    Notes:
    - Bx, By, Bz are in Tesla on shape (nlat, nlon, nalt).
    - By convention here:
      Bx = North component, By = East component, Bz = Up component.
    """

    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray
    bmag_t: np.ndarray
    inc_deg: np.ndarray
    dec_deg: np.ndarray
    psi_deg: np.ndarray
    lat_geo: np.ndarray
    lon_geo: np.ndarray
    qd: SimpleNamespace | None = None
    apex: SimpleNamespace | None = None


def _import_pyiri():
    try:
        import PyIRI
        from PyIRI import sh_library as sh
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyIRI is required for trace.geomag. Install `PyIRI` and retry."
        ) from exc
    return PyIRI, sh


def _decimal_year(ts: dt.datetime, pyiri_main_library) -> float:
    if hasattr(pyiri_main_library, "decimal_year"):
        return float(pyiri_main_library.decimal_year(ts))
    y0 = dt.datetime(ts.year, 1, 1, tzinfo=ts.tzinfo)
    y1 = dt.datetime(ts.year + 1, 1, 1, tzinfo=ts.tzinfo)
    return ts.year + (ts - y0).total_seconds() / max((y1 - y0).total_seconds(), 1.0)


def _geo_from_input_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    ts: dt.datetime,
    coord_input: str,
    sh,
):
    """
    Convert input coordinates to GEO for IGRF calls.
    Supported inputs: GEO/IGRF, QD, APEX.
    """
    coord = str(coord_input).upper()
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if coord in {"GEO", "IGRF"}:
        return lats, lons

    # Use PyIRI conversion when available.
    # API names can vary by version; try common variants.
    if not hasattr(sh, "Apex_geo_qd"):
        raise ValueError(
            f"coord_input={coord_input!r} requested, but PyIRI.sh_library.Apex_geo_qd is unavailable."
        )

    if coord == "QD":
        for mode in ("QD_2_GEO", "QDLAT_2_GEO", "QD2GEO"):
            try:
                return sh.Apex_geo_qd(lats, lons, ts, mode)
            except Exception:
                continue
        raise ValueError("Unable to convert QD -> GEO with this PyIRI version.")

    if coord == "APEX":
        # Some versions expose APEX<->GEO via Apex_geo_qd mode strings.
        for mode in ("APEX_2_GEO", "APX_2_GEO", "APEX2GEO"):
            try:
                return sh.Apex_geo_qd(lats, lons, ts, mode)
            except Exception:
                continue
        raise ValueError("Unable to convert APEX -> GEO with this PyIRI version.")

    raise ValueError("coord_input must be one of: GEO, IGRF, QD, APEX")


def _to_qd_apex_if_available(lat_geo, lon_geo, ts: dt.datetime, sh):
    qd = None
    apex = None
    if hasattr(sh, "Apex_geo_qd"):
        try:
            qdlat, qdlon = sh.Apex_geo_qd(lat_geo, lon_geo, ts, "GEO_2_QD")
            qd = SimpleNamespace(
                lat=np.asarray(qdlat, dtype=float), lon=np.asarray(qdlon, dtype=float)
            )
        except Exception:
            qd = None
    if hasattr(sh, "Apex"):
        try:
            # Typical PyIRI Apex signature returns (qdlat, mlt) for given GEO.
            ap_lat, ap_lon = sh.Apex(lat_geo, lon_geo, ts)
            apex = SimpleNamespace(
                lat=np.asarray(ap_lat, dtype=float), lon=np.asarray(ap_lon, dtype=float)
            )
        except Exception:
            apex = None
    return qd, apex


def build_geomag_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    alts_km: np.ndarray,
    time: dt.datetime,
    coord_input: str = "GEO",
    coeff_dir: str | None = None,
) -> GeomagGrid:
    """
    Build PHaRLAP-style geomagnetic component grids using PyIRI IGRF.

    Parameters
    ----------
    lats, lons, alts_km : 1D arrays
        Grid axes.
    time : datetime
        Evaluation time.
    coord_input : str
        Input coordinate mode: GEO/IGRF, QD, or APEX.
    coeff_dir : str | None
        Optional IGRF coeff directory override; defaults to `PyIRI.coeff_dir`.
    """
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    alts_km = np.asarray(alts_km, dtype=float).ravel()
    if lats.ndim != 1 or lons.ndim != 1 or alts_km.ndim != 1:
        raise ValueError("lats, lons, alts_km must be 1D arrays")
    if lats.size == 0 or lons.size == 0 or alts_km.size == 0:
        raise ValueError("lats, lons, alts_km must be non-empty")

    PyIRI, sh = _import_pyiri()
    ts = time if isinstance(time, dt.datetime) else dt.datetime.fromisoformat(str(time))
    dec_year = _decimal_year(ts, PyIRI.main_library)
    coeff = PyIRI.coeff_dir if coeff_dir is None else coeff_dir

    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    lat_in = lat2d.ravel()
    lon_in = lon2d.ravel()
    lat_geo, lon_geo = _geo_from_input_coords(lat_in, lon_in, ts, coord_input, sh)
    lat_geo = np.asarray(lat_geo, dtype=float).ravel()
    lon_geo = np.asarray(lon_geo, dtype=float).ravel()

    nlat, nlon, nalt = lats.size, lons.size, alts_km.size
    bmag_t = np.zeros((nlat, nlon, nalt), dtype=float)
    inc_deg = np.zeros((nlat, nlon, nalt), dtype=float)
    dec_deg = np.zeros((nlat, nlon, nalt), dtype=float)
    psi_deg = np.zeros((nlat, nlon, nalt), dtype=float)
    Bx = np.zeros((nlat, nlon, nalt), dtype=float)
    By = np.zeros((nlat, nlon, nalt), dtype=float)
    Bz = np.zeros((nlat, nlon, nalt), dtype=float)

    for k, alt_km in enumerate(alts_km):
        out = PyIRI.igrf_library.inclination(
            coeff,
            dec_year,
            lon_geo,
            lat_geo,
            float(alt_km),
            only_inc=False,
        )
        inc = np.asarray(out[0], dtype=float).ravel()
        dec = np.asarray(out[1], dtype=float).ravel()
        mag_nT = np.asarray(out[6], dtype=float).ravel()

        bmag = mag_nT * 1e-9  # nT -> T
        I = np.deg2rad(inc)
        D = np.deg2rad(dec)

        # NED convention from IGRF-like inclination/declination:
        # Bn = B cos(I) cos(D), Be = B cos(I) sin(D), Bd = B sin(I) (down)
        bn = bmag * np.cos(I) * np.cos(D)
        be = bmag * np.cos(I) * np.sin(D)
        bu = -bmag * np.sin(I)  # convert down -> up

        bmag_t[:, :, k] = bmag.reshape(nlat, nlon)
        inc_deg[:, :, k] = inc.reshape(nlat, nlon)
        dec_deg[:, :, k] = dec.reshape(nlat, nlon)
        psi_deg[:, :, k] = (90.0 - np.abs(inc)).reshape(nlat, nlon)
        Bx[:, :, k] = bn.reshape(nlat, nlon)
        By[:, :, k] = be.reshape(nlat, nlon)
        Bz[:, :, k] = bu.reshape(nlat, nlon)

    qd, apex = _to_qd_apex_if_available(lat_geo, lon_geo, ts, sh)
    lat_geo_grid = lat_geo.reshape(nlat, nlon)
    lon_geo_grid = lon_geo.reshape(nlat, nlon)
    if qd is not None:
        qd = SimpleNamespace(
            lat=np.asarray(qd.lat, dtype=float).reshape(nlat, nlon),
            lon=np.asarray(qd.lon, dtype=float).reshape(nlat, nlon),
        )
    if apex is not None:
        apex = SimpleNamespace(
            lat=np.asarray(apex.lat, dtype=float).reshape(nlat, nlon),
            lon=np.asarray(apex.lon, dtype=float).reshape(nlat, nlon),
        )

    return GeomagGrid(
        Bx=Bx,
        By=By,
        Bz=Bz,
        bmag_t=bmag_t,
        inc_deg=inc_deg,
        dec_deg=dec_deg,
        psi_deg=psi_deg,
        lat_geo=lat_geo_grid,
        lon_geo=lon_geo_grid,
        qd=qd,
        apex=apex,
    )
