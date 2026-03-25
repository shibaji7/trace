"""GITM electron density reader.

Reads GITM (Global Ionosphere-Thermosphere Model) netCDF output files and
interpolates the electron density onto arbitrary lat/lon/altitude grids for
use in HF ray-tracing.

Requires
--------
xarray : for netCDF I/O.

Classes
-------
GITM2d
    Loads a GITM netCDF file and provides ``fetch_dataset`` /
    ``fetch_dataset_3d`` methods that resample Ne onto user-specified grids.
"""

import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from loguru import logger
from scipy.io import loadmat, savemat

from hfpytrace import utils


class GITM2d(object):
    """Electron density from GITM simulation output.

    Reads a GITM netCDF file and interpolates electron density to requested
    (lat, lon, alt) grids.  The dataset is loaded once at construction time.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config with ``density_file_location``, ``density_file_name``, ``scale``,
        and ``kind`` (interpolation scale/kind).
    event : datetime.datetime
        Reference event time; the nearest GITM time step is used.
    """

    def __init__(
        self,
        cfg,
        event,
    ):
        self.cfg = cfg
        self.file_name = os.path.join(
            self.cfg.density_file_location, self.cfg.density_file_name
        )
        self.event = event
        self.load_nc_dataset()
        return

    def load_nc_dataset(self):
        """
        Load netcdf4 dataset available
        """
        self.store = {}
        drop_vars = [
            "wn",
            "vn",
            "un",
            "tn",
            "denn",
            "TEC",
            "wi",
            "vi",
            "ui",
            "SigH",
            "SigP",
            "time",
        ]
        file = self.file_name
        logger.info(f"Load files -> {file}")
        ds = xr.open_dataset(file, drop_variables=drop_vars)
        self.store["time"] = [
            dt.datetime(y, m, d, h, mm)
            for y, m, d, h, mm in zip(
                ds.year.values,
                ds.month.values,
                ds.day.values,
                ds.hour.values,
                ds.minute.values,
            )
        ]
        (
            self.store["glat"],
            self.store["glon"],
            self.store["alt"],
            self.store["eden"],
        ) = (
            ds.glat.values,
            # converted to -180 : 180
            # np.mod(180 + ds.glon.values, 360) - 180,
            np.mod(360 + ds.glon.values, 360),
            ds.alt.values / 1e3,
            ds.dene.values,
        )
        logger.info(f"Shape of eden: {ds.dene}")
        ds.close()
        del ds
        return

    def fetch_dataset(
        self,
        time,
        lats,
        lons,
        alts,
        to_file=None,
        **kwrds,
    ):
        """Fetch 2D electron density along a route at a given time.

        The nearest GITM time step is selected by minimising the absolute
        time difference.

        Parameters
        ----------
        time : datetime.datetime
            Requested time snapshot.
        lats, lons : array-like, shape (npts,)
            Geographic coordinates of route points [°].
        alts : array-like, shape (nalt,)
            Target altitude levels [km].
        to_file : str, optional
            If given, save the result to a ``.mat`` file at this path.
        **kwrds
            Unused; accepted for API compatibility.

        Returns
        -------
        param : np.ndarray, shape (nalt, npts)
            Electron density in cm⁻³.
        alts : np.ndarray
            Model altitude grid [km] (returned for caller convenience).
        """
        i = np.argmin([np.abs((t - time).total_seconds()) for t in self.store["time"]])
        n = len(lats)
        D = self.store["eden"][i]
        glat = self.store["glat"]
        glon = self.store["glon"]
        galt = self.store["alt"]
        out, ix = np.zeros((len(alts), n)) * np.nan, 0

        for lat, lon in zip(lats, lons):
            lon = np.mod(360 + lon, 360)
            idx = np.argmin(np.abs(glon - lon))
            idy = np.argmin(np.abs(glat - lat))
            o = D[:, idy, idx]
            out[:, ix] = (
                utils.interpolate_by_altitude(
                    galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
                )
                * 1e-6
            )
            out[alts < 50, ix] = 0
            ix += 1
        self.param, self.alts = out, galt
        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts

    def _fetch_profile_1d(self, D, glat, glon, galt, lat, lon, alts):
        """Return a single altitude-interpolated Ne profile at (lat, lon).

        Parameters
        ----------
        D : np.ndarray, shape (nalt, nlat, nlon)
            Electron density slice for a single time index.
        glat, glon : np.ndarray
            Model latitude/longitude axes [°].
        galt : np.ndarray
            Model altitude axis [km].
        lat, lon : float
            Target geographic coordinates [°].
        alts : array-like
            Target altitude levels [km].

        Returns
        -------
        np.ndarray, shape (nalt,)
            Electron density in cm⁻³ (zero below 50 km).
        """
        lon = np.mod(360 + lon, 360)
        idx = np.argmin(np.abs(glon - lon))
        idy = np.argmin(np.abs(glat - lat))
        o = D[:, idy, idx]
        p = (
            utils.interpolate_by_altitude(
                galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
            )
            * 1e-6
        )
        p = np.asarray(p, dtype=float)
        p[np.asarray(alts) < 50] = 0
        return p

    def fetch_dataset_3d(
        self,
        time,
        lats,
        lons,
        alts,
        to_file=None,
        workers: int = 1,
    ):
        """
        Fetch 3D electron density cube on (lat, lon, alt) with optional
        point-wise parallelism.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        alts = np.asarray(alts, dtype=float)
        i = np.argmin([np.abs((t - time).total_seconds()) for t in self.store["time"]])
        D = self.store["eden"][i]
        glat = self.store["glat"]
        glon = self.store["glon"]
        galt = self.store["alt"]

        out = np.zeros((lats.size, lons.size, alts.size), dtype=float) * np.nan
        ij = [(ii, jj) for ii in range(lats.size) for jj in range(lons.size)]

        def _job(ii, jj):
            p = self._fetch_profile_1d(D, glat, glon, galt, lats[ii], lons[jj], alts)
            return ii, jj, p

        n_workers = max(1, int(workers))
        if n_workers == 1:
            for ii, jj in ij:
                _, _, p = _job(ii, jj)
                out[ii, jj, :] = p
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for ii, jj, p in ex.map(lambda t: _job(*t), ij):
                    out[ii, jj, :] = p

        self.param3d, self.alts = out, alts
        if to_file:
            savemat(to_file, dict(ne=self.param3d))
        return self.param3d, self.alts

    def load_from_file(self, to_file: str):
        """Load a previously saved electron density array from a ``.mat`` file.

        Parameters
        ----------
        to_file : str
            Path to the ``.mat`` file containing key ``ne``.

        Returns
        -------
        np.ndarray
            Electron density array as stored.
        """
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param
