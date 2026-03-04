import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor
from trace import utils

import numpy as np
import xarray as xr
from loguru import logger
from scipy.io import loadmat, savemat


class GITM2d(object):
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
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param
