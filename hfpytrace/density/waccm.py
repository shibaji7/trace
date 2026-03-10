import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from loguru import logger
from scipy.io import loadmat, savemat

from hfpytrace import utils


class WACCMX2d(object):
    def __init__(
        self,
        cfg,
        event,
    ):
        self.cfg = cfg
        self.event = event
        self.file_name = os.path.join(
            self.cfg.density_file_location, self.cfg.density_file_name
        )
        self.load_nc_dataset()
        return

    def transform_density(self, P, T, var, unit="cm"):
        """
        Transform electron density from mol/mol to /cc or /cm
        Formula
        -------
        P = nKT, K = Boltzmann constant in SI
        lev: pressure in hPa (1hPa=100Pa)
        T: neutral temperature
        e/O2/...: density in mol/mol

        T in K, P in Pa, n is electron density in /cubic meter or /cc
        """
        logger.info(f"Transform density scale/unit")
        P = P * 1e2
        u = 1 if unit == "cm" else 1e-6
        den = np.zeros_like(var)
        for i, p in enumerate(P):
            na = p * utils.pconst["avo"] / (utils.pconst["R"] * T[:, i, :, :])
            n = u * na * var[:, i, :, :]
            den[:, i, :, :] = n
        return den

    def load_nc_dataset(self):
        """
        Load netcdf4 dataset available
        """
        self.store = {}
        drop_vars = [
            "WI",
            "VI",
            "UI",
            "TIon",
            "TElec",
            "Op_CHMP",
            "Op_CHML",
            "OpDens",
            "Op",
            "NOp",
            "O2p",
            "N2p",
            "SOLAR_MASK",
            "amb_diff",
            "dwind",
            "dfield",
            "op_dt",
        ]
        logger.info(f"Load files -> {self.file_name}")
        ds = xr.open_dataset(self.file_name, drop_variables=drop_vars)
        self.store["glat"], self.store["glon"] = (
            ds.lat.values,
            np.mod(360 + ds.lon.values, 360),
        )
        self.store["time"] = [
            self.event.replace(hour=0) + dt.timedelta(seconds=float(d))
            for d in ds.variables["time"][:]
        ]
        self.store["alt"] = ds.variables["Z3GM"][:] / 1e3
        self.store["eden"] = ds.variables["e"][:]
        self.store["T"] = ds.variables["T"][:]
        self.store["lev"] = ds.variables["lev"][:]
        self.store["eden"] = self.transform_density(
            self.store["lev"], self.store["T"], self.store["eden"]
        )
        logger.info(f"Shape of eden: {self.store['eden'].shape}")
        ds.close()
        del ds
        return

    def find_time_index(self, t):
        """Finds the index of the interval in the array where the number falls.

        Returns:
            The index of the interval where the number falls, or -1 if the number is
            outside the range of the array.
        """
        for i in range(len(self.store["time"]) - 1):
            if self.store["time"][i] <= t < self.store["time"][i + 1]:
                return i, i + 1
        return -1, 0

    def fetch_dataset(
        self,
        time,
        lats,
        lons,
        alts,
        to_file=None,
    ):
        # Selecting based on time index
        if time in self.store["time"]:
            logger.info(f"Into exact timestamp {time}")
            # Select the exact index if timestamp in the simulation
            i = self.store["time"].index(time)
            self.param, self.alts = self.fetch_interpolated_data(lats, lons, alts, i)
        else:
            # Select the two index if timestamp in the simulation
            i, j = self.find_time_index(time)
            logger.info(
                f"{time}/ into between timestamp {self.store['time'][i]} & {self.store['time'][j]}"
            )
            weights = (self.store["time"][1] - self.store["time"][0]).total_seconds()
            px, _ = self.fetch_interpolated_data(lats, lons, alts, i)
            py, self.alts = self.fetch_interpolated_data(lats, lons, alts, j)
            i_wg, j_wg = (
                (time - self.store["time"][i]).total_seconds() / weights,
                (self.store["time"][j] - time).total_seconds() / weights,
            )
            self.param = px * i_wg + py * j_wg

        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts

    def fetch_interpolated_data(
        self,
        lats,
        lons,
        alts,
        index,
    ):
        n = len(lats)
        D = self.store["eden"][index]
        glat = self.store["glat"]
        glon = self.store["glon"]
        out, ix = np.zeros((len(alts), n)) * np.nan, 0
        for lat, lon in zip(lats, lons):
            lon = np.mod(360 + lon, 360)
            idx = np.argmin(np.abs(glat - lat))
            idy = np.argmin(np.abs(glon - lon))
            o = D[:, idx, idy]
            galt = self.store["alt"][index, :, idx, idy]
            out[:, ix] = (
                utils.interpolate_by_altitude(
                    galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
                )
                * 1e-6
            )
            ix += 1
        return out, galt

    def _fetch_profile_at_index(self, index, lat, lon, alts):
        D = self.store["eden"][index]
        glat = self.store["glat"]
        glon = self.store["glon"]
        lon = np.mod(360 + lon, 360)
        idx = np.argmin(np.abs(glat - lat))
        idy = np.argmin(np.abs(glon - lon))
        o = D[:, idx, idy]
        galt = self.store["alt"][index, :, idx, idy]
        p = (
            utils.interpolate_by_altitude(
                galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
            )
            * 1e-6
        )
        return np.asarray(p, dtype=float)

    def _fetch_cube_at_index(self, index, lats, lons, alts, workers: int = 1):
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        alts = np.asarray(alts, dtype=float)
        out = np.zeros((lats.size, lons.size, alts.size), dtype=float) * np.nan
        ij = [(ii, jj) for ii in range(lats.size) for jj in range(lons.size)]

        def _job(ii, jj):
            p = self._fetch_profile_at_index(index, lats[ii], lons[jj], alts)
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
        return out

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
        point-wise parallelism and time interpolation.
        """
        if time in self.store["time"]:
            i = self.store["time"].index(time)
            self.param3d = self._fetch_cube_at_index(i, lats, lons, alts, workers)
        else:
            i, j = self.find_time_index(time)
            weights = (self.store["time"][1] - self.store["time"][0]).total_seconds()
            px = self._fetch_cube_at_index(i, lats, lons, alts, workers)
            py = self._fetch_cube_at_index(j, lats, lons, alts, workers)
            i_wg = (time - self.store["time"][i]).total_seconds() / weights
            j_wg = (self.store["time"][j] - time).total_seconds() / weights
            self.param3d = px * i_wg + py * j_wg

        self.alts = np.asarray(alts, dtype=float)
        if to_file:
            savemat(to_file, dict(ne=self.param3d))
        return self.param3d, self.alts

    def load_from_file(self, to_file: str):
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param
