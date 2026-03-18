import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from dateutil import parser as dparser
from loguru import logger
from scipy.io import loadmat, savemat

from hfpytrace import utils


class SAMI3(object):
    def __init__(
        self,
        cfg,
        event,
        round_to=-1,
    ):
        self.cfg = cfg
        self.event = event
        self.file_name = os.path.join(
            self.cfg.density_file_location, self.cfg.density_file_name
        )
        self.cfg.density_simulated_datetime = dparser.isoparse(
            self.cfg.density_simulated_datetime
        )
        self.round_to = round_to
        self.load_nc_dataset()
        return

    def load_nc_dataset(self):
        self.store = {}
        logger.info(f"Load files -> {self.file_name}")
        ds = xr.open_dataset(self.file_name)
        self.store["time"] = [
            self.cfg.density_simulated_datetime + dt.timedelta(hours=float(d))
            for d in ds.variables["time"][:]
        ]
        if self.round_to > 0:
            logger.info("Round to nearest minutes!!!!!!!!!!!!")
            self.store["time"] = [self.round_time(e) for e in self.store["time"]]
        self.dates = self.store["time"]
        self.store["alt"] = ds.variables["alt0"].values  # in km
        self.store["glat"] = ds.variables["lat0"].values  # in deg
        self.store["glon"] = ds.variables["lon0"].values  # in 0-360 deg
        # self.store["glon"] = (
        #     np.mod(self.store["glon"] - 180.0, 360.0) - 180.0
        # )  # to +/- 180
        self.store["eden"] = ds.variables["dene0"][:]  # in cc
        ds.close()
        del ds
        return

    def round_time(self, t):
        """Round a datetime object to any time lapse in seconds
        t : datetime.datetime object, default now.
        """
        seconds = (t.replace(tzinfo=None) - t.min).seconds
        rounding = (seconds + self.round_to / 2) // self.round_to * self.round_to
        return t + dt.timedelta(0, rounding - seconds, -t.microsecond)

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
            dt_bracket = (self.store["time"][j] - self.store["time"][i]).total_seconds()
            alpha = (time - self.store["time"][i]).total_seconds() / dt_bracket
            px, _ = self.fetch_interpolated_data(lats, lons, alts, i)
            py, self.alts = self.fetch_interpolated_data(lats, lons, alts, j)
            self.param = (1.0 - alpha) * px + alpha * py

        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts

    def _bilinear_ne_profile(self, D, lat, lon):
        """Return a raw Ne altitude profile (shape: n_alt) bilinearly interpolated
        at (lat, lon) from the four surrounding 1° grid cells.

        D has shape (n_lon, n_alt, n_lat).
        lon must already be normalised to [0, 360).
        """
        glat = self.store["glat"]
        glon = self.store["glon"]
        n_lon, _, n_lat = D.shape

        # ── latitude bracket ─────────────────────────────────────────────
        i0 = int(np.clip(np.searchsorted(glat, lat, side="right") - 1, 0, n_lat - 2))
        i1 = i0 + 1
        lat_f = float(np.clip((lat - glat[i0]) / (glat[i1] - glat[i0]), 0.0, 1.0))

        # ── longitude bracket with 0–360 wraparound ──────────────────────
        j0 = int((np.searchsorted(glon, lon, side="right") - 1) % n_lon)
        j1 = int((j0 + 1) % n_lon)
        dlon = float(glon[j1] - glon[j0]) if j1 > j0 else float(glon[j1] + 360.0 - glon[j0])
        lon_f = float(np.clip((lon - float(glon[j0])) / dlon if dlon > 0.0 else 0.0, 0.0, 1.0))

        # ── bilinear blend over four corners ─────────────────────────────
        p00 = D[j0, :, i0].astype(float)   # lon0, lat0
        p10 = D[j1, :, i0].astype(float)   # lon1, lat0
        p01 = D[j0, :, i1].astype(float)   # lon0, lat1
        p11 = D[j1, :, i1].astype(float)   # lon1, lat1
        return ((1.0 - lon_f) * (1.0 - lat_f) * p00
                + lon_f * (1.0 - lat_f) * p10
                + (1.0 - lon_f) * lat_f * p01
                + lon_f * lat_f * p11)

    def fetch_interpolated_data(
        self,
        lats,
        lons,
        alts,
        index,
    ):
        n = len(lats)
        D = self.store["eden"][index]
        galt = self.store["alt"]
        out, ix = np.zeros((len(alts), n)) * np.nan, 0
        for lat, lon in zip(lats, lons):
            lon = float(np.mod(360 + lon, 360))
            o = self._bilinear_ne_profile(D, lat, lon)
            out[:, ix] = utils.interpolate_by_altitude(
                galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
            )  # in cc
            ix += 1
        return out, galt

    def _fetch_profile_at_index(self, index, lat, lon, alts):
        D = self.store["eden"][index]
        galt = self.store["alt"]
        lon = float(np.mod(360 + lon, 360))
        o = self._bilinear_ne_profile(D, lat, lon)
        p = utils.interpolate_by_altitude(
            galt, alts, o, self.cfg.scale, self.cfg.kind, method="extp"
        )
        return np.asarray(p, dtype=float), np.asarray(galt, dtype=float)

    def _fetch_cube_at_index(self, index, lats, lons, alts, workers: int = 1):
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        alts = np.asarray(alts, dtype=float)
        out = np.zeros((lats.size, lons.size, alts.size), dtype=float) * np.nan
        ij = [(ii, jj) for ii in range(lats.size) for jj in range(lons.size)]

        def _job(ii, jj):
            p, _ = self._fetch_profile_at_index(index, lats[ii], lons[jj], alts)
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
            dt_bracket = (self.store["time"][j] - self.store["time"][i]).total_seconds()
            alpha = (time - self.store["time"][i]).total_seconds() / dt_bracket
            px = self._fetch_cube_at_index(i, lats, lons, alts, workers)
            py = self._fetch_cube_at_index(j, lats, lons, alts, workers)
            self.param3d = (1.0 - alpha) * px + alpha * py

        self.alts = np.asarray(alts, dtype=float)
        if to_file:
            savemat(to_file, dict(ne=self.param3d))
        return self.param3d, self.alts

    def load_from_file(self, to_file: str):
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param
