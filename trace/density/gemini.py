import datetime as dt
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from trace import utils

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import loadmat, savemat


class GEMINI2d(object):
    def __init__(
        self,
        cfg,
        event,
    ):
        self.cfg = cfg
        self.event = event
        self.load_grid()
        self.search_mat_files()
        return

    def search_mat_files(self):
        """
        Search all files
        """
        self.files = glob.glob(os.path.join(self.cfg.density_file_location, "*.mat"))
        self.files.remove(self.ccord_file)
        self.files.sort()
        self.dates = []
        for fname in self.files:
            day = dt.datetime.strptime(fname.split("_")[0].split("/")[-1], "%Y%m%d")
            seconds = int(fname.split("_")[1].split(".")[0])
            day = day + dt.timedelta(seconds=seconds)
            self.dates.append(day)
        return

    def load_grid(self):
        """
        Load grid file
        """
        logger.info(f"Loading GEMINI Grid files")
        self.ccord_file = self.cfg.density_file_location + self.cfg.grid_coordinate_file
        self.grid = pd.DataFrame()
        with h5py.File(self.ccord_file, "r") as fkey:
            self.grid["glat"] = np.array(fkey.get("glat")[0]).tolist()
            self.grid["glon"] = np.mod((fkey.get("glon")[0] + 180), 360) - 180
            self.grid["alt"] = fkey.get("galt")[0, :]
            self.grid["glat"] = fkey.get("glat")[0, :]
        return

    def load_data(self, fname):
        with h5py.File(fname, "r") as fkey:
            o = self.grid.copy()
            o["nsall"] = fkey.get("nsall")[0, :]
            logger.info(f"Load dataset: {fname}")
        return o

    def fetch_dataset(
        self,
        time: dt.datetime,
        lats,
        lons,
        alts,
        to_file: str = None,
        dlat: float = 0.2,
        dlon: float = 0.2,
        intp_edens_xlim_index: float = 0,
    ):
        self.param = np.zeros((len(alts), len(lats)))
        i = np.argmin([np.abs((t - time).total_seconds()) for t in self.dates])
        file = self.files[i]
        df = self.load_data(file)
        df = df[df.alt >= 0]
        self.param = np.zeros((len(alts), len(lats)))
        for ix, lat, lon in zip(range(len(lats)), lats, lons):
            uf = df[
                (df.glat >= lat - dlat)
                & (df.glat <= lat + dlat)
                & (df.glon >= lon - dlon)
                & (df.glon <= lon + dlon)
            ]
            uf = uf.groupby("alt").mean().reset_index()
            if (len(uf) > 0) and (uf.alt.max() / 1e3 >= max(alts) * 0.7):
                method = "intp" if uf.alt.max() / 1e3 > max(alts) else "extp"
                self.param[:, ix] = (
                    utils.interpolate_by_altitude(
                        np.array(uf.alt) / 1e3,
                        alts,
                        np.array(uf["nsall"]),
                        self.cfg.scale,
                        self.cfg.kind,
                        method=method,
                    )
                    * 1e-6
                )
        logger.info(f"Index: {intp_edens_xlim_index}")
        if intp_edens_xlim_index:
            for j in range(intp_edens_xlim_index, len(lats)):
                self.param[:, j] = self.param[:, intp_edens_xlim_index]
        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, alts

    def _fetch_profile_from_df(self, df, lat, lon, alts, dlat=0.2, dlon=0.2):
        uf = df[
            (df.glat >= lat - dlat)
            & (df.glat <= lat + dlat)
            & (df.glon >= lon - dlon)
            & (df.glon <= lon + dlon)
        ]
        uf = uf.groupby("alt").mean().reset_index()
        p = np.zeros(len(alts), dtype=float)
        if (len(uf) > 0) and (uf.alt.max() / 1e3 >= max(alts) * 0.7):
            method = "intp" if uf.alt.max() / 1e3 > max(alts) else "extp"
            p = (
                utils.interpolate_by_altitude(
                    np.array(uf.alt) / 1e3,
                    alts,
                    np.array(uf["nsall"]),
                    self.cfg.scale,
                    self.cfg.kind,
                    method=method,
                )
                * 1e-6
            )
        return np.asarray(p, dtype=float)

    def fetch_dataset_3d(
        self,
        time: dt.datetime,
        lats,
        lons,
        alts,
        to_file: str = None,
        workers: int = 1,
        dlat: float = 0.2,
        dlon: float = 0.2,
    ):
        """
        Fetch 3D electron density cube on (lat, lon, alt) with optional
        point-wise parallelism.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        alts = np.asarray(alts, dtype=float)
        i = np.argmin([np.abs((t - time).total_seconds()) for t in self.dates])
        file = self.files[i]
        df = self.load_data(file)
        df = df[df.alt >= 0]

        out = np.zeros((lats.size, lons.size, alts.size), dtype=float)
        ij = [(ii, jj) for ii in range(lats.size) for jj in range(lons.size)]

        def _job(ii, jj):
            p = self._fetch_profile_from_df(
                df, lats[ii], lons[jj], alts, dlat=dlat, dlon=dlon
            )
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

        self.param3d = out
        if to_file:
            savemat(to_file, dict(ne=self.param3d))
        return self.param3d, alts

    def load_from_file(self, to_file: str):
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param
