import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

import iricore
import numpy as np
from dateutil import parser as dparser
from loguru import logger
from scipy.io import loadmat, savemat


def _iri_eval_lat(args):
    """
    Process-safe helper to evaluate one latitude row across all longitudes.
    """
    i, lat, lons, time, alt_range, iri_version = args
    lons = np.asarray(lons, dtype=float)
    nalt = int(round((alt_range[1] - alt_range[0]) / alt_range[2])) + 1
    out = np.zeros((lons.size, nalt), dtype=float)
    for j, lon in enumerate(lons):
        iriout = iricore.iri(
            time,
            alt_range,
            float(lat),
            float(lon),
            iri_version,
        )
        out[j, :] = np.asarray(iriout.edens, dtype=float) * 1e-6
    return i, out


class IRI2d(object):
    def __init__(
        self,
        cfg,
        event: dt.datetime,
    ):
        self.cfg = cfg
        self.event = event
        self.iri_version = self.cfg.iri_param.iri_version
        return

    def fetch_dataset(
        self,
        time: dt.datetime,
        lats,
        lons,
        alts,
        to_file: str = None,
    ):
        self.lats, self.alts, self.lons = (lats, alts, lons)
        self.time = time
        self.param = np.zeros((len(self.alts), len(self.lats)))
        alt_range = [alts[0], alts[-1], alts[1] - alts[0]]
        for i in range(len(self.lats)):
            iriout = iricore.iri(
                self.time,
                alt_range,
                self.lats[i],
                self.lons[i],
                self.iri_version,
            )
            self.param[:, i] = iriout.edens * 1e-6
        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts

    def load_from_file(self, to_file: str):
        logger.info(f"Load from file {to_file.split('/')[-1]}")
        self.param = loadmat(to_file)["ne"]
        return self.param


class IRI3d(object):
    """
    Build IRI electron density on a 3D (lat x lon x height) grid.
    """

    def __init__(
        self,
        cfg,
        event: dt.datetime,
    ):
        self.cfg = cfg
        self.event = event
        self.iri_version = self.cfg.iri_param.iri_version
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

        alt_range = [self.alts[0], self.alts[-1], self.alts[1] - self.alts[0]]
        self.param = np.zeros((self.lats.size, self.lons.size, self.alts.size))

        n_workers = max(1, int(workers))
        if n_workers == 1:
            for i in range(self.lats.size):
                for j in range(self.lons.size):
                    iriout = iricore.iri(
                        self.time,
                        alt_range,
                        float(self.lats[i]),
                        float(self.lons[j]),
                        self.iri_version,
                    )
                    self.param[i, j, :] = np.asarray(iriout.edens, dtype=float) * 1e-6
        else:
            logger.info(
                f"Running IRI3d on grid lat={self.lats.size}, lon={self.lons.size}, "
                f"alt={self.alts.size} with process workers={n_workers}"
            )
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = [
                    ex.submit(
                        _iri_eval_lat,
                        (
                            i,
                            float(self.lats[i]),
                            self.lons.tolist(),
                            self.time,
                            alt_range,
                            self.iri_version,
                        ),
                    )
                    for i in range(self.lats.size)
                ]
                for fut in as_completed(futs):
                    i, lat_slice = fut.result()
                    self.param[i, :, :] = lat_slice

        if to_file:
            savemat(to_file, dict(ne=self.param))
        return self.param, self.alts
