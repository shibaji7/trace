import datetime as dt

import iricore
import numpy as np
from dateutil import parser as dparser
from loguru import logger
from scipy.io import loadmat, savemat


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
