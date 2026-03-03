import datetime as dt
import warnings
from dataclasses import dataclass
from trace.utils import pconst

import numpy as np
from loguru import logger


@dataclass
class Collision_en:
    N2: np.ndarray | None = None
    O2: np.ndarray | None = None
    O: np.ndarray | None = None
    H: np.ndarray | None = None
    He: np.ndarray | None = None
    total: np.ndarray | None = None


@dataclass
class Collision_ei:
    O2p: np.ndarray | None = None
    Op: np.ndarray | None = None
    total: np.ndarray | None = None


@dataclass
class Collision_SN:
    en: Collision_en | None = None
    ei: Collision_ei | None = None
    total: np.ndarray | None = None


@dataclass
class Collision:
    nu_ft: np.ndarray | None = None
    nu_av_cc: np.ndarray | None = None
    nu_av_mb: np.ndarray | None = None
    nu_sn: Collision_SN | None = None


class NRLMSISE2D(object):
    """
    Build NRLMSISE-00 neutral background on a 2D (height x path-range) grid.

    Inputs:
    - date: datetime for model evaluation
    - lats, lons: 1D path coordinates, length Nr
    - heights_km: 1D heights, length Nh

    Outputs are stored in `self.msise` as arrays of shape (Nh, Nr), with
    densities in cm^-3 and temperatures in K.
    """

    def __init__(
        self,
        date: dt.datetime,
        lats,
        lons,
        heights_km,
        update_spaceweather: bool = False,
        suppress_spaceweather_warning: bool = True,
    ):
        self.date = date
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.heights_km = np.asarray(heights_km, dtype=float)
        self.update_spaceweather = update_spaceweather
        self.suppress_spaceweather_warning = suppress_spaceweather_warning
        if self.lats.shape != self.lons.shape:
            raise ValueError("lats and lons must have the same shape")
        if self.lats.ndim != 1:
            raise ValueError("lats and lons must be 1D arrays")
        if self.heights_km.ndim != 1:
            raise ValueError("heights_km must be a 1D array")
        self.msise = self.fetch_dataset()

    def fetch_dataset(self) -> dict[str, np.ndarray]:
        try:
            from nrlmsise00.dataset import msise_4d
        except ImportError as exc:
            raise ImportError(
                "nrlmsise00 dataset interface is unavailable. "
                "Install extras: pip install 'nrlmsise00[dataset]'"
            ) from exc

        if self.update_spaceweather:
            try:
                from spaceweather import sw

                sw.update_data()
                logger.info("Updated spaceweather local data files.")
            except Exception as exc:
                logger.warning(f"spaceweather update failed: {exc}")

        nh = self.heights_km.size
        nr = self.lats.size
        out = dict(
            N2=np.zeros((nh, nr), dtype=float),
            O2=np.zeros((nh, nr), dtype=float),
            O=np.zeros((nh, nr), dtype=float),
            H=np.zeros((nh, nr), dtype=float),
            He=np.zeros((nh, nr), dtype=float),
            Tn=np.zeros((nh, nr), dtype=float),
            t_nn=np.zeros((nh, nr), dtype=float),
        )

        logger.info(f"Running NRLMSISE-00 for {self.date} on grid Nh={nh}, Nr={nr}")
        for j, (lat, lon) in enumerate(zip(self.lats, self.lons)):
            with warnings.catch_warnings():
                if self.suppress_spaceweather_warning:
                    warnings.filterwarnings(
                        "ignore",
                        message="Local data files are older than 30days.*",
                        category=UserWarning,
                    )
                ds = msise_4d(
                    self.date, self.heights_km, np.array([lat]), np.array([lon])
                )
            out["N2"][:, j] = ds.variables["N2"].values[0, :, 0, 0]
            out["O2"][:, j] = ds.variables["O2"].values[0, :, 0, 0]
            out["O"][:, j] = ds.variables["O"].values[0, :, 0, 0]
            out["H"][:, j] = ds.variables["H"].values[0, :, 0, 0]
            out["He"][:, j] = ds.variables["He"].values[0, :, 0, 0]
            out["Tn"][:, j] = ds.variables["Talt"].values[0, :, 0, 0]

        out["t_nn"] = out["N2"] + out["O2"] + out["O"] + out["H"] + out["He"]
        return out

    def as_collision_kwargs(self) -> dict[str, np.ndarray]:
        """
        Convenience mapping for ComputeCollision neutral inputs.
        """
        return dict(
            Tn=self.msise["Tn"],
            N2=self.msise["N2"],
            O2=self.msise["O2"],
            O=self.msise["O"],
            H=self.msise["H"],
            He=self.msise["He"],
        )


class ComputeCollision(object):
    """
    Estimate collision profiles from provided plasma/neutral state arrays.

    Expected units match existing TRACE usage:
    - Temperatures (Te, Ti, Tn): K
    - Densities (edens, O2p, Op, N2, O2, O, H, He): cm^-3
    """

    def __init__(
        self,
        Te,
        Ti,
        Tn,
        edens,
        O2p,
        Op,
        N2,
        O2,
        O,
        H,
        He,
        date: dt.datetime | None = None,
    ):
        self.Te = np.asarray(Te, dtype=float)
        self.Ti = np.asarray(Ti, dtype=float)
        self.Tn = np.asarray(Tn, dtype=float)

        self.edens = np.asarray(edens, dtype=float)
        self.O2p = np.asarray(O2p, dtype=float)
        self.Op = np.asarray(Op, dtype=float)

        self.N2 = np.asarray(N2, dtype=float)
        self.O2 = np.asarray(O2, dtype=float)
        self.O = np.asarray(O, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.He = np.asarray(He, dtype=float)

        self.t_nn = self.N2 + self.O2 + self.O + self.H + self.He
        self.date = date
        if date:
            logger.info(f"Compute collision profiles for {date}")

        self.collision = Collision()
        self.collision.nu_sn = Collision_SN(
            en=Collision_en(),
            ei=Collision_ei(),
            total=np.zeros_like(self.Te),
        )

        self.collision.nu_ft = self.calculate_FT_collision_frequency()
        self.collision.nu_av_cc = self.calculate_FT_collision_frequency(2.5)
        self.collision.nu_av_mb = self.calculate_FT_collision_frequency(1.5)
        self.calculate_SN_en_collision_frequency()
        self.calculate_SN_ei_collision_frequency()
        return

    @classmethod
    def from_nrlmsise(
        cls,
        *,
        date: dt.datetime,
        lats,
        lons,
        heights_km,
        Te,
        Ti,
        edens,
        O2p,
        Op,
        update_spaceweather: bool = False,
        suppress_spaceweather_warning: bool = True,
    ):
        """
        Build collision model using neutral fields from NRLMSISE2D and plasma
        fields from the caller (e.g., IRI).
        """
        bg = NRLMSISE2D(
            date=date,
            lats=lats,
            lons=lons,
            heights_km=heights_km,
            update_spaceweather=update_spaceweather,
            suppress_spaceweather_warning=suppress_spaceweather_warning,
        )
        return cls(
            Te=Te,
            Ti=Ti,
            Tn=bg.msise["Tn"],
            edens=edens,
            O2p=O2p,
            Op=Op,
            N2=bg.msise["N2"],
            O2=bg.msise["O2"],
            O=bg.msise["O"],
            H=bg.msise["H"],
            He=bg.msise["He"],
            date=date,
        )

    def calculate_FT_collision_frequency(self, frac: float = 1.0):
        """
        Friedrich-Tonker electron-neutral collision frequency.
        """
        logger.info(
            f"Compute Friedrich-Tonker electron-neutral collision frequency with a={frac}"
        )
        Te = np.clip(self.Te, 1.0, None)
        p = self.t_nn * self.Tn * pconst["boltz"]
        nu = (2.637e6 / np.sqrt(Te) + 4.945e5) * p
        return frac * nu

    def atmospheric_ion_neutral_collision_frequency(self):
        """
        Atmospheric ion-neutral collision frequency from total neutral density.
        """
        return 3.8e-11 * self.t_nn

    def calculate_SN_ei_collision_frequency(self, gamma: float = 0.5572, zi: int = 2):
        """
        Schunk-Nagy electron-ion collision frequency profile.
        """
        logger.warning("Compute Schunk-Nagy electron-ion collision frequency")

        e = pconst["q_e"]
        k = pconst["boltz"]
        me = pconst["m_e"]
        eps0 = pconst["eps0"]
        k_e = 1 / (4 * np.pi * eps0)

        Te = np.clip(self.Te, 1.0, None)
        Ti = np.clip(self.Ti, 1.0, None)
        Ne = np.clip(self.edens, 1e-12, None)

        # Input densities are expected in cm^-3; convert to m^-3 for Debye terms.
        Ne_m3 = Ne * 1e6

        for key, Ni in {"O2p": self.O2p, "Op": self.Op}.items():
            Ni = np.clip(Ni, 1e-12, None)
            Ni_m3 = Ni * 1e6

            ki2 = 4 * np.pi * Ni_m3 * e**2 * zi**2 * k_e / (k * Ti)
            ke2 = 4 * np.pi * Ne_m3 * e**2 * k_e / (k * Te)

            ki2 = np.clip(ki2, 1e-30, None)
            ke2 = np.clip(ke2, 1e-30, None)

            ke = np.sqrt(ke2)
            lam = np.log(4 * k * Te / (gamma**2 * zi * e**2 * k_e * ke)) - (
                ((ke2 + ki2) / ki2) * np.log(np.sqrt((ke2 + ki2) / ke2))
            )
            lam = np.clip(lam, 1e-6, None)

            nu_ei = (
                4
                * np.sqrt(2 * np.pi)
                * Ni
                * (zi * e**2 * k_e) ** 2
                * lam
                / (3 * np.sqrt(me) * (k * Te) ** 1.5)
            )
            setattr(self.collision.nu_sn.ei, key, nu_ei)

        self.collision.nu_sn.ei.total = (
            self.collision.nu_sn.ei.O2p + self.collision.nu_sn.ei.Op
        )
        self.collision.nu_sn.total += self.collision.nu_sn.ei.total

    def calculate_SN_en_collision_frequency(self):
        """
        Schunk-Nagy electron-neutral collision frequency profile.
        """
        logger.warning("Compute Schunk-Nagy electron-neutral collision frequency")

        Te = np.clip(self.Te, 1.0, None)
        sqrt_Te = np.sqrt(Te)

        self.collision.nu_sn.en.N2 = 1e-6 * 2.33e-11 * self.N2 * (1 - 1.12e-4 * Te) * Te
        self.collision.nu_sn.en.O2 = (
            1e-6 * 1.82e-10 * self.O2 * (1 + 3.6e-2 * sqrt_Te) * sqrt_Te
        )
        self.collision.nu_sn.en.O = (
            1e-6 * 8.9e-11 * self.O * (1 + 5.7e-4 * Te) * sqrt_Te
        )
        self.collision.nu_sn.en.He = 1e-6 * 4.6e-10 * self.He * sqrt_Te
        self.collision.nu_sn.en.H = (
            1e-6 * 4.5e-9 * self.H * (1 - 1.35e-4 * Te) * sqrt_Te
        )

        self.collision.nu_sn.en.total = (
            self.collision.nu_sn.en.N2
            + self.collision.nu_sn.en.O2
            + self.collision.nu_sn.en.O
            + self.collision.nu_sn.en.He
            + self.collision.nu_sn.en.H
        )
        self.collision.nu_sn.total += self.collision.nu_sn.en.total

    @staticmethod
    def atmospheric_collision_frequency(ni, nn, T):
        """
        Atmospheric collision profile from ion(electron)/neutral density and temperature.
        """
        na_profile = lambda T, nn: (1.8 * 1e-8 * nn * np.sqrt(T / 300))
        ni_profile = lambda T, ni: (6.1 * 1e-3 * ni * (300 / T) * np.sqrt(300 / T))
        return ni_profile(T, ni) + na_profile(T, nn)
