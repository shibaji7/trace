"""hfpytrace.model — ionospheric ray-tracing model classes.

This sub-package contains the core ray-tracing and dispersion models:

RT1D
    1-D vertical-incidence tracer using a simple ODE integrator.
RT2D / RT2DProfile / RT2DConfig
    2-D great-circle ray-tracing with a configurable ionospheric profile
    (IRI, SAMI3, GEMINI, etc.).
RT3D / RT3DProfile
    3-D oblique ray-tracing via PHaRLAP (``raytrace_3d`` or
    ``raytrace_3d_sp``).
DispersionResult
    Container for refractive index, absorption, and related propagation metrics.
AppletonHartreeDispersion
    Appleton-Hartree magneto-ionic dispersion relation.
SenWyllerDispersion
    Sen-Wyller generalized dispersion relation (includes electron collision
    frequency via a non-Maxwellian velocity distribution).
"""

from loguru import logger

from .dispersion import AppletonHartreeDispersion, DispersionResult, SenWyllerDispersion
from .rt1d import RT1D, RT1DProfile
from .rt2d import RT2D, RT2DConfig, RT2DProfile
from .rt3d import RT3D, RT3DProfile

__all__ = [
    "RT1D",
    "RT1DProfile",
    "RT2D",
    "RT2DProfile",
    "RT2DConfig",
    "RT3D",
    "RT3DProfile",
    "DispersionResult",
    "AppletonHartreeDispersion",
    "SenWyllerDispersion",
]

logger.debug("hfpytrace.model namespace imported")
