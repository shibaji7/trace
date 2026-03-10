
# hfpytrace
<div style="text-align: center;">
  <!-- <img src="docs/assets/Colab-pynasonde-logo2.jpg" alt="Pynasonde" width="50%"> -->
</div>

[![License: MIT](https://img.shields.io/badge/License%3A-MIT-green)](https://choosealicense.com/licenses/mit/) 
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) 
![GitHub Stable Release (latest by date)](https://img.shields.io/github/v/release/shibaji7/trace)
[![Documentation Status](https://img.shields.io/readthedocs/pytrace?logo=readthedocs&label=docs)](https://pytrace.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/shibaji7/trace/branch/main/graph/badge.svg)](https://codecov.io/gh/shibaji7/trace)


Pynasonde is an open-source Python-based application designed for precision ionospheric radio sounding, with a strong focus on analyzing the phase characteristics of radio echoes. Tailored for Space Weather applications, Pynasonde offers a suite of unique tools that help extract valuable insights directly and autonomously from ionogram data.

With Pynasonde, you can achieve accurate echo recognition, noise discrimination, and echo classification into traces. The application also facilitates the scaling of standard ionospheric parameters, 3-D plasma density inversion (including error bars), diagnostics of small-scale irregularities, and determination of vector velocities.

Designed as a comprehensive toolbox, Pynasonde empowers researchers to process raw ionosonde datasets efficiently, providing reliable, real-time insights into ionospheric conditions and phenomena. Whether you're focused on space weather forecasting, radio communication, or scientific exploration, Pynasonde is your go-to tool for precision ionospheric analysis.

## Source Code 

The library source code can be found on the [trace GitHub](https://github.com/shibaji7/trace) repository. 

If you have any questions or concerns please submit an **Issue** on the [trace GitHub](https://github.com/shibaji7/trace) repository. 

## Documentation
Read the docs: https://pytrace.readthedocs.io/en/latest/

## Example Highlights

### RT2D IRI Cartesian Oblique Rays

- Script: `examples/run_rt2d_iri_cartesian.py`
- Docs: https://pytrace.readthedocs.io/en/latest/examples/rt2d_iri_cartesian/

![RT2D IRI Cartesian](docs/examples/figures/rt2d_iri_cartesian_ray_paths.png)

### RT2D IRI Spherical Oblique Rays

- Script: `examples/run_rt2d_iri_spherical.py`
- Docs: https://pytrace.readthedocs.io/en/latest/examples/rt2d_iri_spherical/

![RT2D IRI Spherical](docs/examples/figures/rt2d_iri_spherical_ray_paths.png)

## IRI Backend Note

`hfpytrace.density.iri` now uses `PyIRI` (`PyIRI.sh_library.IRI_density_1day`) for
IRI electron density fetch.

Set runtime knobs in `iri_param` (JSON config):

```json
"iri_param": {
  "f107": 150.0,
  "foF2_coeff": "CCIR",
  "hmF2_model": "SHU2015",
  "coord": "GEO"
}
```

`iri_version` is deprecated and no longer used.
