# Installation

<div class="hero">
  <h3>Install `hfpytrace` in a clean Python 3.11 environment</h3>
  <p>
    Recommended path: create a virtual environment, install from PyPI, then validate with an example run.
  </p>
</div>

!!! warning "Beta Installation Notes (Updated March 3, 2026)"
    Installation steps and dependencies can change while the project is in beta.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install hfpytrace
```

## Environment Requirements

- Python: `>=3.11`
- pip / virtualenv (or conda)
- Optional for maps: `cartopy>=0.19`

## Recommended Workflows

### Pip + venv

```bash
python3.11 -m venv trace-env
source trace-env/bin/activate
pip install hfpytrace
```

### Conda

```bash
conda create -n trace-env python=3.11 -y
conda activate trace-env
pip install hfpytrace
```

## Developer Install

From repository root:

```bash
pip install -e .[dev]
```

## Dependencies

Core dependencies are declared in `setup.py` and include scientific/data libraries such as `numpy`, `scipy`, `matplotlib`, `pandas`, `xarray`, and `nrlmsise00`.

## Notes

- Upgrade existing install: `pip install --upgrade hfpytrace`
- If cartopy install fails, use your OS-specific geospatial package prerequisites first.
- Installed package includes default configs in `trace/cfg/config2D.json` and `trace/cfg/config3D.json`.
