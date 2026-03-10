# Examples: PHaRLAP + PyIRI

These examples use `hfpytrace/pharlap.py` directly.

## 1) Fetch and verify PHaRLAP assets

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python fetch_pharlap_lib.py
```

This triggers `hfpytrace.ensure_pharlap_lib()` and prints:
- cache root (`PHARLAP_LIB_PATH`)
- MATLAB library path resolved by `get_matlab_pharlap_lib(...)`

## PyIRI Configuration

IRI fetch now uses `PyIRI` (via `PyIRI.sh_library.IRI_density_1day`).
Set model knobs in `iri_param` inside `hfpytrace/cfg/config2D.json` / `hfpytrace/cfg/config3D.json`:

```json
"iri_param": {
  "f107": 150.0,
  "foF2_coeff": "CCIR",
  "hmF2_model": "SHU2015",
  "coord": "GEO"
}
```

`iri_version` is no longer used by the backend.

## 2) Build IRI grid and run PHaRLAP wrapper

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python run_pharlap_iri.py --no-matlab
```

`--no-matlab` validates the Python-side setup only (IRI fetch + PHaRLAP input arrays).

To call MATLAB engine through `Engine.run_pharlap(...)`, remove `--no-matlab`:

```bash
python run_pharlap_iri.py
```

## 3) Build 3D IRI/collision grids and run PHaRLAP 3D

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python run_pharlap_iri_3d.py --no-matlab
```

To run MATLAB engine via `Engine.run_pharlap_3d_sp(...)` or `Engine.run_pharlap_3d(...)`:

```bash
python run_pharlap_iri_3d.py
```

To use a custom config path:

```bash
python run_pharlap_iri.py --config /absolute/path/config2D.json
python run_pharlap_iri_3d.py --config /absolute/path/config3D.json
```

> Note: MATLAB `geoplot3` figure output is temporarily disabled in `run_pharlap_iri_3d.py`.

## 4) RT Model examples (RT1D / RT2D)

These scripts are pure-Python examples inspired by the notebook workflows, rewritten for
the `hfpytrace.model` class APIs and TRACE-native IRI/collision inputs.

### 4.1 1D NVIS O/X tracer from IRI (config1D)

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python rtmodel_nvis_ox_iri_1d.py --fmin 1 --fmax 12 --nfreq 111
```

Uses `hfpytrace/cfg/config1D.json`, builds a 1D IRI profile, then runs:
- `RT1D.NVIS_tracer(..., mode="O")`
- `RT1D.NVIS_tracer(..., mode="X")`

Output figure default:
- `docs/examples/figures/rt1d_nvis_ox_iri.png`

### 4.2 1D O-mode: Appleton vs Sen-Wyller (3 frequencies)

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python rtmodel_omode_appleton_sw_demo.py --freqs 3,5,7
```

Compares O-mode outputs from:
- `formulation="appleton"`
- `formulation="senwyller"`

Output figure default:
- `docs/examples/figures/rt1d_omode_appleton_vs_sw.png`

### 4.3 1D O-mode Appleton absorption (dB/km) from IRI

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python rtmodel_omode_absorption_appleton_iri_1d.py
```

Three custom frequencies (default includes 30 MHz):

```bash
python rtmodel_omode_absorption_appleton_iri_1d.py --freqs 15,22,30
```

Output figure default:
- `docs/examples/figures/rt1d_omode_absorption_appleton_iri.png`

Figure includes:
- left panel: absorption [dB/km] vs altitude
- right panel: plasma frequency and collision frequency [MHz] vs altitude

### 4.4 RT2D IRI Cartesian oblique rays (2D)

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python run_rt2d_iri_cartesian.py
```

Output figure:
- `docs/examples/figures/rt2d_iri_cartesian_ray_paths.png`

### 4.5 RT2D IRI spherical oblique rays (2D)

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python run_rt2d_iri_spherical.py
```

Output figure:
- `docs/examples/figures/rt2d_iri_spherical_ray_paths.png`
