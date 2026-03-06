# Examples: PHaRLAP + PyIRI

These examples use `trace/pharlap.py` directly.

## 1) Fetch and verify PHaRLAP assets

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python fetch_pharlap_lib.py
```

This triggers `trace.ensure_pharlap_lib()` and prints:
- cache root (`PHARLAP_LIB_PATH`)
- MATLAB library path resolved by `get_matlab_pharlap_lib(...)`

## PyIRI Configuration

IRI fetch now uses `PyIRI` (via `PyIRI.sh_library.IRI_density_1day`).
Set model knobs in `iri_param` inside `config2D.json` / `config3D.json`:

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
python run_pharlap_iri.py --config ../trace/config2D.json --no-matlab
```

`--no-matlab` validates the Python-side setup only (IRI fetch + PHaRLAP input arrays).

To call MATLAB engine through `Engine.run_pharlap(...)`, remove `--no-matlab`:

```bash
python run_pharlap_iri.py --config ../trace/config2D.json
```

## 3) Build 3D IRI/collision grids and run PHaRLAP 3D

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python run_pharlap_iri_3d.py --config ../trace/config3D.json --no-matlab
```

To run MATLAB engine via `Engine.run_pharlap_3d_sp(...)` or `Engine.run_pharlap_3d(...)`:

```bash
python run_pharlap_iri_3d.py --config ../trace/config3D.json
```

> Note: MATLAB `geoplot3` figure output is temporarily disabled in `run_pharlap_iri_3d.py`.

## 4) RT Model examples (RT1D / RT2D)

These scripts are pure-Python examples inspired by the notebook workflows, rewritten for
the `trace.model` class APIs and TRACE-native IRI/collision inputs.

### 4.1 Vertical forward operator (1D)

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python rtmodel_virtual_height_demo.py --cfg ../trace/config2D.json --plot
```
Use `--synthetic` to skip IRI/collision and use an analytic test profile.

### 4.2 Cartesian / spherical Snell tracing (1D)

```bash
python rtmodel_cartesian_snell_demo.py --cfg ../trace/config2D.json --plot
python rtmodel_spherical_snell_demo.py --cfg ../trace/config2D.json --plot
```

### 4.3 Cartesian gradient tracing and verification (2D)

```bash
python rtmodel_cartesian_gradient_demo.py --cfg ../trace/config2D.json --plot
python rtmodel_raytrace_verification_demo.py --cfg ../trace/config2D.json --plot
```
