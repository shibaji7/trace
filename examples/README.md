# Examples: PHaRLAP + IRI

These examples use `trace/pharlap.py` directly.

## 1) Fetch and verify PHaRLAP assets

```bash
cd /home/chakras4/Research/CodeBase/trace/examples
python fetch_pharlap_lib.py
```

This triggers `trace.ensure_pharlap_lib()` and prints:
- cache root (`PHARLAP_LIB_PATH`)
- MATLAB library path resolved by `get_matlab_pharlap_lib(...)`

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

> Note: MATLAB `geoplot3` figure output is currently WIP and requires Mapping Toolbox with display-enabled MATLAB. In headless sessions, TRACE uses a fallback 3D `plot3` (ECEF) renderer.
