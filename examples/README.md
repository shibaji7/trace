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
