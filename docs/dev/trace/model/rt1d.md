# `trace.model.rt1d`

<span class="api-badge api-package">Package</span>

Lean 1D profile API for single-point altitude workflows.

Recent updates include:

- `NVIS_tracer(...)` support for stretched nonuniform vertical regridding:
  - `use_nonuniform_grid`
  - `nonuniform_points`
  - `nonuniform_sharpness`
- tighter tracer behavior for contiguous propagation segments
- `loguru` diagnostics in initialization, fetch, and tracer execution paths

## Key Classes

<span class="api-badge api-class">Class</span> `RT1DProfile`  
<span class="api-badge api-class">Class</span> `RT1D` (compatibility shim)

## Key Methods

<span class="api-badge api-method">Method</span> `RT1DProfile.from_cfg()`  
<span class="api-badge api-method">Method</span> `RT1DProfile.fetch_iri()`  
<span class="api-badge api-method">Method</span> `RT1DProfile.fetch_msise()`  
<span class="api-badge api-method">Method</span> `RT1DProfile.fetch_geomag()`  
<span class="api-badge api-method">Method</span> `RT1DProfile.den_to_plasma_freq_hz()`  
<span class="api-badge api-method">Method</span> `RT1DProfile.plasma_freq_to_den()`  
<span class="api-badge api-method">Method</span> `RT1DProfile.inclination_to_vertical_angle()`
<span class="api-badge api-method">Method</span> `RT1D.NVIS_tracer()`

## NVIS Tracer Notes

`RT1D.NVIS_tracer(...)` is the default 1D vertical-forward-style tracer used by
the examples. It returns:

- `vh_km`
- `turning_height_km`
- `n_profile`
- `reason`

For smoother curves near reflection regions, enable nonuniform regridding
(enabled by default in current examples).

## API

::: trace.model.rt1d

## Source Code

```python title="trace/model/rt1d.py" linenums="1"
--8<-- "trace/model/rt1d.py"
```
