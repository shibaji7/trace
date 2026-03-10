# `trace.model.rt3d`

<span class="api-badge api-package">Package</span>

Profile-first 3D ionospheric ray-tracing module for gridded workflows.

This module provides:

- 3D profile construction/validation (`RT3DProfile`)
- IRI / NRLMSISE / geomag fetch helpers on `(lat, lon, alt)` grids
- refractive-index volume construction from dispersion relations
- Cartesian and spherical oblique tracing (`gradient`, `hamiltonian`)
- helper interpolation/evaluation APIs used by solver kernels

## Key Classes

<span class="api-badge api-class">Class</span> `RT3DProfile`  
<span class="api-badge api-class">Class</span> `RT3D`

## Key Methods

<span class="api-badge api-method">Method</span> `RT3DProfile.from_cfg()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.fetch_iri()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.fetch_msise()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.fetch_geomag()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.force_zero_density_below()`
<span class="api-badge api-method">Method</span> `RT3D.build_refractive_index_interpolators()`  
<span class="api-badge api-method">Method</span> `RT3D.trace_cartesian_gradient()`  
<span class="api-badge api-method">Method</span> `RT3D.trace_cartesian_hamiltonian()`  
<span class="api-badge api-method">Method</span> `RT3D.trace_spherical_gradient()`  
<span class="api-badge api-method">Method</span> `RT3D.oblique_trace()`

## Runtime Notes

- `build_refractive_index_interpolators(...)` is the expensive preparation step.
- Solver calls use the prepared interpolators and return `SimpleNamespace` outputs
  with trajectory arrays and summary metrics (`group_path_km`, `group_delay_sec`,
  `status`, `reason`).
- Domain-edge and non-finite refractive-index handling are implemented in the
  Hamiltonian kernel to reduce NaN-driven failures.

## API

::: trace.model.rt3d

## Source Code

```python title="trace/model/rt3d.py" linenums="1"
--8<-- "trace/model/rt3d.py"
```
