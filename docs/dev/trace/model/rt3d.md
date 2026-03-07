# `trace.model.rt3d`

<span class="api-badge api-package">Package</span>

Profile-first 3D scaffolding for ionospheric gridded workflows.

This module currently provides:

- 3D profile construction/validation (`RT3DProfile`)
- IRI / NRLMSISE / geomag fetch helpers on `(lat, lon, alt)` grids
- basic `RT3D` container initialization for future 3D tracing kernels

## Key Classes

<span class="api-badge api-class">Class</span> `RT3DProfile`  
<span class="api-badge api-class">Class</span> `RT3D`

## Key Methods

<span class="api-badge api-method">Method</span> `RT3DProfile.from_cfg()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.fetch_iri()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.fetch_msise()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.fetch_geomag()`  
<span class="api-badge api-method">Method</span> `RT3DProfile.force_zero_density_below()`

## API

::: trace.model.rt3d

## Source Code

```python title="trace/model/rt3d.py" linenums="1"
--8<-- "trace/model/rt3d.py"
```

