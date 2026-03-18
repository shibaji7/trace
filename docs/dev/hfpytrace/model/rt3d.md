# `hfpytrace.model.rt3d`

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

::: hfpytrace.model.rt3d

## Source Code

```python title="hfpytrace/model/rt3d.py" linenums="1"
--8<-- "hfpytrace/model/rt3d.py"
```

## Collision Frequency Support

`RT3DProfile` and `RT3D` mirror the same collision API as RT2D, operating on 3D fields of shape `(nlat, nlon, nalt)`.

### Workflow

```python
from hfpytrace.model.rt3d import RT3D, RT3DProfile

prof = RT3DProfile.from_cfg(cfg, fetch_iri=True, fetch_msise=True)
rt = RT3D(profile=prof)
rt.fetch_collision()                    # stores ComputeCollision on prof.collision

# Extract a specific model's nu array for downstream use
nu_3d = RT3D._extract_collision_hz(prof.collision, "FT")
# nu_3d.shape == (nlat, nlon, nalt)
```

### Supported `collision_type` Keys

| Key | Model |
|-----|-------|
| `"FT"` | Friedrich-Tonker (ν_ft, a=1.0) |
| `"FT_cc"` | Friedrich-Tonker (ν_av_cc, a=2.5) |
| `"FT_mb"` | Friedrich-Tonker (ν_av_mb, a=1.5) |
| `"SN_en"` | Schunk-Nagy electron-neutral total |
| `"SN_ei"` | Schunk-Nagy electron-ion total |
| `"SN"` | Schunk-Nagy full (en + ei) |
| `"atm"` | Atmospheric ion-neutral approximation |

### Custom Plasma State

```python
rt.fetch_collision(
    Te=Te_3d,    # shape (nlat, nlon, nalt), K
    Ti=Ti_3d,
    Op=Op_3d,    # O+ density in cm^-3
    O2p=O2p_3d,
)
```
