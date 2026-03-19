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

## Multi-Hop Ground Reflections

`RT3D.oblique_trace` accepts an `nhops` keyword (default `1`) to model multiple
ionospheric reflections in a single call.

### Algorithm

For each hop beyond the first:

1. The ODE is restarted at the **domain left edge** (`x=0, y=0`) so the full
   n-field grid is available (horizontal-homogeneity assumption).
2. Output `x_km` / `y_km` are **offset** by the accumulated physical ground-hit
   position of the previous hop so that concatenated segments form a continuous path.
3. **Specular reflection** geometry is applied at each ground hit:
   - Cartesian: `vz → −vz`, azimuth = `atan2(vy, vx)` (unchanged)
   - Spherical: `vr → −vr`, azimuth = `atan2(vlon, vlat)` (unchanged)

### Output namespace additions

| Attribute | Type | Description |
|---|---|---|
| `nhops_completed` | `int` | Actual hops traced (≤ `nhops`) |
| `group_path_km` | `float` | Accumulated across all hops |
| `group_delay_sec` | `float` | Accumulated across all hops |

### Usage

```python
# 2-hop trace (one ground reflection)
out = rt.oblique_trace(
    freq_hz=8e6,
    elevation_deg=30.0,
    azimuth_deg=45.0,
    coordinate_system="cartesian",
    nhops=2,
    x0_km=0.0, y0_km=0.0, z0_km=0.0,
    s_max_km=2000.0,
)
print(out.nhops_completed, out.group_path_km)

# 3-hop spherical trace
out3 = rt.oblique_trace(
    freq_hz=8e6,
    elevation_deg=25.0,
    coordinate_system="spherical",
    nhops=3,
    s_max_km=4000.0,
)
```

### Notes

- If the ray does not reach the ground on hop *k* (penetrates, hits domain edge,
  or runs out of `s_max_km`), the loop stops and `nhops_completed < nhops`.
- `s_max_km` applies independently to **each hop** — increase it proportionally
  when requesting multiple hops.
- The Hamiltonian solver supports `nhops` for the cartesian coordinate system.
  Spherical Hamiltonian falls back to the gradient solver automatically.

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
