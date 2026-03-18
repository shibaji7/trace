# `hfpytrace.model.rt2d`

<span class="api-badge api-package">Package</span>

Class-based 2D ray tracing utilities for profile-driven workflows, refractive-index
interpolation, and oblique tracing in Cartesian or spherical coordinates.

## Key Classes

<span class="api-badge api-class">Class</span> `RT2DConfig`  
<span class="api-badge api-class">Class</span> `RT2DProfile`  
<span class="api-badge api-class">Class</span> `RT2D`

## Key Methods

<span class="api-badge api-method">Method</span> `RT2D.build_refractive_index_interpolators()`  
<span class="api-badge api-method">Method</span> `RT2D.trace_cartesian_gradient()`  
<span class="api-badge api-method">Method</span> `RT2D.trace_spherical_gradient()`  
<span class="api-badge api-method">Method</span> `RT2D.oblique_trace()`  
<span class="api-badge api-method">Method</span> `RT2D.trace()`
<span class="api-badge api-method">Method</span> `RT2D.trace_fan()`
<span class="api-badge api-method">Method</span> `RT2DProfile.from_cfg()`
<span class="api-badge api-method">Method</span> `RT2DProfile.fetch_iri()`
<span class="api-badge api-method">Method</span> `RT2DProfile.force_zero_density_below()`

## API

::: hfpytrace.model.rt2d

## Source Code

```python title="hfpytrace/model/rt2d.py" linenums="1"
--8<-- "hfpytrace/model/rt2d.py"
```

## Collision Frequency Support

`RT2DProfile` and `RT2D` support user-defined collision frequency models that feed directly into the Appleton-Hartree or Sen-Wyller dispersion backends.

### Workflow

```python
from hfpytrace.model.rt2d import RT2D, RT2DProfile

# Build profile with MSIS neutral background
prof = RT2DProfile.from_cfg(cfg, fetch_iri=True, fetch_msise=True)

# Compute all collision models at once (uses Tn, Ne defaults)
rt = RT2D(profile=prof)
rt.fetch_collision()                    # stores ComputeCollision on prof.collision

# Trace with a named collision model
ray = rt.oblique_trace(
    freq_hz=10.5e6,
    elevation_deg=25.0,
    collision_type="SN",                # Schunk-Nagy full (en + ei)
)
```

### Supported `collision_type` Keys

| Key | Model |
|-----|-------|
| `"FT"` | Friedrich-Tonker electron-neutral (ν_ft, a=1.0) |
| `"FT_cc"` | Friedrich-Tonker (ν_av_cc, a=2.5) |
| `"FT_mb"` | Friedrich-Tonker (ν_av_mb, a=1.5) |
| `"SN_en"` | Schunk-Nagy electron-neutral total |
| `"SN_ei"` | Schunk-Nagy electron-ion total |
| `"SN"` | Schunk-Nagy full (en + ei) |
| `"atm"` | Atmospheric ion-neutral approximation |

### Custom Plasma State

```python
# Supply your own Te, Ti, ion fractions
rt.fetch_collision(
    Te=Te_array,   # shape (nz, nx), K
    Ti=Ti_array,
    Op=Op_array,   # O+ density in cm^-3
    O2p=O2p_array,
)
```

### Direct Array

You may also pass `collision_hz` directly as a 2D array (nz, nx) without calling `fetch_collision()`:

```python
ray = rt.oblique_trace(
    freq_hz=10.5e6,
    elevation_deg=25.0,
    collision_hz=my_nu_array,
)
```

`collision_hz` and `collision_type` are mutually exclusive.
