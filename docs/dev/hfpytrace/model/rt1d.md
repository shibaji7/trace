# `hfpytrace.model.rt1d`

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

::: hfpytrace.model.rt1d

## Source Code

```python title="hfpytrace/model/rt1d.py" linenums="1"
--8<-- "hfpytrace/model/rt1d.py"
```

## Collision Frequency Support

`RT1DProfile` and `RT1D` support user-defined collision frequency models.

### Workflow

```python
from hfpytrace.model.rt1d import RT1D

rt = RT1D(cfg=cfg, fetch_iri=True, fetch_msise=True)
rt.fetch_collision()                    # compute all models; store on profile.collision

result = rt.NVIS_tracer(
    freq_mhz=freqs,
    mode="O",
    collision_type="SN",                # Schunk-Nagy full (en + ei)
)
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
    Te=Te_array,  # shape (nz,), K
    Ti=Ti_array,
    Op=Op_array,  # cm^-3
    O2p=O2p_array,
)
```

`collision_hz` (direct array) and `collision_type` (named model) are mutually exclusive in `NVIS_tracer`.
