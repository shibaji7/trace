# `hfpytrace.density.sami`

<span class="api-badge api-package">Package</span>

SAMI3 model adapter for 2D and 3D electron density grid workflows.

## Key Classes

<span class="api-badge api-class">Class</span> `SAMI3`

## Key Methods

<span class="api-badge api-method">Method</span> `SAMI3.fetch_dataset()`
<span class="api-badge api-method">Method</span> `SAMI3.fetch_dataset_3d()`
<span class="api-badge api-method">Method</span> `SAMI3._bilinear_ne_profile()`

## Implementation Notes

### Spatial interpolation

`fetch_interpolated_data` and `_fetch_profile_at_index` use **bilinear horizontal
interpolation** (`_bilinear_ne_profile`) over the four surrounding 1° grid cells
instead of nearest-neighbour selection. Longitude wraparound at 0°/360° is handled
automatically.

### Temporal interpolation

`fetch_dataset` and `fetch_dataset_3d` perform **linear interpolation** between the
two bracketing SAMI3 time frames using

```
α  = (t − t_i) / (t_j − t_i)
Ne = (1 − α) · Ne(t_i) + α · Ne(t_j)
```

The denominator uses the actual bracket width `t_j − t_i`, so non-uniform output
cadences are handled correctly.

## API

::: hfpytrace.density.sami

## Source Code

```python title="hfpytrace/density/sami.py" linenums="1"
--8<-- "hfpytrace/density/sami.py"
```
