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
