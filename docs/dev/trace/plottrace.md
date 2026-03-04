# `trace.plottrace`

<span class="api-badge api-package">Package</span>

Visualization helpers for density fields and ray-path overlays.

## Key Classes

<span class="api-badge api-class">Class</span> `PlotRays`
<span class="api-badge api-class">Class</span> `PlotRays3D`
<span class="api-badge api-class">Class</span> `MatlabGeoPlot3D`

## Key Methods

<span class="api-badge api-method">Method</span> `PlotRays.set_density()`  
<span class="api-badge api-method">Method</span> `PlotRays.lay_rays()`  
<span class="api-badge api-method">Method</span> `PlotRays.save()`

!!! warning "3D Geoplot3 Status"
    `MatlabGeoPlot3D` is **WIP**. Full `geoplot3` output requires Mapping Toolbox and display support. A headless `plot3` ECEF fallback is implemented for non-display environments.

## API

::: trace.plottrace

## Source Code

```python title="trace/plottrace.py" linenums="1"
--8<-- "trace/plottrace.py"
```
