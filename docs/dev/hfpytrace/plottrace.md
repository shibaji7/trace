# `hfpytrace.plottrace`

<span class="api-badge api-package">Package</span>

Visualization helpers for density fields and ray-path overlays.

The module now exposes most plotting style/layout constants as optional user
inputs instead of hardcoded internals. Existing defaults are preserved, but the
caller can override style, DPI, face colors, arc rendering, ray stroke styling,
colorbar placement, and 3-panel spacing from Python.

## Key Classes

<span class="api-badge api-class">Class</span> `PlotRays`
<span class="api-badge api-class">Class</span> `PlotRays3D`
<span class="api-badge api-class">Class</span> `PlotRays3DRouteFaces`
<span class="api-badge api-class">Class</span> `MatlabGeoPlot3D`

## Key Methods

<span class="api-badge api-method">Function</span> `setup()`  
<span class="api-badge api-method">Method</span> `PlotRays.set_density()`  
<span class="api-badge api-method">Method</span> `PlotRays.lay_rays()`  
<span class="api-badge api-method">Method</span> `PlotRays.save()`
<span class="api-badge api-method">Method</span> `PlotRays3D.plot_faces()`  
<span class="api-badge api-method">Method</span> `PlotRays3DRouteFaces.plot_route_faces()`  
<span class="api-badge api-method">Method</span> `MatlabGeoPlot3D.plot_rays()`

## Notes

- `setup(...)` accepts optional style/font/grid/DPI controls and keeps the
  current SciencePlots-like defaults when omitted.
- `PlotRays(...)` accepts optional figure/save/arc/ground/tick styling inputs.
- `PlotRays3D(...)` accepts optional 3D face styling inputs, including:
  `top_aspect`, `panel_wspace`, `cbar_after_panel`, ray stroke colors, and
  top-panel appearance.
- `PlotRays3DRouteFaces.plot_route_faces(...)` renders a 3-panel route-aligned
  layout:
  1. along-track vs height
  2. bearing-spread vs height
  3. top view (along-track vs bearing-spread)
- In 3-panel mode, the default colorbar anchor is after panel 2.

!!! warning "3D Geoplot3 Status"
    `MatlabGeoPlot3D` is **WIP**. Full `geoplot3` output requires Mapping Toolbox and display support. A headless `plot3` ECEF fallback is implemented for non-display environments.

## API

::: hfpytrace.plottrace

## Source Code

```python title="hfpytrace/plottrace.py" linenums="1"
--8<-- "hfpytrace/plottrace.py"
```
