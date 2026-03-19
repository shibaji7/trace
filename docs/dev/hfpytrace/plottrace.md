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

## Earth-Curvature Arc Rendering (`oth=True`)

### `PlotRays` (2-D)

The ground arc is drawn using the **arc-length parameterization**:

```
x  = linspace(xlim[0], xlim[1], arc_samples)   # ground range [km]
y  = Re * (cos(x / Re) − 1)                     # = get_arc_heights(h=0, x)
```

This is exactly consistent with `get_arc_heights(height, dist)` which applies
`(Re + h) * cos(dist/Re) − Re` to ray altitude arrays.  Ray endpoints at `z=0`
land precisely on the drawn arc at any ground range, including multi-hop rays
that extend well beyond the nominal route length.

!!! note "Previous behaviour (fixed)"
    Before this fix the arc was drawn with the full-circle parameterization
    `x = Re·cos(θ)`, `y = Re·sin(θ) − Re`, whose x-axis convention (Cartesian
    circle coordinate) differs from the arc-length x used by `get_arc_heights`.
    The divergence grew to ~9 km at 2 000 km range, causing 2-hop ray endpoints
    to visually float above the ground arc.

### `PlotRays3D` / `PlotRays3DRouteFaces`

`_create_axis` already drew the arc with `get_arc_heights((x−center)*scale_km)`
and passed the same transformation to `_plot_face` and `_plot_rays_on_face`, so
no coordinate change was required.  The tick labels (`_draw_curved_ground_ticks`)
are computed in the same way.

## Multi-Hop Ray Visualization

Multi-hop rays produced by `RT3D.oblique_trace(nhops=N)` or
`RT2D.oblique_trace(nhops=N)` have their `x_km` / `y_km` (3D) or `x_km` / `z_km`
(2D) concatenated across all hop segments.  The plotting classes handle them
transparently — no code changes are needed on the plotting side.

Tips for clean multi-hop figures:

- **Extend `xlim`** to cover the maximum x reached by any ray.  For a 2-hop fan
  each ray can reach `2 × single_hop_range`.  `run_rt2d_multihop.py`,
  `run2D.py`, and `run_rt3d_multihop.py` all compute `x_max` from the rays
  automatically.
- **`ylim`** for `PlotRays` should remain `[-600, 700]` (or similar) when
  `oth=True` so that the curved-Earth arc and negative-y ray tails are visible.
- `nhops_completed` in the ray namespace can be used to colour or filter rays
  by the number of reflections they completed.

## API

::: hfpytrace.plottrace

## Source Code

```python title="hfpytrace/plottrace.py" linenums="1"
--8<-- "hfpytrace/plottrace.py"
```
