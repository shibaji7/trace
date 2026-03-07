# RT2D Spherical Oblique Rays from IRI

<div class="hero">
  <h3>2D Spherical Gradient Tracing Workflow</h3>
  <p>
    Build a route-based 2D IRI grid, run spherical oblique rays with
    <code>RT2D.oblique_trace(..., coordinate_system="spherical")</code>, and
    visualize spherical-ray outputs on the same route section.
  </p>
</div>

This page explains:

- `examples/run_rt2d_iri_spherical.py`

## Call Flow

1. `main()` parses CLI (`--config`, `--event`, `--mode`, `--formulation`, `--r-earth-km`).
2. `_run(...)` fetches IRI on a ground-referenced altitude grid through:
   - `RT2DProfile.from_cfg(..., fetch_iri=True)`
3. Density below config floor is zeroed:
   - `RT2DProfile.force_zero_density_below(cfg.start_height_km)`
4. `RT2D(profile=...)` initializes interpolation and refractive-index support.
5. `_trace_fan_spherical(...)` runs:
   - `oblique_trace(..., coordinate_system="spherical", r_earth_km=...)`
6. `_plot_density_and_rays(...)` overlays rays over the electron density field.

## Key Code

### 1) Spherical Tracer Call

```python
out = model.oblique_trace(
    freq_hz=float(f_mhz) * 1e6,
    elevation_deg=float(elev_deg),
    coordinate_system="spherical",
    x0_km=0.0,
    z0_km=float(heights_km[0]),
    mode=mode,
    formulation=formulation,
    r_earth_km=float(r_earth_km),
)
```

### 2) Plot Overlay

```python
p = PlotRays(oth=True, xlim=[0.0, 1500.0], ylim=[-100.0, y_max * 1.02], figsize=(7, 4))
p.set_density(X, Z, profile.ne_cm3, pf=None)
p.lay_rays(outputs=rays, kind="edens", lcolor="k", lw=0.6, param_alpha=0.85)
```

## Run

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/run_rt2d_iri_spherical.py
```

Custom options:

```bash
python examples/run_rt2d_iri_spherical.py \
  --config trace/cfg/config2D.json \
  --event 2017-05-27T16:00:00Z \
  --mode O \
  --formulation appleton-hartree \
  --r-earth-km 6371
```

## Output Figure

![RT2D IRI Spherical](figures/rt2d_iri_spherical_ray_paths.png)

## Related Files

- `examples/run_rt2d_iri_spherical.py`
- `trace/model/rt2d.py`
- `trace/density/iri.py`
- `trace/plottrace.py`

