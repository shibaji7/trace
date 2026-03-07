# RT2D Cartesian Oblique Rays from IRI

<div class="hero">
  <h3>2D Cartesian Gradient Tracing Workflow</h3>
  <p>
    Build a route-based 2D IRI electron-density grid, run Cartesian oblique rays with
    <code>RT2D.oblique_trace(..., coordinate_system="cartesian")</code>, and overlay
    rays on the density field.
  </p>
</div>

This page explains:

- `examples/run_rt2d_iri_cartesian.py`

## Call Flow

1. `main()` parses CLI (`--config`, `--event`, `--mode`, `--formulation`).
2. `_run(...)` builds a ground-referenced altitude grid and fetches IRI through:
   - `RT2DProfile.from_cfg(..., fetch_iri=True)`
3. Density below config floor is zeroed by:
   - `RT2DProfile.force_zero_density_below(cfg.start_height_km)`
4. `RT2D(profile=...)` initializes the 2D tracer.
5. `_trace_fan_cartesian(...)` runs a fan of rays using:
   - `oblique_trace(..., coordinate_system="cartesian")`
6. `_plot_density_and_rays(...)` draws electron density + rays using `PlotRays`.

## Key Code

### 1) Build Profile and Apply Lower-Altitude Mask

```python
profile = RT2DProfile.from_cfg(
    cfg=cfg,
    time=event_time,
    alt_km=alt_km,
    fetch_iri=True,
    fetch_msise=False,
    fetch_geomag=False,
    workers=workers,
)
n_rows = profile.force_zero_density_below(float(cfg.start_height_km))
model = RT2D(profile=profile)
```

### 2) Cartesian Oblique Fan Tracing

```python
out = model.oblique_trace(
    freq_hz=float(f_mhz) * 1e6,
    elevation_deg=float(elev_deg),
    coordinate_system="cartesian",
    x0_km=0.0,
    z0_km=float(heights_km[0]),
    mode=mode,
    formulation=formulation,
)
```

### 3) Density + Rays Plot

```python
p = PlotRays(oth=True, xlim=[0.0, 1500.0], ylim=[-100.0, y_max * 1.02], figsize=(7, 4))
p.set_density(X, Z, profile.ne_cm3, pf=None)
p.lay_rays(outputs=rays, kind="edens", lcolor="k", lw=0.6, param_alpha=0.85)
```

## Run

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/run_rt2d_iri_cartesian.py
```

Custom options:

```bash
python examples/run_rt2d_iri_cartesian.py \
  --config trace/cfg/config2D.json \
  --event 2017-05-27T16:00:00Z \
  --mode O \
  --formulation appleton-hartree
```

## Output Figure

![RT2D IRI Cartesian](figures/rt2d_iri_cartesian_ray_paths.png)

## Related Files

- `examples/run_rt2d_iri_cartesian.py`
- `trace/model/rt2d.py`
- `trace/density/iri.py`
- `trace/plottrace.py`

