# PHaRLAP + IRI 3D Ray Trace

!!! warning "3D Geoplot3 Status"
    MATLAB `geoplot3` rendering is temporarily disabled in
    `examples/run_pharlap_iri_3d.py`.

<div class="hero">
  <h3>End-to-End 3D Workflow</h3>
  <p>Build 3D IRI density + 3D collision grids, run PHaRLAP 3D, and generate side/front ray-face visualizations.</p>
</div>

This page explains the example script:

- `examples/run_pharlap_iri_3d.py`

It is an end-to-end wrapper that builds volumetric ionosphere/background inputs, runs PHaRLAP 3D through MATLAB Engine, and generates:

1. a Python side/front ray-face plot over `ne_grid`

## PyIRI Backend Config

TRACE now uses `PyIRI` for IRI density fetch (`PyIRI.sh_library.IRI_density_1day`).
Use `iri_param` in `trace/cfg/config3D.json`:

```json
"iri_param": {
  "f107": 150.0,
  "foF2_coeff": "CCIR",
  "hmF2_model": "SHU2015",
  "coord": "GEO"
}
```

`iri_version` is deprecated and ignored by the current backend.

## Call Flow

1. `main()` parses CLI args (`--config`, `--event`, `--no-matlab`) and loads config.
2. `_run(...)` builds 3D spatial axes from `cfg.iono_grid`:
   - latitude vector
   - longitude vector
   - height vector
3. IRI electron density is fetched into a 3D cube:
   - `IRI3d.fetch_dataset(...)` -> `ne_grid(lat, lon, height)`
4. Collision frequency is computed on the same 3D cube:
   - `ComputeCollision.from_nrlmsise_3d(...)` -> `collision_freq(lat, lon, height)`
5. Geomagnetic grids are built from PyIRI IGRF:
   - `trace.geomag.build_geomag_grid(...)` -> `Bx/By/Bz(lat, lon, height)`
   - fallback to a uniform default field if geomag evaluation fails
6. Ray launch vectors are built from elevation + bearing fan:
   - `elevs`, `ray_bearings`, `freqs`
7. PHaRLAP is executed:
   - `Engine.run_pharlap_3d_sp(...)` when `use_spherical=true`
   - otherwise `Engine.run_pharlap_3d(...)`
8. Plot product is generated:
   - `_plot_ray_faces(...)` with `PlotRays3D`

## Key Code (From `run_pharlap_iri_3d.py`)

### 1) 3D Grid Construction

```python
iono_cfg = cfg.iono_grid
lats = _build_axis(float(iono_cfg.lat_start), float(iono_cfg.lat_step), int(iono_cfg.num_lats))
lons = _build_axis(float(iono_cfg.lon_start), float(iono_cfg.lon_step), int(iono_cfg.num_lons))
heights = _build_axis(
    float(iono_cfg.height_start_km),
    float(iono_cfg.height_step_km),
    int(iono_cfg.num_heights),
)
```

### 2) 3D IRI + 3D Collision

```python
ne_grid = _build_iri_3d(cfg, event_time, lats, lons, heights)
collision_freq = _build_collision_3d(
    cfg, event_time, ne_grid, lats, lons, heights
)
iono_en_grid_5 = ne_grid.copy()
```

Internally:

```python
iri = IRI3d(cfg, event_time)
ne_grid, _ = iri.fetch_dataset(event_time, lats, lons, heights, workers=int(cfg.worker))
```

and

```python
cc = ComputeCollision.from_nrlmsise_3d(
    date=event_time,
    lats=lats,
    lons=lons,
    heights_km=heights,
    Te=te_3d, Ti=ti_3d, edens=ne_grid, O2p=o2p_3d, Op=op_3d,
    workers=int(cfg.worker),
)
collision_freq = cc.collision.nu_ft
```

### 3) PHaRLAP 3D Call

```python
if bool(getattr(cfg, "use_spherical", True)):
    ray_data, ray_path_data, ray_state_vec = eng.run_pharlap_3d_sp(
        origin_lat=origin_lat, origin_lon=origin_lon, origin_ht=origin_ht,
        elevs=elevs, ray_bearings=ray_bearings, freqs=freqs,
        OX_mode=int(cfg.OX_mode), nhops=int(cfg.nhops), tol=float(cfg.threshold),
        rad_earth_m=float(cfg.radius_earth_m),
        iono_en_grid=ne_grid, iono_en_grid_5=iono_en_grid_5,
        collision_freq=collision_freq, iono_grid_parms=iono_grid_parms,
        Bx=Bx, By=By, Bz=Bz, geomag_grid_parms=geomag_grid_parms,
    )
else:
    ray_data, ray_path_data, ray_state_vec = eng.run_pharlap_3d(...)
```

### 4) Python Face Plot

```python
_plot_ray_faces(
    ne_grid=ne_grid, ray_path_data=ray_path_data,
    lats=lats, lons=lons, heights=heights,
    origin_lat=origin_lat, origin_lon=origin_lon,
    out_file=out_file,
)
```

## Run

From repository root:

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/run_pharlap_iri_3d.py
```

Custom timestamp:

```bash
python examples/run_pharlap_iri_3d.py --event 2017-05-27T16:00:00Z
```

Build-only (skip MATLAB raytrace):

```bash
python examples/run_pharlap_iri_3d.py --no-matlab
```

## Main Outputs

- `docs/examples/figures/pharlap_iri_3d_ray_faces.png`

## Rendered Figures

### Side/Front Faces (Python)
![PHaRLAP IRI 3D Faces](figures/pharlap_iri_3d_ray_faces.png)

## Related Files

- `examples/run_pharlap_iri_3d.py`
- `trace/pharlap.py`
- `trace/collision.py`
- `trace/density/iri.py`
- `trace/plottrace.py`

## See Also

- [PHaRLAP + IRI 2D Ray Trace](pharlap_iri.md)
