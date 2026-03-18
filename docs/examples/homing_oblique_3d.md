# Oblique HF Link Homing — Homing3D + IRI

<div class="hero">
  <h3>3-D Lat/Lon Circle Homing on an IRI-2016 Volume</h3>
  <p>
    Find all (azimuth, elevation) pairs whose ray lands within a user-defined
    radius of a target receiver using <code>Homing3D</code>, then plot landing
    points on a map and build a synthetic oblique ionogram.
  </p>
</div>

This page explains:

- `examples/run_homing_oblique_3d.py`

## Overview

For an oblique HF link, finding the launch angle that hits a distant receiver is a
2-D homing problem in (azimuth, elevation) space.  `Homing3D` decomposes this into
independent 1-D elevation root-finds per azimuth slice, making the search tractable:

- For each azimuth `ψ`, sweep elevation `φ` and compute `D(φ,ψ)` = great-circle
  distance from the landing point to the target.
- Fit a spline to `D(φ)` and find elevations where `D ≤ tol_km` (entering/leaving
  the acceptance circle).
- Re-trace each accepted `(ψ, φ)` pair and collect the virtual height.

Multiple hits per azimuth correspond to different hop modes (1-hop, 2-hop, …) or
loop-like paths caused by ionospheric structure.

## Call Flow

1. `build_profile(...)` fetches a 3-D IRI volume covering the TX→RX bounding box
   plus a 5° margin (0.5° lat × 0.5° lon × 5 km altitude, 80–500 km):
   - `RT3DProfile(...)` constructs the lat/lon/alt grid.
   - `profile.fetch_iri(workers=N)` populates electron density.
2. `RT3D(profile=profile)` initialises the 3-D tracer.
3. `Homing3D` is configured with `HomingConfig`:
   - `tol_km=30`, `az_step_deg=5`, `elev_step_deg=3`.
4. `homing.home(freq_hz=f, target_lat=..., target_lon=...)` is called at each
   frequency.
5. Two figures are produced: a cartopy map of landing points and an h'-vs-f scatter.

## Key Code

### 1) Build 3-D Profile

```python
lats = np.arange(lat_lo, lat_hi + 0.5, 0.5)
lons = np.arange(lon_lo, lon_hi + 0.5, 0.5)
alts = np.arange(80.0, 501.0, 5.0)

profile = RT3DProfile(lats=lats, lons=lons, alts_km=alts, time=event_time)
profile.fetch_iri(workers=4)
model = RT3D(profile=profile)
```

### 2) Configure Homing

```python
from hfpytrace.homing import Homing3D, HomingConfig

cfg = HomingConfig(
    tol_km=30.0,
    az_min_deg=0.0, az_max_deg=360.0, az_step_deg=5.0,
    elev_min_deg=2.0, elev_max_deg=89.0, elev_step_deg=3.0,
    fine_points=1000, max_roots_per_az=5, mode="O",
)
homing = Homing3D(
    model,
    tx_lat=40.0, tx_lon=-95.0,
    config=cfg,
    coordinate_system="spherical",   # lat/lon output directly
    solver="gradient",
    trace_kw=dict(s_max_km=6000.0, max_step_km=5.0),
)
```

### 3) Home at One Frequency

```python
rays = homing.home(freq_hz=5e6, target_lat=45.0, target_lon=-90.0)
for r in rays:
    print(f"az={r.azimuth_deg:.0f}°  el={r.elevation_deg:.1f}°  "
          f"land=({r.landing_lat:.2f}°,{r.landing_lon:.2f}°)  "
          f"dist={r.dist_to_target_km:.1f} km  h'={r.virtual_height_km:.0f} km")
```

### 4) Full Synthetic Ionogram

```python
iono = homing.synthesize_ionogram(
    np.arange(3e6, 10e6, 0.2e6),
    target_lat=45.0, target_lon=-90.0,
)
# ndarray (N, 6): freq_hz | virtual_height_km | azimuth_deg |
#                 elevation_deg | landing_lat | landing_lon
```

## Outputs

| File | Description |
|---|---|
| `output/homing_3d_map.png` | Cartopy map: TX, RX acceptance circle, all homed landing points coloured by frequency |
| `output/homing_3d_ionogram.png` | Virtual height vs frequency, coloured by launch azimuth |

## Usage

```bash
# Default: TX=(40°N,95°W), RX=(45°N,85°W), tol=30 km, 2017-05-27T18:00
python examples/run_homing_oblique_3d.py

# Custom link
python examples/run_homing_oblique_3d.py \
    --date 2021-11-04T14:00 \
    --tx-lat 51.8 --tx-lon 103.1 \
    --rx-lat 55.0 --rx-lon 82.9 \
    --tol 25 --az-step 3 --el-step 2 \
    --fmin 4 --fmax 12 --fstep 0.1
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--date` | `2017-05-27T18:00` | ISO event time |
| `--tx-lat` / `--tx-lon` | `40.0` / `-95.0` | Transmitter [°N, °E] |
| `--rx-lat` / `--rx-lon` | `45.0` / `-85.0` | Target receiver [°N, °E] |
| `--tol` | `30.0` | Acceptance radius [km] |
| `--fmin` | `3.0` | Minimum frequency [MHz] |
| `--fmax` | `10.0` | Maximum frequency [MHz] |
| `--fstep` | `0.2` | Frequency step [MHz] |
| `--az-step` | `5.0` | Azimuth sweep step [°] |
| `--el-step` | `3.0` | Elevation sweep step [°] |
| `--out` | `./output` | Output directory |
| `--workers` | `4` | IRI fetch threads |

## Tuning Tips

| Situation | Recommendation |
|---|---|
| Slow runtime | Increase `--az-step` and `--el-step`; reduce `--fstep` |
| Missing paths | Decrease `--tol` only after confirming the profile covers the path |
| Unexpected hits far from RX | Decrease `--tol` or narrow the azimuth range with `HomingConfig` overrides |
| Only 1-hop visible | Increase `--fmax`; 2-hop paths appear at lower frequencies |

## Coordinate System Choice

`coordinate_system="spherical"` (default) returns `lat_deg`/`lon_deg` arrays
directly from `RT3D.trace_spherical_gradient`.  Use `"cartesian"` if your profile
grid is defined in flat-Earth (x, y) coordinates — the homing module will convert
landing (x, y) offsets to (lat, lon) using the transmitter location.

## See Also

- [`hfpytrace.homing`](../dev/hfpytrace/homing.md) — full API reference
- [`hfpytrace.model.rt3d`](../dev/hfpytrace/model/rt3d.md) — RT3D tracer
- [RT3D IRI Cartesian Example](rt3d_iri_cartesian.md) — oblique fan tracing
- [NVIS Homing2D Example](homing_nvis_2d.md) — 2-D vertical incidence
