# NVIS Ionogram Synthesis — Homing2D + IRI

<div class="hero">
  <h3>Vertical-Incidence Homing on a 2-D IRI Profile</h3>
  <p>
    Find all ray paths that return to the ionosonde (ground range = 0) using
    <code>Homing2D</code>, then assemble a synthetic ionogram and overlay homed
    rays on the IRI electron-density cross-section.
  </p>
</div>

This page explains:

- `examples/run_homing_nvis_2d.py`

## Overview

Vertical-incidence sounding probes the ionosphere directly overhead by transmitting
upward and recording the group delay of any ray that returns to the transmitter.  For
a horizontally stratified ionosphere, only the exactly-vertical ray returns; in the
presence of tilts or travelling ionospheric disturbances (TIDs), oblique paths also
return and produce additional traces or cusps on the ionogram (Laryunin 2025).

`Homing2D` automates the search: for each operating frequency it sweeps elevation,
fits a spline to `D(φ) = ground_range(φ)`, and finds every elevation that drives
`D` to zero (or within `tol_km` of zero).  Multiple roots = multipath.

## Call Flow

1. `build_profile(...)` fetches a short meridional IRI-2016 slice (±3° latitude,
   60–500 km altitude):
   - `RT2DProfile(...)` constructs the grid.
   - `profile.fetch_iri(workers=N)` populates electron density.
2. `RT2D(profile=profile)` initialises the 2-D tracer.
3. `Homing2D` is configured with `HomingConfig`:
   - `tol_km=15`, `elev_step_deg=2`, `elev_max_deg=89`.
4. `homing.home(freq_hz=f)` is called at each frequency in `[fmin, fmax]`.
5. Results are collected into an ionogram array and two figures are saved.

## Key Code

### 1) Build Profile

```python
alt_km  = np.arange(60.0, 501.0, 2.0)
lats    = np.linspace(center_lat - 3.0, center_lat + 3.0, 61)
lons    = np.full(61, center_lon)
profile = RT2DProfile(alt_km=alt_km, lats=lats, lons=lons, time=event_time)
profile.fetch_iri(workers=4)
model   = RT2D(profile=profile)
```

### 2) Configure Homing

```python
from hfpytrace.homing import Homing2D, HomingConfig

cfg    = HomingConfig(tol_km=15.0, elev_min_deg=0.0, elev_max_deg=89.0,
                      elev_step_deg=2.0, fine_points=2000, mode="O")
homing = Homing2D(model, config=cfg,
                  trace_kw=dict(x0_km=0.0, z0_km=60.0,
                                s_max_km=3000.0, max_step_km=2.0))
```

### 3) Sweep Frequencies

```python
for f_hz in freqs_hz:
    rays = homing.home(freq_hz=f_hz)            # list[HomingResult]
    for r in rays:
        # r.virtual_height_km  →  ionogram pixel
        # r.x_km, r.z_km      →  ray path for overlay plot
```

### 4) Convenience Wrapper

```python
iono = homing.synthesize_ionogram(freqs_hz)     # ndarray (N, 5)
# columns: freq_hz | virtual_height_km | elevation_deg | ground_range_km | miss_km
```

## Outputs

| File | Description |
|---|---|
| `output/nvis_ionogram_2d.png` | Synthetic ionogram (f vs h') |
| `output/nvis_profile_2d.png` | IRI density cross-section with homed ray paths |

## Usage

```bash
# Default: 2017-05-27T18:00, ionosonde at (40°N, 95°W), 2–12 MHz
python examples/run_homing_nvis_2d.py

# Custom event
python examples/run_homing_nvis_2d.py \
    --date 2021-11-04T12:00 \
    --lat 51.8 --lon 103.1 \
    --fmin 1 --fmax 15 --fstep 0.02 \
    --tol 10 --out ./my_output
```

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--date` | `2017-05-27T18:00` | ISO event time |
| `--lat` | `40.0` | Ionosonde latitude [°N] |
| `--lon` | `-95.0` | Ionosonde longitude [°E] |
| `--fmin` | `2.0` | Minimum frequency [MHz] |
| `--fmax` | `12.0` | Maximum frequency [MHz] |
| `--fstep` | `0.1` | Frequency step [MHz] |
| `--tol` | `15.0` | Homing tolerance [km] |
| `--out` | `./output` | Output directory |
| `--workers` | `4` | IRI fetch threads |

## Interpreting Multi-Root Results

When `homing.home()` returns more than one `HomingResult` at a given frequency,
each entry is a distinct propagation mode:

- **One root near 90°** — standard vertical echo from the `F` layer.
- **Two roots at high elevation** — loop-like ray paths caused by a trough in
  electron density (inner cusp, Laryunin 2025).
- **Roots at lower elevation** — off-vertical echoes from horizontal gradients
  (TID tilts).

The `miss_km` field quantifies how closely each root satisfies the zero-range
condition; values near zero are the most reliable ionogram pixels.

## See Also

- [`hfpytrace.homing`](../dev/hfpytrace/homing.md) — full API reference
- [`hfpytrace.model.rt2d`](../dev/hfpytrace/model/rt2d.md) — RT2D tracer
- [RT2D IRI Cartesian Example](rt2d_iri_cartesian.md) — oblique fan tracing
- [Homing3D Oblique Example](homing_oblique_3d.md) — 3-D lat/lon homing
