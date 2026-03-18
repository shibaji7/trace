# `hfpytrace.homing`

<span class="api-badge api-package">Package</span>

HF ray homing-in for 2-D and 3-D ionospheric tracers.

"Homing-in" finds all launch angles whose ray path arrives at (or within a tolerance
of) a prescribed target on the ground.  For NVIS / ionosonde work the target ground
range is zero; for oblique links it is a specific range or lat/lon point.

The algorithm follows [Laryunin (2025)](https://doi.org/10.1016/j.asr.2025.01.069):

1. **Fan sweep** – rays are launched over a coarse elevation grid (plus azimuth
   grid in 3-D) and the miss-distance `D` is recorded for each angle.
2. **Root-finding** – a cubic spline is fitted to `D(φ)` and zero-crossings are
   located with Brent's method.  Each crossing is one propagation mode (ordinary,
   loop-like, off-vertical …).
3. **Re-trace** – the ODE is integrated again at each refined angle to collect the
   group delay, virtual height, and full path geometry.

---

## Key Classes

<span class="api-badge api-class">Class</span> `HomingConfig`
<span class="api-badge api-class">Class</span> `HomingResult`
<span class="api-badge api-class">Class</span> `Homing2D`
<span class="api-badge api-class">Class</span> `Homing3D`

---

## `HomingConfig`

Shared sweep and tolerance parameters.  Passed to either `Homing2D` or `Homing3D`.
Per-call keyword arguments on `home()` override these values without mutating the
stored config.

| Field | Type | Default | Description |
|---|---|---|---|
| `tol_km` | float | `10.0` | Acceptance radius [km] |
| `elev_min_deg` | float | `-30.0` | Lower elevation bound [°] |
| `elev_max_deg` | float | `89.0` | Upper elevation bound [°] |
| `elev_step_deg` | float | `2.0` | Coarse elevation step [°] |
| `az_min_deg` | float | `0.0` | *(3-D only)* Lower azimuth bound [°] |
| `az_max_deg` | float | `360.0` | *(3-D only)* Upper azimuth bound [°] |
| `az_step_deg` | float | `5.0` | *(3-D only)* Azimuth step [°] |
| `fine_points` | int | `2000` | Spline interpolation resolution |
| `max_roots` | int | `10` | Safety cap on total returned rays |
| `max_roots_per_az` | int | `5` | *(3-D only)* Safety cap per azimuth slice |
| `mode` | str | `"O"` | Polarisation mode (`"O"` or `"X"`) |

---

## `HomingResult`

Frozen (immutable) record returned by `home()`.  Call `.to_dict()` for a plain
`dict` of scalar fields.

**Common fields (2-D and 3-D)**

| Field | Description |
|---|---|
| `freq_hz` | Operating frequency [Hz] |
| `elevation_deg` | Homed launch elevation [°] |
| `group_path_km` | Integrated group path [km] |
| `group_delay_sec` | Two-way group delay [s] |
| `virtual_height_km` | Virtual height `h' = c·τ/2` [km] |
| `status` | Ray termination status (`"ground"`, `"domain"`, …) |
| `mode` | Polarisation mode |

**2-D only**

| Field | Description |
|---|---|
| `ground_range_km` | Actual landing ground range [km] |
| `miss_km` | Signed miss-distance from target [km] |
| `x_km`, `z_km` | Ray path in (range, altitude) |

**3-D only**

| Field | Description |
|---|---|
| `azimuth_deg` | Homed launch azimuth [°] |
| `landing_lat`, `landing_lon` | Landing coordinates [°] |
| `dist_to_target_km` | Great-circle distance from landing to target [km] |
| `lat_deg`, `lon_deg` | Full path lat/lon arrays (spherical solver) |
| `extra` | Dict of additional raw path arrays (`r_km`, `x_km`, …) |

---

## `Homing2D`

Homing over a 2-D slice (`RT2D`).

### Constructor

```python
Homing2D(rt2d, config=HomingConfig(), trace_fn=None, trace_kw={})
```

### Key Methods

<span class="api-badge api-method">Method</span> `home(freq_hz, *, x_target_km=0.0, tol_km=None, elev_min/max/step_deg=None, mode=None)`
<span class="api-badge api-method">Method</span> `synthesize_ionogram(freqs_hz, *, x_target_km=0.0, tol_km=None, mode=None)`

`synthesize_ionogram` returns `ndarray (N, 5)` –
columns: `freq_hz | virtual_height_km | elevation_deg | ground_range_km | miss_km`.

### Algorithm

```
D(φ) = ground_range(φ) − x_target_km
sign changes → brentq root → precise re-trace
```

Inner cusps / loop-like paths (Laryunin 2025) produce two adjacent sign changes at
nearly the same elevation and are found automatically.

---

## `Homing3D`

Homing over a 3-D volume (`RT3D`).  Decomposes the 2-D search into independent
1-D elevation root-finds per azimuth slice.

### Constructor

```python
Homing3D(rt3d, *, tx_lat, tx_lon, config=HomingConfig(),
         coordinate_system="spherical", solver="gradient", trace_kw={})
```

### Key Methods

<span class="api-badge api-method">Method</span> `home(freq_hz, *, target_lat, target_lon, tol_km=None, az_*/elev_*=None, mode=None)`
<span class="api-badge api-method">Method</span> `synthesize_ionogram(freqs_hz, *, target_lat, target_lon, tol_km=None, mode=None)`

`home()` returns results sorted by ascending `azimuth_deg`.

`synthesize_ionogram` returns `ndarray (N, 6)` –
columns: `freq_hz | virtual_height_km | azimuth_deg | elevation_deg | landing_lat | landing_lon`.

| `coordinate_system` | Landing coordinates source |
|---|---|
| `"spherical"` | `ray.lat_deg[-1]`, `ray.lon_deg[-1]` directly |
| `"cartesian"` | Flat-Earth from `ray.x_km[-1]`, `ray.y_km[-1]` using `tx_lat`/`tx_lon` |

---

## Quick-start Examples

### NVIS ionogram (2-D)

```python
from hfpytrace.homing import Homing2D, HomingConfig
import numpy as np

homing = Homing2D(model, config=HomingConfig(tol_km=10.0),
                  trace_kw=dict(x0_km=0.0, z0_km=60.0, s_max_km=3000.0))

rays = homing.home(freq_hz=7e6)                      # all modes at 7 MHz
iono = homing.synthesize_ionogram(np.arange(2e6, 12e6, 0.02e6))  # (N, 5)
```

### Oblique homing (2-D, 800 km target)

```python
rays = homing.home(freq_hz=7e6, x_target_km=800.0, tol_km=20.0)
```

### Oblique link homing (3-D)

```python
from hfpytrace.homing import Homing3D, HomingConfig

homing = Homing3D(model, tx_lat=40.0, tx_lon=-95.0,
                  config=HomingConfig(tol_km=25.0, az_step_deg=5.0))
rays = homing.home(freq_hz=5e6, target_lat=45.0, target_lon=-90.0)
iono = homing.synthesize_ionogram(np.arange(3e6, 10e6, 0.05e6),
                                   target_lat=45.0, target_lon=-90.0)  # (N, 6)
```

### Config override (no mutation)

```python
tight = homing.home(freq_hz=5e6, tol_km=5.0)
loose = homing.home(freq_hz=5e6, tol_km=30.0)
assert homing.config.tol_km == 25.0   # unchanged
```

---

## API

::: hfpytrace.homing

## Source Code

```python title="hfpytrace/homing.py" linenums="1"
--8<-- "hfpytrace/homing.py"
```
