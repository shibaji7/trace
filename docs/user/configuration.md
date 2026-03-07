# Configuration Files

TRACE ships with two JSON templates under `trace/cfg/`:

- `config1D.json`: settings for single-point 1D profile workflows (`RT1DProfile`).
- `config2D.json`: settings for 2D PHaRLAP workflows.
- `config3D.json`: settings for 3D PHaRLAP workflows.

You can either:

- use package defaults (installed location), or
- pass your own absolute/relative JSON path via `--config`.

## Where Configs Are Loaded From

The examples use `trace.utils.resolve_config_path(...)`:

- if `--config` is provided and exists, it is used.
- if `--config` is omitted, TRACE loads installed defaults:
  - `trace/cfg/config2D.json` for 2D
  - `trace/cfg/config3D.json` for 3D

Programmatic config loading helpers:

- `trace.utils.load_config_1D(config_path=None)`
- `trace.utils.load_config_2D(config_path=None)`
- `trace.utils.load_config_3D(config_path=None)`

## `config2D.json` Key Groups

- Event/time:
  - `event`, `time_window`, `time_gaps`
- Route:
  - `route.start.lat/lon`
  - `route.end.lat/lon` (optional when `route.bearing` is used)
  - `max_ground_range_km`, `number_of_ground_step_km`
- Height/elevation:
  - `start_height_km`, `end_height_km`, `height_incriment_km`
  - `start_elevation`, `end_elevation`, `elevation_inctiment`
- Raytrace control:
  - `frequency` (MHz), `nhops`, `threshold`, `radius_earth`
- IRI setup:
  - `iri_param.f107`
  - `iri_param.foF2_coeff` (`CCIR` or `URSI`)
  - `iri_param.hmF2_model` (for example `SHU2015`)
  - `iri_param.coord` (`GEO`)
- Runtime:
  - `worker` (kept for compatibility; IRI evaluation is vectorized)

## `config1D.json` Key Groups

- Event/time:
  - `event`
- Location:
  - `origin.lat`, `origin.lon`
- Height profile:
  - `start_height_km`, `end_height_km`, `height_incriment_km`
- IRI setup:
  - `iri_param.f107`
  - `iri_param.foF2_coeff`
  - `iri_param.hmF2_model`
  - `iri_param.coord`
- Geomagnetic setup:
  - `geomag_grid.coord_input`
  - `geomag_grid.coeff_dir`
- Runtime:
  - `worker`

`config1D.json` is used by RT model examples such as:

- `examples/rtmodel_nvis_ox_iri_1d.py`
- `examples/rtmodel_omode_appleton_sw_demo.py`

## `config3D.json` Key Groups

- Event/time:
  - `event`, `time_window`, `time_gaps`
- Origin and launch fan:
  - `origin.lat/lon/height_km`
  - `start_elevation`, `end_elevation`, `elevation_inctiment`
  - `start_bearing`, `end_bearing`, `bearing_increment`
- 3D ionosphere grid (`iono_grid`):
  - `lat_start`, `lat_step`, `num_lats`
  - `lon_start`, `lon_step`, `num_lons`
  - `height_start_km`, `height_step_km`, `num_heights`
- 3D geomagnetic grid (`geomag_grid`):
  - same structure as `iono_grid`
- Raytrace control:
  - `frequency` (MHz), `nhops`, `threshold`, `OX_mode`, `use_spherical`
  - `radius_earth_m`
- IRI setup:
  - `iri_param.f107`, `foF2_coeff`, `hmF2_model`, `coord`

## Minimal Usage

```bash
python examples/run_pharlap_iri.py
python examples/run_pharlap_iri_3d.py
```

With a custom file:

```bash
python examples/run_pharlap_iri.py --config /absolute/path/my2d.json
python examples/run_pharlap_iri_3d.py --config /absolute/path/my3d.json
```
