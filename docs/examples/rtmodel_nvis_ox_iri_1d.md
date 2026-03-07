# RT1D NVIS O/X from IRI (config1D)

<div class="hero">
  <h3>Single-Panel 1D Tracer Workflow</h3>
  <p>
    Build a 1D IRI profile from <code>config1D.json</code>, run O/X NVIS tracers, and overlay
    IRI plasma-frequency (<code>f<sub>p</sub></code>) and electron-density (<code>N<sub>e</sub></code>) diagnostics.
  </p>
</div>

This page documents:

- `examples/rtmodel_nvis_ox_iri_1d.py`

## What This Example Produces

One panel with:

1. Bottom x-axis: frequency [MHz]
2. Y-axis: height / altitude [km]
3. O-mode and X-mode virtual-height traces
4. IRI plasma-frequency profile (`f_p`) on the same main axis
5. Top x-axis: IRI electron density (`N_e`)

## CLI Interface

```bash
python examples/rtmodel_nvis_ox_iri_1d.py --help
```

Key options:

- `--config`: custom 1D config path (default is installed `trace/cfg/config1D.json`)
- `--event`: UTC timestamp override
- `--fmin`, `--fmax`, `--nfreq`: tracer frequency sweep
- `--formulation`: `appleton` or `senwyller` for O/X traces
- `--uniform-grid`: disable stretched nonuniform regridding
- `--nonuniform-points`: regridded vertical points (default `240`)
- `--nonuniform-sharpness`: regridding concentration near turning point (default `10.0`)
- `--no-geomag`: skip geomagnetic fetch
- `--out`: output image path

## Regridding and Jagged Traces

`RT1D.NVIS_tracer(...)` supports a stretched vertical grid to reduce jagged
turning-height steps:

- `use_nonuniform_grid=True`
- `nonuniform_points`
- `nonuniform_sharpness`

Recommended tuning:

- points: `300-600`
- sharpness: `10-18`

## Example Runs

Default:

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/rtmodel_nvis_ox_iri_1d.py --fmin 1 --fmax 12 --nfreq 301
```

Smoother tracer sweep:

```bash
python examples/rtmodel_nvis_ox_iri_1d.py \
  --fmin 1 --fmax 12 --nfreq 301 \
  --nonuniform-points 500 \
  --nonuniform-sharpness 16
```

## Logging

This script emits structured logs through `loguru`:

- config/time resolution
- RT1D initialization path
- IRI/geomag fetch summaries
- tracer run summary
- saved output location

## Related API

- `trace.model.rt1d.RT1D`
- `trace.model.rt1d.RT1DProfile`
- `trace.model.dispersion.AppletonHartreeDispersion`
- `trace.model.dispersion.SenWyllerDispersion`
