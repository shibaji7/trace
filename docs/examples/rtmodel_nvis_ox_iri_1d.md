# RT1D NVIS O/X from IRI (config1D)

<div class="hero">
  <h3>End-to-End 1D Tracer Example</h3>
  <p>
    Build a single-point IRI profile, run O/X NVIS tracers, and overlay IRI
    plasma-frequency and electron-density diagnostics on one scientific plot.
  </p>
</div>

This page explains:

- `examples/rtmodel_nvis_ox_iri_1d.py`

## Call Flow

1. `main()` parses CLI arguments (`--config`, `--event`, frequency sweep, regridding options).
2. `load_config_1D(...)` resolves bundled or user-provided `config1D.json`.
3. `RT1D(...)` initializes a 1D profile and fetches:
   - IRI electron density (`fetch_iri=True`)
   - geomagnetic profile (`fetch_geomag=True`, optional via `--no-geomag`)
4. `RT1D.NVIS_tracer(...)` runs O-mode and X-mode tracer sweeps.
5. `_plot_results(...)` draws a single-panel figure:
   - bottom x-axis: frequency [MHz]
   - y-axis: height/altitude [km]
   - O/X virtual-height traces
   - IRI plasma-frequency profile on same axis
   - top x-axis: IRI `N_e`
6. Script logs summary diagnostics with `loguru` and saves figure to `--out`.

## Key Code (From `rtmodel_nvis_ox_iri_1d.py`)

### 1) Build RT1D Profile from Config

```python
cfg = load_config_1D(config_path)
rt = RT1D(
    cfg=cfg,
    time=event_time,
    fetch_iri=True,
    fetch_geomag=not args.no_geomag,
    fetch_msise=False,
    workers=max(1, int(getattr(cfg, "worker", 1))),
)
```

### 2) Run O/X Tracers

```python
freqs_mhz = np.linspace(args.fmin, args.fmax, args.nfreq)
o_res = rt.NVIS_tracer(
    freq_mhz=freqs_mhz,
    mode="O",
    formulation=args.formulation,
    use_nonuniform_grid=not args.uniform_grid,
    nonuniform_points=int(args.nonuniform_points),
    nonuniform_sharpness=float(args.nonuniform_sharpness),
)
x_res = rt.NVIS_tracer(
    freq_mhz=freqs_mhz,
    mode="X",
    formulation=args.formulation,
    use_nonuniform_grid=not args.uniform_grid,
    nonuniform_points=int(args.nonuniform_points),
    nonuniform_sharpness=float(args.nonuniform_sharpness),
)
```

### 3) Plot Traces + IRI Diagnostics

```python
pf_mhz = RT1D.den_to_plasma_freq_hz(ne_m3) / 1e6
ax.plot(freqs_mhz, o_res.vh_km, ...)
ax.plot(freqs_mhz, x_res.vh_km, ...)
ax.plot(pf_mhz, alt_km, ..., label="IRI fp profile")

ax_top = ax.twiny()
ax_top.semilogx(ne_m3, alt_km, ..., label="IRI Ne profile")
```

## Regridding Controls (Jagged-Trace Reduction)

`RT1D.NVIS_tracer(...)` supports stretched nonuniform vertical remapping:

- `--uniform-grid` disables regridding
- `--nonuniform-points` increases points near reflection region
- `--nonuniform-sharpness` concentrates samples near turning height

Typical tuning:

- points: `300-600`
- sharpness: `10-18`

## Run

Default sweep:

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/rtmodel_nvis_ox_iri_1d.py --fmin 1 --fmax 12 --nfreq 301
```

Smoothed sweep:

```bash
python examples/rtmodel_nvis_ox_iri_1d.py \
  --fmin 1 --fmax 12 --nfreq 301 \
  --nonuniform-points 500 \
  --nonuniform-sharpness 16
```

## Output

Default output:

- `docs/examples/figures/rt1d_nvis_ox_iri.png`

## Related Files

- `examples/rtmodel_nvis_ox_iri_1d.py`
- `trace/model/rt1d.py`
- `trace/model/dispersion.py`
- `trace/density/iri.py`
- `trace/geomag.py`

## See Also

- [RT1D O-Mode Appleton vs Sen-Wyller](rtmodel_omode_appleton_sw_demo.md)
- [PHaRLAP + IRI 2D Ray Trace](pharlap_iri.md)
