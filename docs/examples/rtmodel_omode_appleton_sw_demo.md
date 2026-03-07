# RT1D O-Mode: Appleton vs Sen-Wyller

<div class="hero">
  <h3>1D O-Mode Formulation Comparison</h3>
  <p>
    Compare Appleton-Hartree and Sen-Wyller tracer outputs on the same IRI
    profile, frequency sweep, and plotting frame.
  </p>
</div>

This page explains:

- `examples/rtmodel_omode_appleton_sw_demo.py`

## Call Flow

1. `main()` parses config/time/frequency arguments (`--fmin`, `--fmax`, `--nfreq`, `--mode`).
2. `load_config_1D(...)` loads bundled or user-provided `config1D.json`.
3. `RT1D(...)` creates the 1D profile and fetches IRI (and geomag unless `--no-geomag`).
4. Two tracer runs are executed with the same mode and sweep:
   - `formulation="appleton"`
   - `formulation="senwyller"`
5. A single-panel figure is built with:
   - Appleton and Sen-Wyller virtual-height traces
   - IRI `f_p` profile on main axis
   - IRI `N_e` profile on top axis
6. Script logs numerical summaries and output path via `loguru`.

## Key Code (From `rtmodel_omode_appleton_sw_demo.py`)

### 1) Shared Profile Setup

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

### 2) Same-Mode, Different-Formulation Tracers

```python
out_a = rt.NVIS_tracer(
    freq_mhz=freqs_mhz,
    mode=args.mode,
    formulation="appleton",
    use_nonuniform_grid=not args.uniform_grid,
    nonuniform_points=int(args.nonuniform_points),
    nonuniform_sharpness=float(args.nonuniform_sharpness),
)
out_s = rt.NVIS_tracer(
    freq_mhz=freqs_mhz,
    mode=args.mode,
    formulation="senwyller",
    use_nonuniform_grid=not args.uniform_grid,
    nonuniform_points=int(args.nonuniform_points),
    nonuniform_sharpness=float(args.nonuniform_sharpness),
)
```

### 3) Overlay Plot (Trace + `f_p` + `N_e`)

```python
ax.plot(freqs_mhz, vh_app, ...)
ax.plot(freqs_mhz, vh_sw, ...)
ax.plot(pf_mhz, alt_km, ..., label="IRI fp profile")

ax_top = ax.twiny()
ax_top.semilogx(ne_m3, alt_km, ..., label="IRI Ne profile")
```

## Run

O-mode sweep:

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/rtmodel_omode_appleton_sw_demo.py \
  --fmin 3 --fmax 7 --nfreq 51 --mode O
```

Three-frequency compact run:

```bash
python examples/rtmodel_omode_appleton_sw_demo.py \
  --fmin 3 --fmax 7 --nfreq 3 --mode O
```

## Interpretation Note

At upper frequencies, Sen-Wyller may deviate strongly from Appleton-Hartree due
to generalized collisional response behavior. Keep all profile/config inputs the
same when comparing formulations.

## Output

Default output:

- `docs/examples/figures/rt1d_omode_appleton_vs_sw.png`

## Rendered Figure

![RT1D O Mode Appleton SenWyller](figures/rt1d_omode_appleton_vs_sw.png)

## Related Files

- `examples/rtmodel_omode_appleton_sw_demo.py`
- `trace/model/rt1d.py`
- `trace/model/dispersion.py`
- `trace/density/iri.py`
- `trace/geomag.py`

## See Also

- [RT1D NVIS O/X from IRI](rtmodel_nvis_ox_iri_1d.md)
- [PHaRLAP + IRI 2D Ray Trace](pharlap_iri.md)
