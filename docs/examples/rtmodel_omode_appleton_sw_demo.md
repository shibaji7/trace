# RT1D O-Mode: Appleton vs Sen-Wyller

<div class="hero">
  <h3>Formulation Comparison on a Shared 1D Profile</h3>
  <p>
    Compare Appleton-Hartree and Sen-Wyller O-mode tracer outputs over the same
    IRI-based 1D profile and frequency sweep.
  </p>
</div>

This page documents:

- `examples/rtmodel_omode_appleton_sw_demo.py`

## Plot Layout

Single-panel style aligned with the O/X example:

1. Bottom x-axis: frequency [MHz]
2. Y-axis: height / altitude [km]
3. Appleton O-mode and Sen-Wyller O-mode traces
4. IRI plasma-frequency profile (`f_p`) on the same axis
5. Top x-axis: IRI electron density (`N_e`)

## CLI Interface

```bash
python examples/rtmodel_omode_appleton_sw_demo.py --help
```

Key options:

- `--fmin`, `--fmax`, `--nfreq`: frequency sweep
- `--mode`: mode passed to both formulations (typically `O`)
- `--uniform-grid`, `--nonuniform-points`, `--nonuniform-sharpness`
- `--config`, `--event`, `--no-geomag`, `--out`

## Example Runs

O-mode comparison:

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/rtmodel_omode_appleton_sw_demo.py \
  --fmin 3 --fmax 7 --nfreq 51 --mode O
```

Three-frequency compact comparison:

```bash
python examples/rtmodel_omode_appleton_sw_demo.py \
  --fmin 3 --fmax 7 --nfreq 3 --mode O
```

## Logging

`loguru` output includes:

- profile load/fetch diagnostics
- formulation tracer run summaries
- output file path

## Interpretation Note

Sen-Wyller can differ strongly from Appleton-Hartree near upper-frequency
regions where collision handling and generalized response terms become dominant.
Use matched config/time/profile settings when comparing formulations.

## Related API

- `trace.model.rt1d.RT1D.NVIS_tracer`
- `trace.model.dispersion.AppletonHartreeDispersion`
- `trace.model.dispersion.SenWyllerDispersion`
