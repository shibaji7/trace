# RT1D O-Mode Appleton Absorption (IRI)

<div class="hero">
  <h3>1D O-Mode Absorption Curves in dB/km</h3>
  <p>
    Uses a 1D IRI profile with NRLMSISE-driven collision frequency and computes
    Appleton-Hartree O-mode absorption for three frequencies (default includes 30 MHz).
  </p>
</div>

This page explains:

- `examples/rtmodel_omode_absorption_appleton_iri_1d.py`

## Call Flow

1. Load `config1D.json` via `load_config_1D(...)`.
2. Build profile:
   - `RT1DProfile.from_cfg(..., fetch_iri=True, fetch_msise=True, fetch_geomag=True)`
3. Build collision profile from NRLMSISE neutrals through:
   - `ComputeCollision.from_nrlmsise(...)`
4. For each requested frequency, evaluate:
   - `AppletonHartreeDispersion(...).evaluate(mode="O")`
5. Plot a two-panel figure:
   - left: absorption vs altitude in `dB/km`
   - right: plasma frequency (`f_p`) and collision frequency (`nu`) in `MHz`

## Run

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/rtmodel_omode_absorption_appleton_iri_1d.py
```

Custom frequencies (must be 3 values):

```bash
python examples/rtmodel_omode_absorption_appleton_iri_1d.py --freqs 15,22,30
```

Disable automatic collision-frequency safety rescale:

```bash
python examples/rtmodel_omode_absorption_appleton_iri_1d.py --no-auto-nu-rescale
```

## Output Figure

Default output:

- `docs/examples/figures/rt1d_omode_absorption_appleton_iri.png`

![RT1D O-mode absorption](figures/rt1d_omode_absorption_appleton_iri.png)
