# PHaRLAP + IRI 2D Ray Trace

This example runs the Python wrapper in:

- `examples/run_pharlap_iri.py`

It does the following:

1. Builds a great-circle route and height grid from `trace/config2D.json`.
2. Fetches IRI electron density (`ne_grid`).
3. Builds collision frequency using `trace.collision` (NRLMSISE-00 driven neutrals).
4. Calls `trace.pharlap.Engine.run_pharlap(...)`.
5. Plots `ray_path_data[i].ground_range` vs `ray_path_data[i].height` on top of `ne_grid` with `trace.plottrace.PlotRays`.

## Run

From repository root:

```bash
cd /home/chakras4/Research/CodeBase/trace
python examples/run_pharlap_iri.py --config trace/config2D.json
```

To only validate IRI/collision inputs and skip MATLAB call:

```bash
python examples/run_pharlap_iri.py --config trace/config2D.json --no-matlab
```

## Output

The example saves:

- `pharlap_iri_ray_paths.png`

If copied into docs assets, it can be viewed below:

![PHaRLAP IRI Ray Paths](figures/pharlap_iri_ray_paths.png)

## Related Files

- `examples/run_pharlap_iri.py`
- `trace/pharlap.py`
- `trace/collision.py`
- `trace/plottrace.py`
