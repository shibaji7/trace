# `trace.density.iri`

<span class="api-badge api-package">Package</span>

IRI model adapter for route/height electron density generation.
Backend implementation uses `PyIRI` (`PyIRI.sh_library.IRI_density_1day`).

Config parameters (from `cfg.iri_param`):
- `f107` (default `150.0`)
- `foF2_coeff` (default `"CCIR"`)
- `hmF2_model` (default `"SHU2015"`)
- `coord` (default `"GEO"`)

`iri_version` is deprecated and ignored.

## Key Classes

<span class="api-badge api-class">Class</span> `IRI2d`
<span class="api-badge api-class">Class</span> `IRI3d`

## Key Methods

<span class="api-badge api-method">Method</span> `IRI2d.fetch_dataset()`  
<span class="api-badge api-method">Method</span> `IRI2d.load_from_file()`
<span class="api-badge api-method">Method</span> `IRI3d.fetch_dataset()`

## API

::: trace.density.iri

## Source Code

```python title="trace/density/iri.py" linenums="1"
--8<-- "trace/density/iri.py"
```
