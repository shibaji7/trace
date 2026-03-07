# API Reference

!!! warning "Beta API (Updated March 3, 2026)"
    API signatures and internal implementations are still evolving.

!!! note "Runtime Logging"
    TRACE now emits runtime diagnostics via `loguru` across core workflows
    (profile fetch, tracer execution, plotting, and geomagnetic grid builds).

All module pages now follow the same format:

1. Scope summary
2. Key classes / methods
3. Auto-generated API (`mkdocstrings`)
4. Full source code (embedded)

## Legend

<span class="api-badge api-package">Package</span>
<span class="api-badge api-class">Class</span>
<span class="api-badge api-method">Method / Function</span>

## Module Tree

```text
trace
├── __init__
├── collision
├── dispersion
├── geomag
├── homing
├── pharlap
├── plottrace
├── rt2d
├── utils
├── model
│   ├── __init__
│   ├── dispersion
│   ├── rt1d
│   ├── rt2d
│   └── rt3d
└── density
    ├── __init__
    ├── gemini
    ├── gitm
    ├── iri
    ├── sami
    ├── waccm
    └── wamipe
```

### `trace` Package

- [`trace`](trace/__init__.md)
- [`trace.collision`](trace/collision.md)
- [`trace.dispersion`](trace/dispersion.md)
- [`trace.geomag`](trace/geomag.md)
- [`trace.homing`](trace/homing.md)
- [`trace.pharlap`](trace/pharlap.md)
- [`trace.plottrace`](trace/plottrace.md)
- [`trace.rt2d`](trace/rt2d.md)
- [`trace.utils`](trace/utils.md)

### `trace.model` Subpackage

- [`trace.model`](trace/model/__init__.md)
- [`trace.model.dispersion`](trace/model/dispersion.md)
- [`trace.model.rt1d`](trace/model/rt1d.md)
- [`trace.model.rt2d`](trace/model/rt2d.md)
- [`trace.model.rt3d`](trace/model/rt3d.md)

### `trace.density` Subpackage

- [`trace.density`](trace/density/__init__.md)
- [`trace.density.gemini`](trace/density/gemini.md)
- [`trace.density.gitm`](trace/density/gitm.md)
- [`trace.density.iri`](trace/density/iri.md)
- [`trace.density.sami`](trace/density/sami.md)
- [`trace.density.waccm`](trace/density/waccm.md)
- [`trace.density.wamipe`](trace/density/wamipe.md)
