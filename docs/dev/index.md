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
hfpytrace
├── __init__
├── collision
├── geomag
├── homing
├── pharlap
├── plottrace
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

### `hfpytrace` Package

- [`hfpytrace`](hfpytrace/__init__.md)
- [`hfpytrace.collision`](hfpytrace/collision.md)
- [`hfpytrace.geomag`](hfpytrace/geomag.md)
- [`hfpytrace.homing`](hfpytrace/homing.md)
- [`hfpytrace.pharlap`](hfpytrace/pharlap.md)
- [`hfpytrace.plottrace`](hfpytrace/plottrace.md)
- [`hfpytrace.utils`](hfpytrace/utils.md)

### `hfpytrace.model` Subpackage

- [`hfpytrace.model`](hfpytrace/model/__init__.md)
- [`hfpytrace.model.dispersion`](hfpytrace/model/dispersion.md)
- [`hfpytrace.model.rt1d`](hfpytrace/model/rt1d.md)
- [`hfpytrace.model.rt2d`](hfpytrace/model/rt2d.md)
- [`hfpytrace.model.rt3d`](hfpytrace/model/rt3d.md)

### `hfpytrace.density` Subpackage

- [`hfpytrace.density`](hfpytrace/density/__init__.md)
- [`hfpytrace.density.gemini`](hfpytrace/density/gemini.md)
- [`hfpytrace.density.gitm`](hfpytrace/density/gitm.md)
- [`hfpytrace.density.iri`](hfpytrace/density/iri.md)
- [`hfpytrace.density.sami`](hfpytrace/density/sami.md)
- [`hfpytrace.density.waccm`](hfpytrace/density/waccm.md)
- [`hfpytrace.density.wamipe`](hfpytrace/density/wamipe.md)
