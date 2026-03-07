# Examples

<div class="hero">
  <h3>Executable Workflows</h3>
  <p>Hands-on examples for model setup, PHaRLAP execution, and result visualization.</p>
</div>

<div class="doc-card-grid">
  <div class="doc-card">
    <strong>RT1D NVIS O/X (IRI)</strong>
    Single-panel 1D tracer plot with O/X virtual height traces, IRI plasma-frequency profile, and top-axis IRI electron density.
    <br><a href="rtmodel_nvis_ox_iri_1d/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>RT1D O-Mode Appleton vs Sen-Wyller</strong>
    Formulation comparison using a common 1D IRI profile and shared tracer axis conventions.
    <br><a href="rtmodel_omode_appleton_sw_demo/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>PHaRLAP + IRI 2D Ray Trace</strong>
    End-to-end workflow: route setup, IRI density grid, collision model, PHaRLAP 2D run, and plotting.
    <br><a href="pharlap_iri/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>PHaRLAP + IRI 3D Ray Trace</strong>
    End-to-end workflow: 3D IRI/collision grids, PHaRLAP 3D run, and side/front face plots.
    <br><a href="pharlap_iri_3d/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>RT2D IRI Cartesian Ray Trace</strong>
    Pure-Python 2D Cartesian oblique tracing with IRI route profiles and density-ray overlay plots.
    <br><a href="rt2d_iri_cartesian/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>RT2D IRI Spherical Ray Trace</strong>
    Pure-Python 2D spherical oblique tracing with Earth-radius geometry and density-ray overlays.
    <br><a href="rt2d_iri_spherical/">Open Example</a>
  </div>
</div>

!!! warning "3D Geoplot3 Status"
    MATLAB `geoplot3` support is temporarily disabled in the 3D example script.

## Figure Gallery

![RT1D NVIS OX IRI](figures/rt1d_nvis_ox_iri.png)
![RT1D O Mode Appleton vs Sen-Wyller](figures/rt1d_omode_appleton_vs_sw.png)
![PHaRLAP IRI Ray Paths](figures/pharlap_iri_ray_paths.png)
![PHaRLAP IRI 3D Faces](figures/pharlap_iri_3d_ray_faces.png)
![RT2D IRI Cartesian](figures/rt2d_iri_cartesian_ray_paths.png)
![RT2D IRI Spherical](figures/rt2d_iri_spherical_ray_paths.png)
