# Examples

<div class="hero">
  <h3>Executable Workflows</h3>
  <p>Hands-on examples for model setup, PHaRLAP execution, and result visualization.</p>
</div>

<div class="doc-card-grid">
  <div class="doc-card">
    <strong>PHaRLAP + IRI 2D Ray Trace</strong>
    End-to-end workflow: route setup, IRI density grid, collision model, PHaRLAP 2D run, and plotting.
    <br><a href="pharlap_iri/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>PHaRLAP + IRI 3D Ray Trace</strong>
    End-to-end workflow: 3D IRI/collision grids, PHaRLAP 3D run, side/front face plots, and MATLAB 3D globe/fallback plots.
    <br><a href="pharlap_iri_3d/">Open Example</a>
  </div>
</div>

!!! warning "3D Geoplot3 Status"
    MATLAB `geoplot3` support is currently **WIP**. It requires Mapping Toolbox and display-enabled MATLAB sessions for full globe rendering. Headless environments use a fallback 3D ECEF `plot3` renderer.
