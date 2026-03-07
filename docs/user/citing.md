# Citing

<div class="hero">
  <h3>How to cite TRACE</h3>
  <p>
    If TRACE contributes to published work, please cite the software release and core scientific dependencies.
  </p>
</div>

!!! warning "Beta Citation Notice (Updated March 3, 2026)"
    Citation metadata may be updated while releases are stabilized.

## Recommended TRACE Citation

Use the project release DOI when available.

```text
TRACE Development Team (Year). TRACE: HF Ray Tracing Toolkit (Version X.Y.Z). DOI: <release-doi>
```

## BibTeX Template

```bibtex
@software{trace,
  author = {TRACE Development Team},
  title = {TRACE: HF Ray Tracing Toolkit},
  year = {2026},
  version = {X.Y.Z},
  doi = {<release-doi>},
  url = {https://github.com/shibaji7/trace}
}
```

## Publications Using TRACE

- Add your paper here via pull request.

## Core Scientific Python References

- Matplotlib: Hunter, J.D. (2007), Computing in Science & Engineering.
- NumPy: van der Walt et al. (2011), Computing in Science & Engineering.
- SciPy: Virtanen et al. (2020), Nature Methods.
- Pandas: McKinney (2010), Proc. 9th Python in Science Conference.

## PyIRI Citation

TRACE uses `PyIRI` for IRI-based electron density generation. Please also cite:

```text
Forsythe, V. V., Bilitza, D., Burrell, A. G., Dymond, K. F., Fritz, B. A., & McDonald, S. E. (2024). PyIRI: Whole-globe approach to the International Reference Ionosphere modeling implemented in Python. Space Weather, 22, e2023SW003739. https://doi.org/10.1029/2023SW003739
```

```bibtex
@article{forsythe2024pyiri,
  author = {Forsythe, V. V. and Bilitza, D. and Burrell, A. G. and Dymond, K. F. and Fritz, B. A. and McDonald, S. E.},
  title = {PyIRI: Whole-globe approach to the International Reference Ionosphere modeling implemented in Python},
  journal = {Space Weather},
  volume = {22},
  pages = {e2023SW003739},
  year = {2024},
  doi = {10.1029/2023SW003739},
  url = {https://doi.org/10.1029/2023SW003739}
}
```

## SciencePlots Citation

TRACE uses `SciencePlots` styling in plotting workflows. Please also cite:

```bibtex
@article{SciencePlots,
  author       = {John D. Garrett},
  title        = {{garrettj403/SciencePlots}},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0.9},
  doi          = {10.5281/zenodo.4106649},
  url          = {http://doi.org/10.5281/zenodo.4106649}
}
```

## PHaRLAP Citation / Acknowledgement

TRACE interfaces with PHaRLAP for 2D/3D ray tracing. Please include a PHaRLAP
acknowledgement in publications using PHaRLAP-derived results.

Recommended acknowledgement (from PHaRLAP distribution notes):

```text
The results published in this paper were obtained using the HF propagation toolbox, PHaRLAP, created by Dr Manuel Cervera, Defence Science and Technology Group, Australia.
```

Optional BibTeX software entry:

```bibtex
@software{pharlap,
  author = {Cervera, Manuel A.},
  title = {PHaRLAP: Provision of High-frequency Raytracing Laboratory for Propagation studies},
  year = {2023},
  note = {Defence Science and Technology Group, Australia},
  url = {Available by request from the author}
}
```
