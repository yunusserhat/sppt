# Changelog

## 0.1.1 (2026-02-28)

### Fixes

- Fixed Colab notebooks — converted from VS Code XML format to valid Jupyter JSON
- Fixed PyPI badge (switched to `img.shields.io`)
- Fixed CI workflow to trigger on version tags
- Added `environment: pypi` to CI publish job for Trusted Publisher support

### Added

- Zenodo integration with DOI: [10.5281/zenodo.18813171](https://doi.org/10.5281/zenodo.18813171)
- `CITATION.cff` for GitHub "Cite this repository" support
- `.zenodo.json` metadata for automatic Zenodo archiving
- Zenodo GitHub Actions workflow for automated uploads on release
- ORCID and affiliation metadata

---

## 0.1.0 (2026-02-28)

### Initial Release

- Core `sppt()` function with full parameter support
- Bootstrap resampling engine using sparse matrices (`scipy.sparse`)
- S-Index and Robust S-Index calculation
- SIndex_Bivariate per-unit directional change indicator
- Percentage mode (spatial distribution) and count mode
- Fixed base variable option
- Choropleth map generation (standard + publication quality)
- Export to Shapefile, GeoPackage, CSV, TXT, Pickle
- Bundled Vancouver DA Crime 2021 sample dataset (1,019 polygons)
- `load_sample_data()` convenience function
- Google Colab-compatible Jupyter notebooks
- 28 unit tests
- Based on the R package [`sppt.aggregated.data`](https://github.com/martin-a-andresen/sppt.aggregated.data) by Martin A. Andresen
