# Changelog

## 0.1.5 (2026-02-28)

### Added

- Added Andresen (2016) journal article reference to README and CITATION.cff

### Changed

- Updated PyPI project description (README sync)

---

## 0.1.4 (2026-02-28)

### Changed

- Removed manual Zenodo workflow (now using native Zenodo-GitHub integration)
- Clean final release

---

## 0.1.3 (2026-02-28)

### Changed

- Updated Zenodo DOI to 10.5281/zenodo.18813433
- Clean release for PyPI + Zenodo sync

---

## 0.1.2 (2026-02-28)

### Changed

- Renamed GitHub repository to `sppt-python` for Zenodo compatibility
- Updated all repository URLs (Colab links, badges, citation)

---

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
