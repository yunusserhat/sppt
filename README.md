# sppt

<!-- badges: start -->
[![PyPI version](https://img.shields.io/pypi/v/sppt.svg)](https://pypi.org/project/sppt/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.00000000.svg)](https://doi.org/10.5281/zenodo.00000000)
<!-- badges: end -->

The `sppt` package performs spatial pattern point tests on aggregated count data using bootstrap resampling. It compares spatial distributions between variables and calculates S-Index metrics to quantify spatial pattern overlap.

This is a **Python reimplementation** of the canonical R implementation from the `sppt.aggregated.data` package, designed to match the R algorithmic logic and outputs as closely as practical.

Maintainer and package author: **Yunus Serhat Bıçakçı**  
Original R package author: **Martin Andresen**

## Installation

Install the latest release from PyPI:

```bash
pip install sppt
```

Install the development version from GitHub:

```bash
pip install git+https://github.com/yunusserhat/sppt.git
```

Or install from source:

```bash
git clone https://github.com/yunusserhat/sppt.git
cd sppt
pip install -e ".[dev]"
```

## Key Features

- **Bootstrap resampling**: Generate confidence intervals for spatial distributions using efficient sparse matrix operations
- **S-Index metrics**: Quantify spatial pattern overlap between variables
- **Flexible output**: Export results as shapefiles, CSV, GeoPackage, or RDS
- **Visualization**: Create and export maps showing spatial patterns (S-Index bivariate)
- **Multiple comparison modes**: Compare percentages (spatial distribution) or absolute counts
- **GeoDataFrame support**: Full support for spatial data with geometry preservation

## Basic Usage

```python
from sppt import sppt
import pandas as pd

# Load your data (DataFrame or GeoDataFrame)
data = pd.read_csv("your_data.csv")

# Basic analysis
result = sppt(
    data=data,
    group_col="DAUID",
    count_cols=["Base", "Test"],
    B=200,
    seed=123
)

# Analysis with overlap statistics
result = sppt(
    data=data,
    group_col="DAUID",
    count_cols=["Base", "Test"],
    B=200,
    check_overlap=True,
    seed=123
)

# Export results and maps
result = sppt(
    data=data,
    group_col="DAUID",
    count_cols=["TFV", "TOV"],
    B=200,
    check_overlap=True,
    create_maps=True,
    export_maps=True,
    export_dir="output/",
    export_results=True,
    export_format="shp",
    seed=123
)
```

## Parameters

### Core Parameters
- `data`: DataFrame or GeoDataFrame with aggregated count data
- `group_col`: Column name for spatial group identifiers
- `count_cols`: List of column names for count data (first is "base", second is "test")
- `B`: Number of bootstrap samples (default: 200)
- `seed`: Random seed for reproducibility

### Analysis Options
- `use_percentages`: Use percentages (True) or counts (False) (default: True)
- `fix_base`: Fix first variable without bootstrapping (default: False)
- `check_overlap`: Calculate overlap statistics (default: False)
- `conf_level`: Confidence level for intervals (default: 0.95)

### Output Column Naming
- `new_col`: Optional list of new column names for output (defaults to `count_cols` names)

### Visualization Options
- `create_maps`: Create maps for bivariate case (default: True)
- `export_maps`: Export generated maps as PNG (default: False)
- `export_dir`: Directory for map exports (default: None)
- `map_dpi`: DPI for exported maps (default: 300)

### Export Options
- `export_results`: Export results to file (default: False)
- `export_format`: Format for export: "shp", "csv", "txt", "rds", "gpkg" (default: "shp")
- `export_results_dir`: Directory for results export (default: None)

## Output

The function returns the input data with added columns:
- `{variable}_L`: Lower confidence bound
- `{variable}_U`: Upper confidence bound
- `intervals_overlap`: Binary indicator of overlap (if `check_overlap=True`)
- `SIndex_Bivariate`: Spatial pattern comparison (-1, 0, 1) (if two variables)

When `check_overlap=True`, the function also prints statistics:
- **S-Index**: Proportion of observations with overlapping intervals
- **Robust S-Index**: S-Index excluding zero-count observations

### SIndex_Bivariate Values
- `-1`: Base variable is greater than Test (no overlap)
- `0`: No significant difference (intervals overlap)
- `1`: Test variable is greater than Base (no overlap)

## Example: Crime Data Analysis

```python
from sppt import sppt
import pandas as pd

# Load data
data = pd.read_csv("Vancouver_2021_DAs.csv")

# Compare theft from vehicle (TFV) vs theft of vehicle (TOV)
result = sppt(
    data=data,
    group_col="DAUID",
    count_cols=["TFV", "TOV"],
    B=200,
    conf_level=0.95,
    check_overlap=True,
    create_maps=True,
    export_maps=True,
    export_dir="output/",
    map_dpi=600,
    export_results=True,
    export_format="gpkg",
    seed=171717,
    use_percentages=True,
    fix_base=False
)

# View S-Index statistics
print(f"S-Index: {result.attrs['s_index']}")
print(f"Robust S-Index: {result.attrs['robust_s_index']}")

# Map shows:
# - Gray: TFV > TOV
# - White: No significant change
# - Black: TOV > TFV
```

## Development

This is a Python reimplementation of the R `sppt.aggregated.data` package. To develop:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .
```

Generate reference outputs for parity tests (requires R with sppt.aggregated.data):

```bash
python scripts/generate_reference.py
```

## Architecture

```
sppt/
  __init__.py      # Public API: sppt()
  core.py          # Main sppt() function and helper functions
  bootstrap.py     # Bootstrap resampling utilities
  metrics.py       # CI computation, overlap tests, S-Index metrics
tests/
  test_core.py     # Unit tests
  parity/          # Parity tests comparing Python to R output
scripts/
  generate_reference.py  # Generates R reference outputs
```

### Bootstrap Algorithm

The implementation uses sparse matrix operations for efficient bootstrap resampling:
1. Expand counts to individual events (one per unit)
2. Create one-hot encoding of events to groups
3. Resample with replacement using multinomial distribution
4. Compute group counts via sparse matrix multiplication

## Citation

If you use this package in your research, please cite:

```
Bıçakçı, Y.S. (2026). sppt: Spatial Point Pattern Test for Aggregated Data (Python package).
Version 0.1.0. URL: https://github.com/yunusserhat/sppt

Original method and canonical R implementation:
Andresen, M.A. (2026). sppt.aggregated.data.
URL: https://github.com/martin-a-andresen/sppt.aggregated.data
```

## License

MIT License - see LICENSE file for details

## Contact

- GitHub: [yunusserhat](https://github.com/yunusserhat)
- Repository: [yunusserhat/sppt](https://github.com/yunusserhat/sppt)
- Issues: [Report a bug](https://github.com/yunusserhat/sppt/issues)