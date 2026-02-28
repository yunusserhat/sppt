# sppt — Spatial Point Pattern Test for Aggregated Data

[![PyPI version](https://badge.fury.io/py/sppt.svg)](https://badge.fury.io/py/sppt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18813433.svg)](https://doi.org/10.5281/zenodo.18813433)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yunusserhat/sppt-python/blob/main/notebooks/01_quickstart.ipynb)

A Python implementation of the **Spatial Point Pattern Test (SPPT)** for aggregated count data. Uses bootstrap resampling to compare spatial distributions between variables and calculates **S-Index** metrics to quantify spatial pattern overlap.

> **Based on the original R package [`sppt.aggregated.data`](https://github.com/martin-a-andresen/sppt.aggregated.data) by [Martin A. Andresen](https://github.com/martin-a-andresen).** This Python port faithfully reimplements the statistical methods, algorithms, and outputs of the R version.

---

## Features

- **Bootstrap resampling** with sparse-matrix acceleration (`scipy.sparse` + `numpy`)
- **S-Index & Robust S-Index** for quantifying spatial pattern overlap
- **Bivariate comparison** (base vs. test variable) with directional change detection
- **Percentage or count mode** — compare spatial distributions or absolute values
- **Fixed base option** — bootstrap only the test variable when the base is known
- **Automatic choropleth maps** via `matplotlib` + `geopandas`
- **Multiple export formats** — Shapefile, GeoPackage, CSV, TXT, Pickle
- **Google Colab compatible** — works out of the box in cloud notebooks

---

## Installation

```bash
pip install sppt
```

For development:

```bash
git clone https://github.com/yunusserhat/sppt-python.git
cd sppt-python
pip install -e ".[dev]"
```

---

## Quick Start

```python
import geopandas as gpd
from sppt import sppt

# Load spatial data
data = gpd.read_file("your_data.shp")

# Compare two variables across spatial units
result = sppt(
    data=data,
    group_col="DAUID",                # spatial unit identifier
    count_col=["Crime_2020", "Crime_2021"],  # [base, test]
    B=200,                            # bootstrap samples
    check_overlap=True,               # compute S-Index
    seed=42,                          # reproducibility
)

# Access results
print(result.s_index)          # e.g. 0.7380
print(result.robust_s_index)   # e.g. 0.7289
print(result.data.head())      # DataFrame with CI bounds + overlap columns
```

---

## How It Works

### Algorithm

1. **Expand** aggregated counts to individual events (uncount)
2. **Build** a sparse one-hot matrix (n × G) for group membership
3. **Draw** B multinomial bootstrap samples
4. **Aggregate** via matrix multiply: `group_counts = onehot.T @ W`
5. **Convert** to percentages (optional) and extract quantile-based confidence intervals
6. **Compare** intervals between variables to detect significant spatial changes

### S-Index Interpretation

| S-Index | Meaning |
|---------|---------|
| 1.0     | Perfect overlap — no spatial pattern change |
| 0.5     | Half the areas show significant change |
| 0.0     | Complete spatial difference |

The **Robust S-Index** excludes spatial units where all variables are zero.

### SIndex_Bivariate (per spatial unit)

| Value | Meaning |
|-------|---------|
| -1    | Base > Test (decline) |
|  0    | No significant difference |
| +1    | Test > Base (increase) |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | — | GeoDataFrame or DataFrame with count data |
| `group_col` | `"group"` | Column identifying spatial units |
| `count_col` | — | Column name(s) with counts. Pass `["base", "test"]` for bivariate |
| `B` | `200` | Number of bootstrap samples |
| `seed` | `None` | Random seed for reproducibility |
| `conf_level` | `0.95` | Confidence level for intervals |
| `check_overlap` | `False` | Compute overlap + S-Index statistics |
| `fix_base` | `False` | Skip bootstrapping the base (first) variable |
| `use_percentages` | `True` | Compare spatial distributions (%) vs. raw counts |
| `create_maps` | `True` | Generate choropleth map for bivariate case |
| `export_maps` | `False` | Save map to disk |
| `export_dir` | `None` | Directory for map export |
| `map_dpi` | `300` | Resolution for exported maps |
| `export_results` | `False` | Save results to disk |
| `export_format` | `"shp"` | Format: `"shp"`, `"gpkg"`, `"csv"`, `"txt"`, `"pickle"` |
| `export_results_dir` | `None` | Directory for results export |

---

## Examples

### Example 1: Vancouver Crime Data

```python
import geopandas as gpd
from sppt import sppt

data = gpd.read_file("Vancouver_DAs_Crime_2021.shp")
data = data.to_crs(epsg=26910)

result = sppt(
    data=data,
    group_col="DAUID",
    count_col=["TFV", "TOV"],  # Total Family Violence vs Total Other Violence
    B=200,
    check_overlap=True,
    create_maps=True,
    seed=171717,
)
```

**Output:**
```
========================================
Spatial Pattern Overlap Statistics
Using: Percentages (spatial distribution)
========================================
S-Index:           0.7380
Robust S-Index:    0.7289
----------------------------------------
Total observations:                 1019
Observations with overlap:          752
Observations with non-zero counts:  985
========================================
```

### Example 2: Fixed Base Variable

```python
result = sppt(
    data=data,
    group_col="DAUID",
    count_col=["Census_Official", "Survey_Estimate"],
    B=200,
    fix_base=True,       # don't bootstrap the census data
    check_overlap=True,
    seed=42,
)
```

### Example 3: Count Mode

```python
result = sppt(
    data=data,
    group_col="DAUID",
    count_col=["Crime_2020", "Crime_2021"],
    B=200,
    use_percentages=False,  # compare absolute counts
    check_overlap=True,
    seed=42,
)
```

### Example 4: Export Results

```python
result = sppt(
    data=data,
    group_col="DAUID",
    count_col=["TFV", "TOV"],
    B=500,
    check_overlap=True,
    export_results=True,
    export_format="gpkg",           # GeoPackage
    export_results_dir="output/",
    export_maps=True,
    export_dir="output/maps/",
    map_dpi=600,                    # publication quality
    seed=171717,
)
```

---

## Interactive Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| [Quickstart](notebooks/01_quickstart.ipynb) | Basic usage with Vancouver crime data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yunusserhat/sppt-python/blob/main/notebooks/01_quickstart.ipynb) |
| [Advanced Examples](notebooks/02_advanced_examples.ipynb) | All modes, export, publication maps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yunusserhat/sppt-python/blob/main/notebooks/02_advanced_examples.ipynb) |

---

## Sample Data

The package includes the Vancouver Dissemination Areas Crime 2021 dataset (1,019 polygons) for testing:

```python
from sppt import load_sample_data

data = load_sample_data()
print(data.columns)
# ['DAUID', 'DGUID', 'LANDAREA', 'PRUID', 'BNEC', 'BNER',
#  'MISCHIEF', 'TFV', 'THEFT', 'TOB', 'TOV', 'geometry']
```

---

## Output Columns

After running `sppt()`, your data gains these columns:

| Column | Description |
|--------|-------------|
| `{var}_L` | Lower bound of confidence interval |
| `{var}_U` | Upper bound of confidence interval |
| `intervals_overlap` | `1` if CIs overlap, `0` otherwise |
| `SIndex_Bivariate` | `-1` (base > test), `0` (overlap), `1` (test > base) |

---

## Citation

If you use this package in your research, please cite both the Python package and the original R implementation:

```bibtex
@software{bicakci2026sppt,
  author  = {Bıçakçı, Yunus Serhat},
  title   = {sppt: Spatial Point Pattern Test for Aggregated Data (Python)},
  year    = {2026},
  url     = {https://github.com/yunusserhat/sppt-python},
  doi     = {10.5281/zenodo.18813433},
  note    = {Python implementation based on the R package by Martin A. Andresen}
}

@software{andresen2025sppt,
  author  = {Andresen, Martin A.},
  title   = {sppt.aggregated.data: Spatial Point Pattern Test for Aggregated Data (R)},
  year    = {2025},
  url     = {https://github.com/martin-a-andresen/sppt.aggregated.data}
}
```

---

## Acknowledgements

This package is a faithful Python reimplementation of the R package [`sppt.aggregated.data`](https://github.com/martin-a-andresen/sppt.aggregated.data) created by **[Martin A. Andresen](https://github.com/martin-a-andresen)**. The statistical methodology, bootstrap algorithm, S-Index calculations, and output structure are directly based on his original work.

---

## License

[MIT](LICENSE)
