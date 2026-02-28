"""
sppt — Spatial Point Pattern Test for Aggregated Data (Python)
==============================================================

A faithful Python reimplementation of the R package ``sppt.aggregated.data``
by Martin A. Andresen.

Usage
-----
>>> from sppt import sppt, load_sample_data
>>> data = load_sample_data()
>>> result = sppt(data, group_col="DAUID", count_col=["TFV", "TOV"],
...               B=200, check_overlap=True, seed=171717)
>>> print(result.s_index, result.robust_s_index)
"""

from __future__ import annotations

import os as _os

from .core import sppt, SPPTResult
from .bootstrap import bootstrap_single_variable
from .overlap import compute_overlap, compute_s_indices, compute_sindex_bivariate
from .mapping import create_bivariate_map, create_publication_map
from .export import export_results


def load_sample_data():
    """Load the bundled Vancouver DA Crime 2021 dataset.

    Returns a GeoDataFrame with 1,019 Dissemination Area polygons and
    crime count columns (TFV, TOV, THEFT, MISCHIEF, etc.), projected to
    EPSG:26910 (NAD83 / UTM Zone 10N).

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd

    data_dir = _os.path.join(_os.path.dirname(__file__), "data")
    shp_path = _os.path.join(data_dir, "Vancouver_DAs_Crime_2021.shp")
    if not _os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Sample data not found at {shp_path}. "
            "Reinstall the package or check your installation."
        )
    gdf = gpd.read_file(shp_path)
    return gdf.to_crs(epsg=26910)


__all__ = [
    "sppt",
    "SPPTResult",
    "load_sample_data",
    "bootstrap_single_variable",
    "compute_overlap",
    "compute_s_indices",
    "compute_sindex_bivariate",
    "create_bivariate_map",
    "create_publication_map",
    "export_results",
]

__version__ = "0.1.2"
