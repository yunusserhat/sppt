"""Core SPPT functionality."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sppt.bootstrap import (
    convert_to_percentages,
    expand_counts_to_events,
    sparse_bootstrap,
)
from sppt.metrics import (
    check_interval_overlap,
    compute_confidence_intervals,
    compute_s_index,
    compute_sindex_bivariate,
)


def sppt(
    data,
    group_col: str,
    count_cols: list[str],
    B: int = 200,
    new_col: Optional[list[str]] = None,
    conf_level: float = 0.95,
    check_overlap: bool = False,
    fix_base: bool = False,
    use_percentages: bool = True,
    seed: Optional[int] = None,
    create_maps: bool = True,
    export_maps: bool = False,
    export_dir: Optional[str] = None,
    map_dpi: int = 300,
    export_results: bool = False,
    export_format: str = "shp",
    export_results_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform bootstrap-based spatial pattern point tests on aggregated count data.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        Input data with aggregated count data
    group_col : str
        Column name for spatial group identifiers
    count_cols : list[str]
        List of column names containing count data.
        First is treated as "base" variable, second as "test" variable.
    B : int, optional
        Number of bootstrap samples (default: 200)
    new_col : list[str], optional
        List of new column names for output. If None, uses count_cols names (default: None)
    conf_level : float, optional
        Confidence level for intervals (default: 0.95)
    check_overlap : bool, optional
        Whether to check interval overlap and compute S-Index (default: False)
    fix_base : bool, optional
        Fix the base (first) variable without bootstrapping (default: False)
    use_percentages : bool, optional
        Use percentages (True) or counts (False) (default: True)
    seed : int, optional
        Random seed for reproducibility (default: None)
    create_maps : bool, optional
        Whether to create maps for bivariate case (default: True)
    export_maps : bool, optional
        Whether to export generated maps (default: False)
    export_dir : str, optional
        Directory for map exports. If None, uses current working directory (default: None)
    map_dpi : int, optional
        DPI for exported maps (default: 300)
    export_results : bool, optional
        Whether to export results (default: False)
    export_format : str, optional
        Export format: "shp", "csv", "txt", "rds", or "gpkg" (default: "shp")
    export_results_dir : str, optional
        Directory for results export. If None, uses current working directory (default: None)

    Returns
    -------
    result : DataFrame or GeoDataFrame
        Input data with added columns:
        - {var}_L, {var}_U: Lower/upper confidence interval bounds
        - intervals_overlap: Binary indicator of overlap (if check_overlap=True)
        - SIndex_Bivariate: -1 (base > test), 0 (overlap), 1 (test > base) (if 2 vars)
        Attributes:
        - s_index: Proportion of observations with overlapping intervals
        - robust_s_index: S-Index excluding zero-count observations
    """
    # Store whether input has geometry (for GeoDataFrames)
    has_geometry = hasattr(data, "geometry")
    geometry_name = data.geometry.name if has_geometry else None

    # Extract data frame
    if has_geometry:
        df = data.drop(columns=[geometry_name]).copy()
    else:
        df = data.copy()

    # Handle multiple count columns
    n_vars = len(count_cols)

    # If no new_col specified, use count_cols names
    new_cols = new_col if new_col is not None else count_cols

    # Validate new_col length
    if new_col is not None and len(new_col) != n_vars:
        raise ValueError("new_col must have the same length as count_cols when both are provided")

    # Validate export_format
    valid_formats = ["shp", "csv", "txt", "rds", "gpkg"]
    if export_format.lower() not in valid_formats:
        warnings.warn(
            f"Unsupported export format: {export_format}. "
            f"Supported formats: {', '.join(valid_formats)}"
        )

    # Input validation
    _validate_input(data, group_col, count_cols, B, conf_level)

    # Process each count column independently
    result = df.copy()
    for i, count_col in enumerate(count_cols):
        # Skip bootstrapping for Base variable if fix_base is True
        if fix_base and i == 0:
            # For Base variable, calculate percentage or keep counts
            total_count = df[count_col].sum()
            if total_count > 0:
                if use_percentages:
                    result[new_cols[i] + "_L"] = (df[count_col] / total_count) * 100
                    result[new_cols[i] + "_U"] = (df[count_col] / total_count) * 100
                else:
                    result[new_cols[i] + "_L"] = df[count_col]
                    result[new_cols[i] + "_U"] = df[count_col]
            else:
                result[new_cols[i] + "_L"] = 0.0
                result[new_cols[i] + "_U"] = 0.0
        else:
            # Bootstrap for this variable
            result = _bootstrap_single_var(
                result,
                group_col,
                count_col,
                B,
                new_col=new_cols[i],
                conf_level=conf_level,
                fix_base=False,
                use_percentages=use_percentages,
                seed=None if seed is None else seed + i,
            )

    # Check for overlap after all variables are processed
    if check_overlap and n_vars > 1:
        result = _check_overlap(
            result,
            group_col,
            count_cols,
            new_cols,
            fix_base,
            use_percentages,
        )

        # Create map for bivariate case if requested
        if create_maps and n_vars == 2:
            _create_map(
                result,
                count_cols,
                has_geometry,
                geometry_name,
                export_maps,
                export_dir,
                map_dpi,
            )

    # Export results if requested
    if export_results and n_vars > 1:
        _export_results(
            result,
            count_cols,
            export_format,
            export_results_dir,
            has_geometry,
            geometry_name,
        )

    # Restore geometry if present
    if has_geometry and geometry_name:
        result[geometry_name] = data[geometry_name].values

    return result


def _bootstrap_single_var(
    data,
    group_col: str,
    count_col: str,
    B: int,
    new_col: str,
    conf_level: float = 0.95,
    fix_base: bool = False,
    use_percentages: bool = True,
    seed: Optional[int] = None,
):
    """
    Bootstrap a single count column.

    This is called recursively for each count column when processing multiple variables.
    """
    rng = np.random.default_rng(seed)

    # Get counts and group IDs
    counts = data[count_col].values
    groups = data[group_col].values

    # Get unique groups and integer indices
    unique_groups, group_idx = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)

    # Expand counts to events using integer indices
    events_groups, n_events = expand_counts_to_events(counts, group_idx)

    if n_events == 0:
        # No events: set bounds to 0
        data[new_col + "_L"] = 0.0
        data[new_col + "_U"] = 0.0
        return data

    # Perform bootstrap with integer indices
    group_counts = sparse_bootstrap(events_groups, n_groups, B, rng)

    # Convert to percentages if requested
    if use_percentages:
        group_values = convert_to_percentages(group_counts)
    else:
        # Keep as counts
        group_values = group_counts

    # Calculate confidence intervals
    lower, upper = compute_confidence_intervals(group_values, conf_level)

    # Create result dataframe
    df_stat = pd.DataFrame({
        group_col: unique_groups,
        new_col + "_L": lower,
        new_col + "_U": upper,
    })

    # Convert group column back to original type
    df_stat[group_col] = df_stat[group_col].astype(data[group_col].dtype)

    # Join with original data
    result = data.merge(df_stat, on=group_col, how="left")

    # Fill NA with 0
    result[new_col + "_L"] = result[new_col + "_L"].fillna(0)
    result[new_col + "_U"] = result[new_col + "_U"].fillna(0)

    return result


def _check_overlap(
    result,
    group_col: str,
    count_cols: list[str],
    new_cols: list[str],
    fix_base: bool,
    use_percentages: bool,
):
    """
    Check interval overlap and compute S-Index metrics.
    """
    # Get lower and upper bounds
    lower_cols = [new_col + "_L" for new_col in new_cols]
    upper_cols = [new_col + "_U" for new_col in new_cols]

    n_groups = len(result)

    # Extract bounds as arrays
    lower = np.column_stack([result[lower_col].values for lower_col in lower_cols])
    upper = np.column_stack([result[upper_col].values for upper_col in upper_cols])

    if fix_base:
        # When fix_base is TRUE, check if Base value falls within Test interval
        base_lower = result[new_cols[0] + "_L"].values
        base_upper = result[new_cols[0] + "_U"].values
        test_lower = result[new_cols[1] + "_L"].values
        test_upper = result[new_cols[1] + "_U"].values

        # Base value is fixed, so check if it falls within Test interval
        intervals_overlap = (
            (base_lower >= test_lower) & (base_upper <= test_upper)
        ).astype(int)
    else:
        # Check if all intervals overlap
        intervals_overlap = check_interval_overlap(lower, upper)

    # Add overlap column
    result["intervals_overlap"] = intervals_overlap

    # Compute SIndex_Bivariate for exactly two variables
    if len(count_cols) == 2:
        sindex_bivariate = compute_sindex_bivariate(
            result,
            count_cols,
            intervals_overlap,
        )
        result["SIndex_Bivariate"] = sindex_bivariate

    # Compute S-Index and Robust S-Index
    counts_data = result[count_cols]

    s_index, robust_s_index = compute_s_index(intervals_overlap, counts_data, count_cols)

    # Print statistics
    print("\n========================================")
    print("Spatial Pattern Overlap Statistics")
    if fix_base:
        print("Mode: Fixed Base (Test randomized)")
    if use_percentages:
        print("Using: Percentages (spatial distribution)")
    else:
        print("Using: Counts (absolute values)")
    print("========================================")
    print(f"S-Index:          {s_index:.4f}")
    print(f"Robust S-Index:   {robust_s_index:.4f}")
    print("----------------------------------------")
    print(f"Total observations:                {n_groups}")
    print(f"Observations with overlap:         {intervals_overlap.sum()}")
    nonzero_mask = (counts_data > 0).any(axis=1).sum()
    print(f"Observations with non-zero counts: {nonzero_mask}")
    print("========================================\n")

    # Store as attributes
    result.attrs["s_index"] = s_index
    result.attrs["robust_s_index"] = robust_s_index
    result.attrs["fix_base"] = fix_base
    result.attrs["use_percentages"] = use_percentages

    return result


def _create_map(
    result,
    count_cols: list[str],
    has_geometry: bool,
    geometry_name: Optional[str],
    export_maps: bool,
    export_dir: Optional[str],
    map_dpi: int,
):
    """
    Create map for bivariate case.
    """
    if not has_geometry:
        # Can't create map without geometry
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Note: Could not create map. matplotlib is not installed.")
        return

    try:
        # Determine export directory
        if export_maps:
            if export_dir is None:
                export_dir = "."
            import os
            if not os.path.exists(export_dir):
                os.makedirs(export_dir, exist_ok=True)

        # Calculate dimensions for specified DPI
        width_px = 8 * map_dpi
        height_px = 6 * map_dpi

        # Get variable names for legend
        base_name = count_cols[0]
        test_name = count_cols[1]

        # Create map
        if export_maps:
            import os
            fig_path = os.path.join(export_dir, "map_bivariate_s_index.png")
            plt.figure(figsize=(width_px / map_dpi, height_px / map_dpi), dpi=map_dpi)
        else:
            plt.figure(figsize=(8, 6))

        # Get the geometry column
        geom_col = geometry_name if geometry_name else result.geometry.name

        # Create a temporary dataframe for plotting
        plot_data = result.copy()
        if geom_col not in plot_data.columns:
            # Try to get from geometry attribute
            pass

        # Plot SIndex_Bivariate
        # Map values: -1 (base > test), 0 (overlap), 1 (test > base)
        # Colors: gray80 (base > test), white (overlap), black (test > base)
        categories = plot_data["SIndex_Bivariate"]
        category_colors = categories.map({
            -1: "gray80",
            0: "white",
            1: "black",
        })

        # Get geometry
        if geom_col and geom_col in plot_data.columns:
            geoms = plot_data[geom_col]
        else:
            geoms = plot_data.geometry

        # Create plot
        import matplotlib.patches as mpatches
        from io import BytesIO

        # Create legend patches
        base_patch = mpatches.Patch(color="gray80", label=f"{base_name} > {test_name}")
        overlap_patch = mpatches.Patch(color="white", label="Insignificant change")
        test_patch = mpatches.Patch(color="black", label=f"{test_name} > {base_name}")

        # Simple scatter plot for points or use plt.scatter
        # For simplicity, we'll create a basic visualization
        ax = plt.gca()

        # Handle different geometry types
        if hasattr(geoms.iloc[0], "x"):
            # Point geometry
            x_coords = geoms.apply(lambda g: g.x if g else None)
            y_coords = geoms.apply(lambda g: g.y if g else None)
            colors = category_colors.values
            ax.scatter(x_coords, y_coords, c=colors, edgecolor="gray30", s=50)
        else:
            # For other geometries, use a simple visualization
            # This is a simplified approach
            for idx, (geom, color) in enumerate(zip(geoms, category_colors)):
                if geom is not None:
                    try:
                        if hasattr(geom, "boundary"):
                            # Polygon
                            x, y = geom.exterior.xy
                            ax.fill(x, y, facecolor=color, edgecolor="gray30", alpha=0.7)
                        elif hasattr(geom, "x"):
                            # Point
                            ax.scatter(geom.x, geom.y, c=color, edgecolor="gray30", s=50)
                    except:
                        pass

        ax.set_title("S-Index Bivariate")
        ax.legend(handles=[base_patch, overlap_patch, test_patch], loc="upper left")

        if export_maps:
            plt.savefig(fig_path, dpi=map_dpi, bbox_inches="tight")
            plt.close()
            print(f"Map exported successfully to: {export_dir}")
            print("  - map_bivariate_s_index.png\n")
        else:
            plt.show()
            print("Map created successfully.\n")

    except Exception as e:
        print(f"Note: Could not create/export map. Error: {e}\n")


def _export_results(
    result,
    count_cols: list[str],
    export_format: str,
    export_results_dir: Optional[str],
    has_geometry: bool,
    geometry_name: Optional[str],
):
    """
    Export results to file.
    """
    import os

    # Determine export directory
    if export_results_dir is None:
        export_results_dir = "."
    if not os.path.exists(export_results_dir):
        os.makedirs(export_results_dir, exist_ok=True)

    # Create filename from count_cols variables
    var_names = "_".join(count_cols)
    base_filename = f"sppt_output_{var_names}"

    export_format = export_format.lower()

    try:
        if export_format == "shp":
            # Export as shapefile (requires geometry)
            if has_geometry:
                filepath = os.path.join(export_results_dir, f"{base_filename}.shp")
                # For shapefile, we need to drop geometry column and set it properly
                geom_col = geometry_name if geometry_name else "geometry"
                if geom_col in result.columns:
                    # Convert to GeoDataFrame
                    import geopandas as gpd
                    geometry = result[geom_col]
                    geo_result = gpd.GeoDataFrame(
                        result.drop(columns=[geom_col]),
                        geometry=geometry,
                        crs=None
                    )
                    geo_result.to_file(filepath)
                else:
                    # Try to use geometry attribute
                    geo_result = gpd.GeoDataFrame(result, geometry=result.geometry, crs=None)
                    geo_result.to_file(filepath)
                print(f"Results exported as shapefile: {filepath}")
            else:
                print(f"Warning: Cannot export as shapefile: data is not a GeoDataFrame")

        elif export_format in ["csv", "txt"]:
            # Export as CSV/TXT
            filepath = os.path.join(export_results_dir, f"{base_filename}.{export_format}")
            if has_geometry:
                # Drop geometry for CSV
                geom_col = geometry_name if geometry_name else "geometry"
                result_export = result.drop(columns=[geom_col]) if geom_col in result.columns else result
            else:
                result_export = result
            result_export.to_csv(filepath, index=False)
            print(f"Results exported as {export_format.upper()}: {filepath}")

        elif export_format == "gpkg":
            # Export as GeoPackage
            if has_geometry:
                filepath = os.path.join(export_results_dir, f"{base_filename}.gpkg")
                import geopandas as gpd
                geom_col = geometry_name if geometry_name else "geometry"
                if geom_col in result.columns:
                    geometry = result[geom_col]
                    geo_result = gpd.GeoDataFrame(
                        result.drop(columns=[geom_col]),
                        geometry=geometry,
                        crs=None
                    )
                    geo_result.to_file(filepath, driver="GPKG")
                else:
                    geo_result = gpd.GeoDataFrame(result, geometry=result.geometry, crs=None)
                    geo_result.to_file(filepath, driver="GPKG")
                print(f"Results exported as GeoPackage: {filepath}")
            else:
                print(f"Warning: Cannot export as GeoPackage: data is not a GeoDataFrame")

        elif export_format == "rds":
            # For RDS, we'll export as CSV since Python doesn't have native RDS support
            # User can use pandas to_csv and R to read
            filepath = os.path.join(export_results_dir, f"{base_filename}.csv")
            if has_geometry:
                geom_col = geometry_name if geometry_name else "geometry"
                result_export = result.drop(columns=[geom_col]) if geom_col in result.columns else result
            else:
                result_export = result
            result_export.to_csv(filepath, index=False)
            print(f"Results exported as CSV (RDS not natively supported in Python): {filepath}")

        else:
            print(f"Warning: Unsupported export format: {export_format}")
            print(f"Supported formats: shp, csv, txt, rds, gpkg")

    except Exception as e:
        print(f"Warning: Failed to export results: {e}")


def _validate_input(
    data, group_col: str, count_cols: list[str], B: int, conf_level: float
) -> None:
    """
    Validate input data and parameters.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        Input data
    group_col : str
        Column name for group identifiers
    count_cols : list[str]
        Column names for count data
    B : int
        Number of bootstrap samples
    conf_level : float
        Confidence level for intervals

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check that data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}")

    # Check that group_col exists
    if group_col not in data.columns:
        raise ValueError(f"group_col '{group_col}' not found in data. Available columns: {list(data.columns)}")

    # Check that all count_cols exist
    missing_cols = [col for col in count_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"count_cols {missing_cols} not found in data. Available columns: {list(data.columns)}"
        )

    # Check that count columns are numeric and non-negative
    for col in count_cols:
        if not np.issubdtype(data[col].dtype, np.number):
            raise ValueError(f"count_col '{col}' must be numeric, got {data[col].dtype}")
        if (data[col] < 0).any():
            raise ValueError(f"count_col '{col}' contains negative values")

    # Check that count_cols is not empty
    if len(count_cols) == 0:
        raise ValueError("count_cols must contain at least one column")

    # Check that B is positive
    if not isinstance(B, int) or B <= 0:
        raise ValueError(f"B must be a positive integer, got {B}")

    # Check that conf_level is between 0 and 1
    if not (0 < conf_level < 1):
        raise ValueError(f"conf_level must be between 0 and 1 (exclusive), got {conf_level}")
