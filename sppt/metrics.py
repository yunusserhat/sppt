"""Metrics computation for SPPT: confidence intervals, overlap, S-Index."""

import numpy as np
import pandas as pd


def compute_confidence_intervals(
    group_values: np.ndarray,
    conf_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals from bootstrap samples.

    Parameters
    ----------
    group_values : np.ndarray
        Bootstrap sample values of shape (n_groups, B)
        Each row is a group, each column is a bootstrap sample
    conf_level : float
        Confidence level (default: 0.95)

    Returns
    -------
    lower : np.ndarray
        Lower confidence bound for each group
    upper : np.ndarray
        Upper confidence bound for each group
    """
    alpha = 1 - conf_level
    lower_prob = alpha / 2
    upper_prob = 1 - alpha / 2

    lower = np.percentile(group_values, lower_prob * 100, axis=1)
    upper = np.percentile(group_values, upper_prob * 100, axis=1)

    return lower, upper


def check_interval_overlap(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Check if confidence intervals overlap.

    For a single variable, this always returns 1 (intervals always overlap with themselves).

    For multiple variables (comparisons), checks if all intervals overlap.

    Parameters
    ----------
    lower : np.ndarray
        Lower bounds, shape (n_groups, n_vars)
    upper : np.ndarray
        Upper bounds, shape (n_groups, n_vars)

    Returns
    -------
    overlaps : np.ndarray
        Binary indicator of overlap for each group, shape (n_groups,)
        1 = intervals overlap, 0 = no overlap
    """
    n_groups, n_vars = lower.shape

    if n_vars == 1:
        # Single variable: intervals always "overlap" with themselves
        return np.ones(n_groups, dtype=int)

    # For multiple variables, check if all intervals overlap
    # Intervals overlap if max(lower bounds) <= min(upper bounds)
    max_lower = np.max(lower, axis=1)
    min_upper = np.min(upper, axis=1)

    return (max_lower <= min_upper).astype(int)


def compute_s_index(
    intervals_overlap: np.ndarray,
    counts: pd.DataFrame,
    count_cols: list[str],
) -> tuple[float, float]:
    """
    Compute S-Index and Robust S-Index.

    Parameters
    ----------
    intervals_overlap : np.ndarray
        Binary indicator of overlap for each group
    counts : pd.DataFrame
        Original count data (geometry dropped)
    count_cols : list[str]
        Column names for count data

    Returns
    -------
    s_index : float
        Proportion of observations with overlapping intervals
    robust_s_index : float
        S-Index excluding observations where all count_col variables are zero
    """
    total_obs = len(intervals_overlap)
    sum_overlap = intervals_overlap.sum()

    s_index = sum_overlap / total_obs

    # Robust S-Index: exclude observations where all count_col variables are zero
    nonzero_mask = (counts[count_cols] > 0).any(axis=1).values
    nonzero_obs = nonzero_mask.sum()

    if nonzero_obs > 0:
        sum_overlap_nonzero = intervals_overlap[nonzero_mask].sum()
        robust_s_index = sum_overlap_nonzero / nonzero_obs
    else:
        robust_s_index = np.nan

    return s_index, robust_s_index


def compute_sindex_bivariate(
    counts: pd.DataFrame,
    count_cols: list[str],
    intervals_overlap: np.ndarray,
) -> np.ndarray:
    """
    Compute SIndex_Bivariate for exactly two variables.

    Parameters
    ----------
    counts : pd.DataFrame
        Original count data
    count_cols : list[str]
        Column names for count data (first = base, second = test)
    intervals_overlap : np.ndarray
        Binary indicator of overlap

    Returns
    -------
    sindex_bivariate : np.ndarray
        -1 if base > test (no overlap, base higher)
        0 if intervals overlap
        1 if test > base (no overlap, test higher)
    """
    if len(count_cols) != 2:
        raise ValueError("SIndex_Bivariate requires exactly 2 count columns")

    base_col = count_cols[0]
    test_col = count_cols[1]

    base_values = counts[base_col].values
    test_values = counts[test_col].values

    # SIndex_Bivariate logic:
    # - intervals_overlap == 1: return 0 (overlap = no significant difference)
    # - test > base: return 1 (test greater)
    # - test < base: return -1 (base greater)
    result = np.zeros(len(intervals_overlap), dtype=int)

    # Where intervals don't overlap and test > base
    no_overlap_test_greater = (intervals_overlap == 0) & (test_values > base_values)
    result[no_overlap_test_greater] = 1

    # Where intervals don't overlap and base > test
    no_overlap_base_greater = (intervals_overlap == 0) & (base_values > test_values)
    result[no_overlap_base_greater] = -1

    # Where intervals overlap: already 0

    return result
