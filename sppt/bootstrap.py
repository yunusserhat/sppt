"""Bootstrap resampling utilities for SPPT."""

import numpy as np
from scipy import sparse


def expand_counts_to_events(
    counts: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand aggregated counts to individual events.

    This replicates the behavior of tidyr::uncount() in R.
    Each count value is expanded into that many copies of its group_id.

    Parameters
    ----------
    counts : np.ndarray
        Array of count values (1D)
    group_ids : np.ndarray
        Array of group IDs corresponding to each count

    Returns
    -------
    events_groups : np.ndarray
        Array of group IDs for each expanded event
    total_events : int
        Total number of events
    """
    # Flatten the expansion - replicate each group_id by its count
    # This is equivalent to R's tidyr::uncount()
    if counts.size == 0:
        return np.array([], dtype=group_ids.dtype), 0

    # Use np.repeat to expand
    events_groups = np.repeat(group_ids, counts)
    total_events = len(events_groups)

    return events_groups, total_events


def sparse_bootstrap(
    events_groups: np.ndarray,
    n_groups: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Perform bootstrap resampling using sparse matrix operations.

    This implements the same algorithm as the R package's sparse matrix approach:
    1. Create a one-hot encoding of events to groups
    2. Resample with replacement using rmultinom
    3. Compute group counts via matrix multiplication

    Parameters
    ----------
    events_groups : np.ndarray
        Array of group indices for each event (integer 0 to n_groups-1)
    n_groups : int
        Number of unique groups
    B : int
        Number of bootstrap samples
    rng : np.random.Generator
        NumPy random number generator

    Returns
    -------
    group_counts : np.ndarray
        Bootstrap sample group counts of shape (n_groups, B)
        Each column is one bootstrap sample
    """
    n_events = len(events_groups)

    if n_events == 0:
        return np.zeros((n_groups, B), dtype=np.float64)

    # Create one-hot encoding: n_events x n_groups
    # Each row has a 1 in the column corresponding to the event's group index
    row_indices = np.arange(n_events)
    one_hot = sparse.csr_matrix(
        (np.ones(n_events), (row_indices, events_groups.astype(int))),
        shape=(n_events, n_groups),
        dtype=np.float64,
    )

    # Resample with replacement: B samples from multinomial with n_events trials
    # Each event has equal probability 1/n_events
    probabilities = np.ones(n_events) / n_events

    # Use scipy's sparse matrix multiplication for efficiency
    # W is n_events x B matrix of resampled indices (each column sums to n_events)
    W = rng.multinomial(n_events, probabilities, size=B).T  # shape: n_events x B

    # Compute group counts: one_hot.T @ W = n_groups x B
    group_counts = one_hot.T @ W

    return np.asarray(group_counts)


def get_group_indices(
    groups: np.ndarray,
    unique_groups: np.ndarray,
) -> np.ndarray:
    """
    Map group IDs to integer indices.

    Parameters
    ----------
    groups : np.ndarray
        Array of group IDs (can be strings, numbers, etc.)
    unique_groups : np.ndarray
        Array of unique group IDs

    Returns
    -------
    group_indices : np.ndarray
        Integer indices corresponding to each group
    """
    # Create a mapping from group ID to integer index
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    # Map each group to its integer index
    return np.array([group_to_idx[g] for g in groups])


def convert_to_percentages(group_counts: np.ndarray) -> np.ndarray:
    """
    Convert counts to percentages for each bootstrap sample.

    Parameters
    ----------
    group_counts : np.ndarray
        Group counts of shape (n_groups, B)

    Returns
    -------
    group_values : np.ndarray
        Group percentages of shape (n_groups, B)
    """
    col_sums = group_counts.sum(axis=0, keepdims=True)  # shape: (1, B)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1.0
    group_values = (group_counts / col_sums) * 100.0
    return group_values
