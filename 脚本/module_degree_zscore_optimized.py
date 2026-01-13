"""
module_degree_zscore_optimized.py

Within-module degree z-score - Numba JIT optimized version

Translated from MATLAB Brain Connectivity Toolbox
Reference: Guimera R, Amaral L. Nature (2005) 433:895-900.

Original author: Mika Rubinov, UNSW, 2008-2010
Python translation and optimization: Claude
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _module_degree_zscore_core(W, Ci, max_module):
    """
    Core computation of within-module degree z-score.
    
    Parameters
    ----------
    W : ndarray
        Connection matrix (n x n), already processed for flag.
    Ci : ndarray
        Community affiliation vector (1-indexed, flattened).
    max_module : int
        Maximum module index.
    
    Returns
    -------
    Z : ndarray
        Within-module degree z-score.
    """
    n = W.shape[0]
    Z = np.zeros(n)
    
    # Pre-allocate arrays for the largest possible module
    member_indices = np.empty(n, dtype=np.int64)
    Koi = np.empty(n, dtype=np.float64)
    
    for module_idx in range(1, max_module + 1):
        # Count members and collect indices in one pass
        n_members = 0
        for i in range(n):
            if Ci[i] == module_idx:
                member_indices[n_members] = i
                n_members += 1
        
        if n_members == 0:
            continue
        
        # Compute within-module degree (Koi) for each member
        for i in range(n_members):
            node_i = member_indices[i]
            degree_sum = 0.0
            for j in range(n_members):
                degree_sum += W[node_i, member_indices[j]]
            Koi[i] = degree_sum
        
        # Compute mean using running sum
        mean_Koi = 0.0
        for i in range(n_members):
            mean_Koi += Koi[i]
        mean_Koi /= n_members
        
        # Compute std with ddof=1 (sample standard deviation, matching MATLAB)
        if n_members > 1:
            var_sum = 0.0
            for i in range(n_members):
                diff = Koi[i] - mean_Koi
                var_sum += diff * diff
            std_Koi = np.sqrt(var_sum / (n_members - 1))
        else:
            std_Koi = 0.0
        
        # Compute and assign z-scores
        if std_Koi == 0.0:
            for i in range(n_members):
                Z[member_indices[i]] = 0.0  # Directly set to 0 instead of NaN
        else:
            inv_std = 1.0 / std_Koi
            for i in range(n_members):
                Z[member_indices[i]] = (Koi[i] - mean_Koi) * inv_std
    
    return Z


def module_degree_zscore(W, Ci, flag=0):
    """
    Within-module degree z-score (Numba JIT optimized).
    
    The within-module degree z-score is a within-module version of degree
    centrality.
    
    Parameters
    ----------
    W : ndarray
        Binary/weighted, directed/undirected connection matrix.
    Ci : ndarray
        Community affiliation vector (1-indexed).
    flag : int, optional
        0: undirected graph (default)
        1: directed graph: out-degree
        2: directed graph: in-degree
        3: directed graph: out-degree and in-degree
    
    Returns
    -------
    Z : ndarray
        Within-module degree z-score.
    
    References
    ----------
    Guimera R, Amaral L. Nature (2005) 433:895-900.
    """
    # Ensure W is a contiguous float64 array
    W = np.ascontiguousarray(W, dtype=np.float64)
    
    # Process W according to flag
    if flag == 2:
        W = np.ascontiguousarray(W.T)
    elif flag == 3:
        W = np.ascontiguousarray(W + W.T)
    # flag == 0 or flag == 1: no action required
    
    # Ensure Ci is a contiguous int64 array
    Ci = np.ascontiguousarray(np.asarray(Ci).flatten(), dtype=np.int64)
    
    max_module = int(np.max(Ci))
    
    return _module_degree_zscore_core(W, Ci, max_module)


@njit(parallel=True, cache=True, fastmath=True)
def _module_degree_zscore_parallel(W, Ci, max_module):
    """
    Parallel version of core computation.
    Beneficial for large graphs with many modules.
    """
    n = W.shape[0]
    Z = np.zeros(n)
    
    # Pre-compute module assignments: create an array of module member indices
    # First, count members per module
    member_counts = np.zeros(max_module + 1, dtype=np.int64)
    for i in range(n):
        module_idx = Ci[i]
        if module_idx >= 1 and module_idx <= max_module:
            member_counts[module_idx] += 1
    
    # Compute starting positions for each module in a flat array
    module_starts = np.zeros(max_module + 2, dtype=np.int64)
    for i in range(1, max_module + 1):
        module_starts[i + 1] = module_starts[i] + member_counts[i]
    
    # Create flat array of all member indices, organized by module
    all_members = np.empty(n, dtype=np.int64)
    current_pos = np.zeros(max_module + 1, dtype=np.int64)
    for i in range(n):
        module_idx = Ci[i]
        if module_idx >= 1 and module_idx <= max_module:
            pos = module_starts[module_idx] + current_pos[module_idx]
            all_members[pos] = i
            current_pos[module_idx] += 1
    
    # Process each module in parallel
    for module_idx in prange(1, max_module + 1):
        start = module_starts[module_idx]
        end = module_starts[module_idx + 1]
        n_members = end - start
        
        if n_members == 0:
            continue
        
        # Compute within-module degree (Koi)
        Koi = np.zeros(n_members)
        for i in range(n_members):
            node_i = all_members[start + i]
            degree_sum = 0.0
            for j in range(n_members):
                node_j = all_members[start + j]
                degree_sum += W[node_i, node_j]
            Koi[i] = degree_sum
        
        # Compute mean
        mean_Koi = 0.0
        for i in range(n_members):
            mean_Koi += Koi[i]
        mean_Koi /= n_members
        
        # Compute std with ddof=1
        if n_members > 1:
            var_sum = 0.0
            for i in range(n_members):
                diff = Koi[i] - mean_Koi
                var_sum += diff * diff
            std_Koi = np.sqrt(var_sum / (n_members - 1))
        else:
            std_Koi = 0.0
        
        # Compute and assign z-scores
        if std_Koi == 0.0:
            for i in range(n_members):
                Z[all_members[start + i]] = 0.0
        else:
            inv_std = 1.0 / std_Koi
            for i in range(n_members):
                Z[all_members[start + i]] = (Koi[i] - mean_Koi) * inv_std
    
    return Z


def module_degree_zscore_parallel(W, Ci, flag=0):
    """
    Within-module degree z-score (Numba JIT with parallel processing).
    
    Use this version for very large graphs with many modules.
    """
    W = np.ascontiguousarray(W, dtype=np.float64)
    
    if flag == 2:
        W = np.ascontiguousarray(W.T)
    elif flag == 3:
        W = np.ascontiguousarray(W + W.T)
    
    Ci = np.ascontiguousarray(np.asarray(Ci).flatten(), dtype=np.int64)
    max_module = int(np.max(Ci))
    
    return _module_degree_zscore_parallel(W, Ci, max_module)
