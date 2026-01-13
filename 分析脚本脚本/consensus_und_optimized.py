"""
consensus_und.py

Optimized Python implementation of consensus clustering using Numba JIT compilation.
This module provides functions for consensus clustering of agreement matrices.

Key optimizations:
- Vectorized modularity gain computation in Louvain algorithm
- Matrix-based graph contraction using indicator matrices
- Efficient agreement matrix computation

Reference: Lancichinetti & Fortunato (2012). Consensus clustering in complex networks.
Scientific Reports 2, Article number: 336.
"""

import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True, cache=True)
def _relabel_partitions_single(c: np.ndarray) -> np.ndarray:
    """
    Relabel a single partition to consecutive integers starting from 1.
    """
    n = len(c)
    c_copy = c.copy()
    d = np.zeros(n, dtype=c.dtype)
    count = 0
    
    filled = 0
    while filled < n:
        count += 1
        ind = -1
        for i in range(n):
            if c_copy[i] != 0:
                ind = i
                break
        
        if ind == -1:
            break
            
        tgt = c_copy[ind]
        for i in range(n):
            if c_copy[i] == tgt:
                d[i] = count
                c_copy[i] = 0
                filled += 1
    
    return d


@jit(nopython=True, cache=True)
def relabel_partitions(ci: np.ndarray) -> np.ndarray:
    """
    Relabel partition labels to be consecutive integers starting from 1.
    """
    n, m = ci.shape
    cinew = np.zeros((n, m), dtype=ci.dtype)
    
    for i in range(m):
        cinew[:, i] = _relabel_partitions_single(ci[:, i])
    
    return cinew


@jit(nopython=True, cache=True)
def unique_partitions(ci: np.ndarray) -> np.ndarray:
    """
    Get unique partitions from partition matrix.
    """
    ci_relabeled = relabel_partitions(ci.copy())
    n, m = ci_relabeled.shape
    
    if m == 0:
        return np.zeros((n, 0), dtype=ci.dtype)
    
    unique_cols = []
    used = np.zeros(m, dtype=np.bool_)
    
    for i in range(m):
        if not used[i]:
            unique_cols.append(i)
            for j in range(i, m):
                if not used[j]:
                    is_same = True
                    for k in range(n):
                        if ci_relabeled[k, i] != ci_relabeled[k, j]:
                            is_same = False
                            break
                    if is_same:
                        used[j] = True
    
    num_unique = len(unique_cols)
    ciu = np.zeros((n, num_unique), dtype=ci.dtype)
    for i, col_idx in enumerate(unique_cols):
        for j in range(n):
            ciu[j, i] = ci_relabeled[j, col_idx]
    
    return ciu


@jit(nopython=True, cache=True, fastmath=True)
def agreement(ci: np.ndarray) -> np.ndarray:
    """
    Compute agreement matrix from partition matrix.
    
    Optimized: Process by community labels to reduce comparisons.
    """
    n, m = ci.shape
    D = np.zeros((n, n), dtype=np.float64)
    
    for col in range(m):
        # Find max label
        max_label = 0
        for i in range(n):
            if ci[i, col] > max_label:
                max_label = ci[i, col]
        
        # For each community
        for label in range(1, max_label + 1):
            # Collect members
            members = np.zeros(n, dtype=np.int64)
            count = 0
            for i in range(n):
                if ci[i, col] == label:
                    members[count] = i
                    count += 1
            
            # Add co-membership (upper triangle only)
            for idx_a in range(count):
                a = members[idx_a]
                for idx_b in range(idx_a + 1, count):
                    b = members[idx_b]
                    D[a, b] += 1.0
    
    # Symmetrize
    for i in range(n):
        for j in range(i + 1, n):
            D[j, i] = D[i, j]
    
    return D


@jit(nopython=True, cache=True, fastmath=True)
def _compute_hnm(B: np.ndarray, Mb: np.ndarray, n_mod: int) -> np.ndarray:
    """
    Compute node-to-module strength matrix using vectorized operations.
    Hnm[u, m] = sum of B[u, v] for all v in module m
    """
    n = len(Mb)
    Hnm = np.zeros((n, n_mod + 1), dtype=np.float64)
    
    # Build indicator matrix and compute Hnm = B @ S
    for u in range(n):
        for v in range(n):
            m = Mb[v]
            Hnm[u, m] += B[u, v]
    
    return Hnm


@jit(nopython=True, cache=True)
def _louvain_one_level(B: np.ndarray, Mb: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Perform one level of Louvain optimization.
    """
    n = len(Mb)
    changed_global = False
    
    # Compute initial Hnm
    n_mod = int(np.max(Mb))
    Hnm = _compute_hnm(B, Mb, n_mod)
    
    flag = True
    while flag:
        flag = False
        
        # Random permutation
        perm = np.arange(n)
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        
        for idx in range(n):
            u = perm[idx]
            ma = Mb[u]
            
            # Find best module to move to
            n_mod_current = int(np.max(Mb))
            best_dQ = 0.0
            best_m = ma
            
            base = -Hnm[u, ma] + B[u, u]
            for m in range(1, n_mod_current + 1):
                if m != ma:
                    dQ = Hnm[u, m] + base
                    if dQ > best_dQ + 1e-10:
                        best_dQ = dQ
                        best_m = m
            
            if best_dQ > 1e-10:
                flag = True
                changed_global = True
                old_ma = ma
                Mb[u] = best_m
                
                # Update Hnm
                for v in range(n):
                    Hnm[v, best_m] += B[v, u]
                    Hnm[v, old_ma] -= B[v, u]
    
    return Mb, changed_global


@jit(nopython=True, cache=True)
def _contract_graph(B: np.ndarray, Mb: np.ndarray) -> np.ndarray:
    """
    Contract graph by aggregating nodes in same module.
    Optimized version that iterates only once through B.
    """
    n = len(Mb)
    n_mod = int(np.max(Mb))
    B_new = np.zeros((n_mod, n_mod), dtype=np.float64)
    
    # Single pass through matrix
    for i in range(n):
        mi = Mb[i] - 1  # 0-indexed
        for j in range(i, n):  # Only upper triangle since B is symmetric
            mj = Mb[j] - 1
            val = B[i, j]
            if i == j:
                B_new[mi, mj] += val
            else:
                B_new[mi, mj] += val
                B_new[mj, mi] += val
    
    return B_new


@jit(nopython=True, cache=True)
def _renumber_modules(Mb: np.ndarray) -> np.ndarray:
    """Renumber modules to be consecutive starting from 1."""
    n = len(Mb)
    unique_mods = np.unique(Mb)
    new_Mb = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        for j in range(len(unique_mods)):
            if Mb[i] == unique_mods[j]:
                new_Mb[i] = j + 1
                break
    
    return new_Mb


@jit(nopython=True, cache=True)
def _community_louvain_core(B: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, float]:
    """
    Core Louvain community detection algorithm.
    """
    np.random.seed(seed)
    n_orig = len(B)
    n = n_orig
    
    # Symmetrize modularity matrix
    B_sym = (B + B.T) * 0.5
    
    # Initial assignment: each node in its own module
    Mb = np.arange(1, n + 1, dtype=np.int64)
    M = Mb.copy()
    
    # Initial modularity (trace of B)
    Q0 = -np.inf
    Q = 0.0
    for i in range(n):
        Q += B_sym[i, i]
    
    first_iteration = True
    
    while Q - Q0 > 1e-10:
        # Phase 1: Local optimization
        Mb, _ = _louvain_one_level(B_sym, Mb)
        
        # Renumber modules
        Mb = _renumber_modules(Mb)
        
        # Update global assignments
        M0_old = M.copy()
        if first_iteration:
            M = Mb.copy()
            first_iteration = False
        else:
            # FIXED: In contracted graph, old module u is represented by node (u-1)
            # So Mb[u-1] gives the NEW module assignment for old module u
            max_old = int(np.max(M0_old))
            for u in range(1, max_old + 1):
                new_val = Mb[u - 1]  # Old module u -> contracted node (u-1) -> new module
                for i in range(n_orig):
                    if M0_old[i] == u:
                        M[i] = new_val
        
        # Phase 2: Contract graph
        n_new = int(np.max(Mb))
        if n_new == n:
            break
            
        B_sym = _contract_graph(B_sym, Mb)
        n = n_new
        Mb = np.arange(1, n + 1, dtype=np.int64)
        
        Q0 = Q
        Q = 0.0
        for i in range(n):
            Q += B_sym[i, i]
    
    return M, Q


def community_louvain(W: np.ndarray, gamma: float = 1.0, 
                      M0: np.ndarray = None, B_type: str = 'modularity',
                      seed: int = None) -> Tuple[np.ndarray, float]:
    """
    Louvain community detection algorithm.
    
    Parameters
    ----------
    W : np.ndarray
        Weighted adjacency matrix (symmetric for undirected networks)
    gamma : float
        Resolution parameter (default: 1.0)
    M0 : np.ndarray
        Initial community assignment (optional, not used in current implementation)
    B_type : str
        Type of modularity matrix ('modularity', 'potts', 'negative_sym', 'negative_asym')
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    M : np.ndarray
        Community assignment vector (1-indexed)
    Q : float
        Modularity value
    """
    W = np.asarray(W, dtype=np.float64)
    n = len(W)
    s = np.sum(W)
    
    if s == 0:
        return np.arange(1, n + 1, dtype=np.int64), 0.0
    
    if B_type == 'modularity':
        k_out = np.sum(W, axis=1)
        k_in = np.sum(W, axis=0)
        B = (W - gamma * np.outer(k_out, k_in) / s) / s
    elif B_type == 'potts':
        B = W - gamma * (~W.astype(bool)).astype(float)
    elif B_type in ('negative_sym', 'negative_asym'):
        W0 = W * (W > 0)
        s0 = np.sum(W0)
        if s0 == 0:
            return np.arange(1, n + 1, dtype=np.int64), 0.0
        B0 = W0 - gamma * np.outer(np.sum(W0, axis=1), np.sum(W0, axis=0)) / s0
        
        W1 = -W * (W < 0)
        s1 = np.sum(W1)
        if s1 > 0:
            B1 = W1 - gamma * np.outer(np.sum(W1, axis=1), np.sum(W1, axis=0)) / s1
        else:
            B1 = np.zeros_like(W)
        
        if B_type == 'negative_sym':
            B = B0 / (s0 + s1) - B1 / (s0 + s1)
        else:
            B = B0 / s0 - B1 / (s0 + s1)
    else:
        raise ValueError(f"Unknown B_type: {B_type}")
    
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    
    M, Q = _community_louvain_core(B, seed)
    
    return M, Q


def consensus_und(d: np.ndarray, tau: float, reps: int, 
                  seed: int = None, verbose: bool = False,
                  parallel: bool = False) -> np.ndarray:
    """
    Consensus clustering of agreement matrix.
    
    Parameters
    ----------
    d : np.ndarray
        Agreement matrix with entries between 0 and 1 denoting the probability
        of finding node i in the same cluster as node j.
    tau : float
        Threshold which controls the resolution of the reclustering.
    reps : int
        Number of times that the clustering algorithm is reapplied.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
    parallel : bool
        Whether to use parallel processing (default: False).
        Note: Currently not implemented, reserved for future use.
    
    Returns
    -------
    ciu : np.ndarray
        Consensus partition (column vector).
    
    References
    ----------
    Lancichinetti & Fortunato (2012). Consensus clustering in complex networks.
    Scientific Reports 2, Article number: 336.
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    
    if seed is not None:
        np.random.seed(seed)
    
    flg = 1
    iteration = 0
    
    while flg == 1:
        iteration += 1
        flg = 0
        
        # Thresholding: dt = d.*(d >= tau).*~eye(n) in MATLAB
        dt = d * (d >= tau)
        np.fill_diagonal(dt, 0)
        
        if verbose:
            print(f"Iteration {iteration}: nnz(dt) = {np.count_nonzero(dt)}")
        
        if np.count_nonzero(dt) == 0:
            ciu = np.arange(1, n + 1, dtype=np.int64).reshape(-1, 1)
        else:
            # Run community_louvain reps times
            ci = np.zeros((n, reps), dtype=np.int64)
            for i in range(reps):
                ci[:, i], _ = community_louvain(dt)
            
            # Relabel partitions and get unique ones
            ci = relabel_partitions(ci)
            ciu = unique_partitions(ci)
            
            nu = ciu.shape[1]
            if verbose:
                print(f"  Number of unique partitions: {nu}")
            
            if nu > 1:
                flg = 1
                # Compute new agreement matrix
                d = agreement(ci) / reps
    
    return ciu


def consensus_und_single(d: np.ndarray, tau: float, reps: int,
                         seed: int = None, verbose: bool = False,
                         parallel: bool = False) -> np.ndarray:
    """
    Consensus clustering returning a 1D partition vector.
    """
    ciu = consensus_und(d, tau, reps, seed, verbose, parallel)
    return ciu.flatten()


if __name__ == "__main__":
    import time
    
    np.random.seed(42)
    n = 100
    
    # Create block-structured test matrix
    d = np.random.rand(n, n) * 0.1
    for i in range(4):
        start = i * 25
        end = (i + 1) * 25
        d[start:end, start:end] = 0.8 + np.random.rand(25, 25) * 0.15
    
    d = (d + d.T) / 2
    np.fill_diagonal(d, 1)
    
    tau = 0.3
    reps = 10
    
    print("Running consensus_und...")
    start = time.time()
    result = consensus_und(d, tau, reps, seed=42, verbose=True)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.3f}s")
    print(f"Result shape: {result.shape}")
    print(f"Unique communities: {np.unique(result.flatten())}")
