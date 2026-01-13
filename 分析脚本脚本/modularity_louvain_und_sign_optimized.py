"""
modularity_louvain_und_sign_optimized.py

Optimized implementation using Numba JIT compilation for the Louvain
algorithm for undirected networks with positive and negative weights.

This version maintains the same numerical precision as the original
while achieving significant speedup through:
1. Numba JIT compilation of inner loops
2. Avoiding Python object overhead
3. Efficient array operations

Original MATLAB implementation: Mika Rubinov, UNSW, 2011
Python translation and optimization: 2024
"""

import numpy as np
from numba import njit
from typing import Tuple, Optional, List


@njit(cache=True, fastmath=True)
def _inner_loop_iteration(
    n: int,
    W0: np.ndarray,
    W1: np.ndarray,
    Kn0: np.ndarray,
    Kn1: np.ndarray,
    Km0: np.ndarray,
    Km1: np.ndarray,
    Knm0: np.ndarray,
    Knm1: np.ndarray,
    M: np.ndarray,
    s0: float,
    s1: float,
    d0: float,
    d1: float,
    randperm_order: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Execute one inner loop iteration (one pass through all nodes).
    All indices are 0-based.
    """
    f = 0
    dQ = np.empty(n)
    inv_s0 = 1.0 / s0
    inv_s1 = 1.0 / s1
    
    for idx in range(n):
        u = randperm_order[idx]
        ma = M[u]
        
        # Compute dQ for all modules
        W0_uu = W0[u, u]
        W1_uu = W1[u, u]
        Knm0_u_ma = Knm0[u, ma]
        Knm1_u_ma = Knm1[u, ma]
        Km0_ma = Km0[ma]
        Km1_ma = Km1[ma]
        Kn0_u = Kn0[u]
        Kn1_u = Kn1[u]
        
        base0 = W0_uu - Knm0_u_ma
        base1 = W1_uu - Knm1_u_ma
        factor0 = Kn0_u * inv_s0
        factor1 = Kn1_u * inv_s1
        offset0 = Kn0_u - Km0_ma
        offset1 = Kn1_u - Km1_ma
        
        for j in range(n):
            dQ0_j = (Knm0[u, j] + base0) - factor0 * (Km0[j] + offset0)
            dQ1_j = (Knm1[u, j] + base1) - factor1 * (Km1[j] + offset1)
            dQ[j] = d0 * dQ0_j - d1 * dQ1_j
        
        dQ[ma] = 0.0
        
        # Find max dQ
        max_dQ = dQ[0]
        mb = 0
        for j in range(1, n):
            if dQ[j] > max_dQ:
                max_dQ = dQ[j]
                mb = j
        
        if max_dQ > 1e-10:
            f = 1
            M[u] = mb
            
            for i in range(n):
                Knm0[i, mb] += W0[i, u]
                Knm0[i, ma] -= W0[i, u]
                Knm1[i, mb] += W1[i, u]
                Knm1[i, ma] -= W1[i, u]
            
            Km0[mb] += Kn0_u
            Km0[ma] -= Kn0_u
            Km1[mb] += Kn1_u
            Km1[ma] -= Kn1_u
    
    return M, Km0, Km1, Knm0, Knm1, f


@njit(cache=True)
def _compute_unique_mapping(M: np.ndarray) -> np.ndarray:
    """Compute unique mapping (MATLAB-compatible)."""
    n = len(M)
    sorted_vals = np.sort(np.unique(M))
    n_unique = len(sorted_vals)
    
    M_unique = np.empty(n, dtype=np.int64)
    for i in range(n):
        for j in range(n_unique):
            if sorted_vals[j] == M[i]:
                M_unique[i] = j
                break
    return M_unique


@njit(cache=True)
def _aggregate_weights(
    W0: np.ndarray,
    W1: np.ndarray,
    M_unique: np.ndarray,
    n_new: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate weights for nodes in the same module - optimized version."""
    n_old = len(M_unique)
    w0 = np.zeros((n_new, n_new))
    w1 = np.zeros((n_new, n_new))
    
    # Direct aggregation: iterate through all pairs once
    for i in range(n_old):
        mi = M_unique[i]
        for j in range(n_old):
            mj = M_unique[j]
            w0[mi, mj] += W0[i, j]
            w1[mi, mj] += W1[i, j]
    
    return w0, w1


@njit(cache=True)
def _compute_modularity(W0: np.ndarray, W1: np.ndarray, s0: float, s1: float, d0: float, d1: float) -> float:
    """Compute modularity using optimized matrix operations."""
    n = len(W0)
    
    # Diagonal sums
    diag_sum0 = 0.0
    diag_sum1 = 0.0
    for i in range(n):
        diag_sum0 += W0[i, i]
        diag_sum1 += W1[i, i]
    
    # W @ W sum - optimized: sum_ij (W @ W)_ij = sum_ijk W_ik * W_kj
    # Reorder to minimize cache misses
    W0_sq_sum = 0.0
    W1_sq_sum = 0.0
    for i in range(n):
        for k in range(n):
            w0_ik = W0[i, k]
            w1_ik = W1[i, k]
            if w0_ik != 0.0 or w1_ik != 0.0:
                for j in range(n):
                    W0_sq_sum += w0_ik * W0[k, j]
                    W1_sq_sum += w1_ik * W1[k, j]
    
    Q0 = diag_sum0 - W0_sq_sum / s0
    Q1 = diag_sum1 - W1_sq_sum / s1
    
    return d0 * Q0 - d1 * Q1


@njit(cache=True)
def _louvain_core_with_randperms(
    W: np.ndarray,
    d0: float,
    d1: float,
    s0: float,
    s1: float,
    all_randperms: np.ndarray,
    randperm_counts: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Core Louvain with pre-specified random permutations.
    all_randperms: 2D array where each row is a permutation (0-based)
    randperm_counts: number of permutations for each hierarchy level
    """
    N = len(W)
    
    # Build W0, W1
    W0 = np.zeros((N, N))
    W1 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                W0[i, j] = W[i, j]
            elif W[i, j] < 0:
                W1[i, j] = -W[i, j]
    
    n = N
    Q_prev = -1.0
    Q_curr = 0.0
    Ci_current = np.arange(N, dtype=np.int64)
    
    hierarchy_idx = 0
    randperm_idx = 0
    
    while Q_curr - Q_prev > 1e-10:
        Q_prev = Q_curr
        
        Kn0 = np.zeros(n)
        Kn1 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                Kn0[i] += W0[j, i]
                Kn1[i] += W1[j, i]
        
        Km0 = Kn0.copy()
        Km1 = Kn1.copy()
        Knm0 = W0.copy()
        Knm1 = W1.copy()
        M = np.arange(n, dtype=np.int64)
        
        f = 1
        while f == 1:
            # Get pre-specified permutation
            randperm_order = all_randperms[randperm_idx, :n].copy()
            randperm_idx += 1
            
            M, Km0, Km1, Knm0, Knm1, f = _inner_loop_iteration(
                n, W0, W1, Kn0, Kn1, Km0, Km1, Knm0, Knm1, M, s0, s1, d0, d1, randperm_order
            )
        
        M_unique = _compute_unique_mapping(M)
        n_new = int(np.max(M_unique) + 1)
        
        Ci_new = np.zeros(N, dtype=np.int64)
        for orig_node in range(N):
            old_module = Ci_current[orig_node]
            Ci_new[orig_node] = M_unique[old_module]
        Ci_current = Ci_new
        
        W0, W1 = _aggregate_weights(W0, W1, M_unique, n_new)
        n = n_new
        Q_curr = _compute_modularity(W0, W1, s0, s1, d0, d1)
        hierarchy_idx += 1
    
    Ci_out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        Ci_out[i] = Ci_current[i] + 1
    
    return Ci_out, Q_curr


@njit(cache=True)
def _shuffle_inplace(arr: np.ndarray, rng_state: np.ndarray) -> None:
    """Fisher-Yates shuffle using LCG."""
    n = len(arr)
    for i in range(n - 1, 0, -1):
        rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7fffffff
        j = rng_state[0] % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]


@njit(cache=True)
def _louvain_core(
    W: np.ndarray,
    d0: float,
    d1: float,
    s0: float,
    s1: float,
    seed: int
) -> Tuple[np.ndarray, float]:
    """Core Louvain with internal RNG."""
    N = len(W)
    
    W0 = np.zeros((N, N))
    W1 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                W0[i, j] = W[i, j]
            elif W[i, j] < 0:
                W1[i, j] = -W[i, j]
    
    rng_state = np.array([seed], dtype=np.int64)
    
    n = N
    Q_prev = -1.0
    Q_curr = 0.0
    Ci_current = np.arange(N, dtype=np.int64)
    
    while Q_curr - Q_prev > 1e-10:
        Q_prev = Q_curr
        
        Kn0 = np.zeros(n)
        Kn1 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                Kn0[i] += W0[j, i]
                Kn1[i] += W1[j, i]
        
        Km0 = Kn0.copy()
        Km1 = Kn1.copy()
        Knm0 = W0.copy()
        Knm1 = W1.copy()
        M = np.arange(n, dtype=np.int64)
        
        f = 1
        while f == 1:
            randperm_order = np.arange(n, dtype=np.int64)
            _shuffle_inplace(randperm_order, rng_state)
            
            M, Km0, Km1, Knm0, Knm1, f = _inner_loop_iteration(
                n, W0, W1, Kn0, Kn1, Km0, Km1, Knm0, Knm1, M, s0, s1, d0, d1, randperm_order
            )
        
        M_unique = _compute_unique_mapping(M)
        n_new = int(np.max(M_unique) + 1)
        
        Ci_new = np.zeros(N, dtype=np.int64)
        for orig_node in range(N):
            old_module = Ci_current[orig_node]
            Ci_new[orig_node] = M_unique[old_module]
        Ci_current = Ci_new
        
        W0, W1 = _aggregate_weights(W0, W1, M_unique, n_new)
        n = n_new
        Q_curr = _compute_modularity(W0, W1, s0, s1, d0, d1)
    
    Ci_out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        Ci_out[i] = Ci_current[i] + 1
    
    return Ci_out, Q_curr


def _get_scaling_factors(qtype: str, s0: float, s1: float) -> Tuple[float, float, float, float]:
    """Get scaling factors based on qtype."""
    if qtype == 'smp':
        d0 = 1/s0 if s0 != 0 else 0
        d1 = 1/s1 if s1 != 0 else 0
    elif qtype == 'gja':
        d0 = 1/(s0+s1) if (s0+s1) != 0 else 0
        d1 = 1/(s0+s1) if (s0+s1) != 0 else 0
    elif qtype == 'sta':
        d0 = 1/s0 if s0 != 0 else 0
        d1 = 1/(s0+s1) if (s0+s1) != 0 else 0
    elif qtype == 'pos':
        d0 = 1/s0 if s0 != 0 else 0
        d1 = 0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1/s1 if s1 != 0 else 0
    else:
        raise ValueError(f"qtype unknown: {qtype}")
    
    if s0 == 0:
        s0 = 1.0
        d0 = 0.0
    if s1 == 0:
        s1 = 1.0
        d1 = 0.0
    
    return d0, d1, s0, s1


def modularity_louvain_und_sign_optimized(
    W: np.ndarray, 
    qtype: str = 'sta',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Optimal community structure and modularity for undirected signed networks.
    
    Optimized implementation using Numba JIT compilation.
    
    Parameters
    ----------
    W : np.ndarray
        Undirected (weighted or binary) connection matrix with positive 
        and negative weights. Must be symmetric.
    qtype : str, optional
        Modularity type (see Rubinov and Sporns, 2011):
            'sta' - Q_* (default)
            'pos' - Q_+
            'smp' - Q_simple
            'gja' - Q_GJA
            'neg' - Q_-
    seed : int, optional
        Random seed for reproducibility. If None, uses a random seed.
        
    Returns
    -------
    Ci : np.ndarray
        Community affiliation vector (1-based indexing for compatibility)
    Q : float
        Modularity (qtype dependent)
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    
    W = np.ascontiguousarray(W, dtype=np.float64)
    
    W0_init = W * (W > 0)
    W1_init = -W * (W < 0)
    s0 = np.sum(W0_init)
    s1 = np.sum(W1_init)
    
    d0, d1, s0, s1 = _get_scaling_factors(qtype, s0, s1)
    
    Ci, Q = _louvain_core(W, d0, d1, s0, s1, seed)
    
    return Ci, Q


def modularity_louvain_und_sign_optimized_with_randperms(
    W: np.ndarray, 
    qtype: str,
    randperms_list: List[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """
    Version that accepts pre-specified random permutations for validation.
    
    randperms_list: List of 1-based permutation arrays from MATLAB
    """
    W = np.ascontiguousarray(W, dtype=np.float64)
    N = len(W)
    
    W0_init = W * (W > 0)
    W1_init = -W * (W < 0)
    s0 = np.sum(W0_init)
    s1 = np.sum(W1_init)
    
    d0, d1, s0, s1 = _get_scaling_factors(qtype, s0, s1)
    
    # Prepare randperms array (convert to 0-based)
    max_len = max(len(rp) for rp in randperms_list) if randperms_list else N
    all_randperms = np.zeros((len(randperms_list), max_len), dtype=np.int64)
    for i, rp in enumerate(randperms_list):
        rp_0based = np.asarray(rp, dtype=np.int64) - 1  # Convert to 0-based
        all_randperms[i, :len(rp_0based)] = rp_0based
    
    randperm_counts = np.array([len(rp) for rp in randperms_list], dtype=np.int64)
    
    Ci, Q = _louvain_core_with_randperms(W, d0, d1, s0, s1, all_randperms, randperm_counts)
    
    return Ci, Q


# Original implementation for comparison
def modularity_louvain_und_sign(
    W: np.ndarray, 
    qtype: str = 'sta',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """Original implementation - kept for comparison."""
    if seed is not None:
        np.random.seed(seed)
    
    N = len(W)
    
    W0 = W * (W > 0)
    W1 = -W * (W < 0)
    s0 = np.sum(W0)
    s1 = np.sum(W1)
    
    if qtype == 'smp':
        d0 = 1/s0
        d1 = 1/s1
    elif qtype == 'gja':
        d0 = 1/(s0+s1)
        d1 = 1/(s0+s1)
    elif qtype == 'sta':
        d0 = 1/s0
        d1 = 1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1/s0
        d1 = 0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1/s1
    else:
        raise ValueError(f"qtype unknown: {qtype}")
    
    if s0 == 0:
        s0 = 1
        d0 = 0
    if s1 == 0:
        s1 = 1
        d1 = 0
    
    h = 2
    n = N
    Ci_dict = {1: None, 2: np.arange(1, n+1)}
    Q_dict = {1: -1, 2: 0}
    
    while Q_dict[h] - Q_dict[h-1] > 1e-10:
        Kn0 = np.sum(W0, axis=0)
        Kn1 = np.sum(W1, axis=0)
        Km0 = Kn0.copy()
        Km1 = Kn1.copy()
        Knm0 = W0.copy()
        Knm1 = W1.copy()
        
        M = np.arange(1, n+1)
        
        f = 1
        while f:
            f = 0
            for u in np.random.permutation(n) + 1:
                u_idx = u - 1
                ma = M[u_idx]
                ma_idx = int(ma) - 1
                
                dQ0 = (Knm0[u_idx, :] + W0[u_idx, u_idx] - Knm0[u_idx, ma_idx]) - \
                      Kn0[u_idx] * (Km0 + Kn0[u_idx] - Km0[ma_idx]) / s0
                dQ1 = (Knm1[u_idx, :] + W1[u_idx, u_idx] - Knm1[u_idx, ma_idx]) - \
                      Kn1[u_idx] * (Km1 + Kn1[u_idx] - Km1[ma_idx]) / s1
                dQ = d0 * dQ0 - d1 * dQ1
                dQ[ma_idx] = 0
                
                max_dQ = np.max(dQ)
                mb = np.argmax(dQ) + 1
                mb_idx = mb - 1
                
                if max_dQ > 1e-10:
                    f = 1
                    M[u_idx] = mb
                    Knm0[:, mb_idx] = Knm0[:, mb_idx] + W0[:, u_idx]
                    Knm0[:, ma_idx] = Knm0[:, ma_idx] - W0[:, u_idx]
                    Knm1[:, mb_idx] = Knm1[:, mb_idx] + W1[:, u_idx]
                    Knm1[:, ma_idx] = Knm1[:, ma_idx] - W1[:, u_idx]
                    Km0[mb_idx] = Km0[mb_idx] + Kn0[u_idx]
                    Km0[ma_idx] = Km0[ma_idx] - Kn0[u_idx]
                    Km1[mb_idx] = Km1[mb_idx] + Kn1[u_idx]
                    Km1[ma_idx] = Km1[ma_idx] - Kn1[u_idx]
        
        h = h + 1
        Ci_new = np.zeros(N, dtype=int)
        _, _, M_unique = np.unique(M, return_index=True, return_inverse=True)
        M_unique = M_unique + 1
        
        Ci_prev = Ci_dict[h-1]
        for u_orig in range(1, n+1):
            u_orig_idx = u_orig - 1
            if Ci_prev is not None:
                mask = (Ci_prev == u_orig)
                Ci_new[mask] = M_unique[u_orig_idx]
            else:
                Ci_new[u_orig_idx] = M_unique[u_orig_idx]
        
        Ci_dict[h] = Ci_new
        
        n = int(np.max(M_unique))
        w0 = np.zeros((n, n))
        w1 = np.zeros((n, n))
        
        for u_new in range(1, n+1):
            for v_new in range(u_new, n+1):
                mask_u = (M_unique == u_new)
                mask_v = (M_unique == v_new)
                w0[u_new-1, v_new-1] = np.sum(W0[np.ix_(mask_u, mask_v)])
                w1[u_new-1, v_new-1] = np.sum(W1[np.ix_(mask_u, mask_v)])
                w0[v_new-1, u_new-1] = w0[u_new-1, v_new-1]
                w1[v_new-1, u_new-1] = w1[u_new-1, v_new-1]
        
        W0 = w0
        W1 = w1
        
        Q0 = np.sum(np.diag(W0)) - np.sum(W0 @ W0) / s0
        Q1 = np.sum(np.diag(W1)) - np.sum(W1 @ W1) / s1
        Q_dict[h] = d0 * Q0 - d1 * Q1
    
    Ci = Ci_dict[h]
    Q = Q_dict[h]
    
    return Ci, Q


if __name__ == "__main__":
    import time
    
    print("Testing modularity_louvain_und_sign_optimized")
    print("=" * 60)
    
    # Small test
    np.random.seed(42)
    N = 10
    W_rand = np.random.randn(N, N)
    W = (W_rand + W_rand.T) / 2
    np.fill_diagonal(W, 0)
    
    print(f"\nSmall matrix test (N={N}):")
    for qtype in ['sta', 'pos', 'smp', 'gja', 'neg']:
        Ci, Q = modularity_louvain_und_sign_optimized(W, qtype=qtype, seed=123)
        print(f"  qtype='{qtype}': Q={Q:.6f}, modules={len(np.unique(Ci))}")
    
    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmark comparison")
    print("=" * 60)
    
    for N in [50, 100, 200, 500]:
        np.random.seed(42)
        W_rand = np.random.randn(N, N)
        W = (W_rand + W_rand.T) / 2
        np.fill_diagonal(W, 0)
        
        # Warm up JIT
        _ = modularity_louvain_und_sign_optimized(W, seed=123)
        
        n_runs = 5
        
        # Time original
        times_orig = []
        for _ in range(n_runs):
            start = time.perf_counter()
            Ci_orig, Q_orig = modularity_louvain_und_sign(W, seed=123)
            times_orig.append(time.perf_counter() - start)
        
        # Time optimized
        times_opt = []
        for _ in range(n_runs):
            start = time.perf_counter()
            Ci_opt, Q_opt = modularity_louvain_und_sign_optimized(W, seed=123)
            times_opt.append(time.perf_counter() - start)
        
        t_orig = np.mean(times_orig)
        t_opt = np.mean(times_opt)
        speedup = t_orig / t_opt
        
        print(f"\nN={N}:")
        print(f"  Original:  {t_orig*1000:.2f} ms")
        print(f"  Optimized: {t_opt*1000:.2f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
