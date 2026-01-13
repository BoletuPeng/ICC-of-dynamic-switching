"""
modularity_und_optimized.py - Optimal community structure and modularity (Numba optimized)

Translated from MATLAB Brain Connectivity Toolbox with Numba JIT optimization
Original authors: Mika Rubinov, Jonathan Power, Dani Bassett, Xindi Wang, Roan LaPlante

This is a performance-optimized version using Numba JIT compilation.
It maintains numerical compatibility with the original MATLAB implementation.

Usage:
    Ci = modularity_und(W)
    Ci, Q = modularity_und(W, gamma)
"""

import numpy as np
from numba import njit
import scipy.linalg as la
from scipy.sparse.linalg import eigsh


# Threshold for using sparse eigenvalue solver
SPARSE_EIGSH_THRESHOLD = 20


@njit(cache=True, fastmath=True)
def _finetune_loop_optimized(Bg, S, q, Ng):
    """
    Highly optimized JIT-compiled fine-tuning loop.
    
    Key optimizations:
    - Incremental update of Bg @ Sit instead of full recomputation
    - Fused computation of Qit and max finding
    - Minimized memory allocations
    """
    qmax = q
    q_best = q
    S_best = S.copy()
    Sit = S.copy()
    
    # Pre-compute initial Bg @ Sit
    Bg_Sit = np.zeros(Ng)
    for i in range(Ng):
        acc = 0.0
        for j in range(Ng):
            acc += Bg[i, j] * Sit[j]
        Bg_Sit[i] = acc
    
    # Track moved nodes: False = unmoved, True = moved
    moved = np.zeros(Ng, dtype=np.bool_)
    
    for _ in range(Ng):
        # Find best node to flip among unmoved nodes
        qmax_new = -np.inf
        imax = 0
        
        for i in range(Ng):
            if not moved[i]:
                Qit_i = qmax - 4.0 * Sit[i] * Bg_Sit[i]
                if Qit_i > qmax_new:
                    qmax_new = Qit_i
                    imax = i
        
        # Flip the selected node and mark as moved
        old_Sit_imax = Sit[imax]
        Sit[imax] = -old_Sit_imax
        moved[imax] = True
        
        # Incrementally update Bg_Sit: Bg_Sit += 2 * (-old_Sit_imax) * Bg[:, imax]
        # Because Sit[imax] changed from old to -old, difference is -2*old
        delta = -2.0 * old_Sit_imax
        for i in range(Ng):
            Bg_Sit[i] += delta * Bg[i, imax]
        
        # Update best partition if improved (with tolerance for MATLAB compatibility)
        if qmax_new > q_best - 1e-14:
            q_best = qmax_new
            for i in range(Ng):
                S_best[i] = Sit[i]
        
        qmax = qmax_new
    
    return S_best, q_best


@njit(cache=True)
def _compute_final_Q(Ci, B, m, N):
    """
    JIT-compiled final modularity computation.
    
    Q = sum over i,j of (Ci[i] == Ci[j]) * B[i,j] / m
    """
    Q = 0.0
    for i in range(N):
        for j in range(N):
            if Ci[i] == Ci[j]:
                Q += B[i, j]
    return Q / m


@njit(cache=True)
def _extract_submatrix(B, ind, Ng):
    """Extract submatrix and compute modified modularity matrix."""
    bg = np.zeros((Ng, Ng))
    for i in range(Ng):
        for j in range(Ng):
            bg[i, j] = B[ind[i], ind[j]]
    
    # Bg = bg - diag(sum(bg, axis=0))
    Bg = bg.copy()
    for i in range(Ng):
        col_sum = 0.0
        for j in range(Ng):
            col_sum += bg[j, i]
        Bg[i, i] -= col_sum
    
    return Bg


def _get_leading_eigenvector(Bg, Ng):
    """
    Get the eigenvector corresponding to the largest eigenvalue.
    
    Uses sparse solver for large matrices (faster) and dense solver for small matrices.
    """
    if Ng >= SPARSE_EIGSH_THRESHOLD:
        # Use sparse solver - only compute one eigenvalue/vector
        try:
            # eigsh finds k largest eigenvalues of symmetric matrix
            eigenvalues, V = eigsh(Bg, k=1, which='LA', tol=1e-10)
            return V[:, 0], eigenvalues[0]
        except:
            # Fall back to dense solver if sparse fails
            pass
    
    # Dense solver for small matrices or fallback
    eigenvalues, V = la.eigh(Bg)
    # eigh returns in ascending order, so largest is last
    return V[:, -1], eigenvalues[-1]


def modularity_und(A, gamma=1.0):
    """
    Compute optimal community structure and modularity for undirected networks.
    
    This is the Numba-optimized version that maintains numerical compatibility
    with the original MATLAB implementation.
    
    Parameters
    ----------
    A : numpy.ndarray
        Undirected weighted/binary connection matrix (must be symmetric)
    gamma : float, optional
        Resolution parameter. Default is 1.0 (classic modularity)
        
    Returns
    -------
    Ci : numpy.ndarray
        Community assignment for each node (1-indexed)
    Q : float
        Maximized modularity value
    """
    A = np.asarray(A, dtype=np.float64)
    
    N = len(A)
    K = np.sum(A, axis=0)
    m = np.sum(K)
    
    # Modularity matrix: B = A - gamma * (K' * K) / m
    B = A - gamma * np.outer(K, K) / m
    
    Ci = np.ones(N, dtype=np.int32)
    cn = 1
    
    # Use numpy arrays instead of lists for community queue
    U = np.zeros(N + 1, dtype=np.int32)
    U[0] = 1
    
    ind = np.arange(N, dtype=np.int32)
    Bg = B.copy()
    Ng = N
    
    while U[0] != 0:
        # Get leading eigenvector
        v1, max_eigenval = _get_leading_eigenvector(Bg, Ng)
        
        # Initial partition based on eigenvector sign
        S = np.ones(Ng, dtype=np.float64)
        S[v1 < 0] = -1.0
        
        # Contribution to modularity: q = S' * Bg * S
        q = float(S @ Bg @ S)
        
        if q > 1e-10:
            # Set diagonal to 0 for fine-tuning
            Bg_mod = Bg.copy()
            np.fill_diagonal(Bg_mod, 0.0)
            
            # JIT-compiled fine-tuning with incremental updates
            S, q = _finetune_loop_optimized(Bg_mod, S, q, Ng)
            S = S.astype(np.int32)
            
            # Check if split was successful
            if abs(np.sum(S)) == Ng:
                # Unsuccessful split - remove U[0]
                U[:-1] = U[1:]
            else:
                cn += 1
                # Split old U[0] into new U[0] and cn
                Ci[ind[S == 1]] = U[0]
                Ci[ind[S == -1]] = cn
                # Prepend cn to U
                U[1:] = U[:-1]
                U[0] = cn
        else:
            # Contribution nonpositive - remove U[0]
            U[:-1] = U[1:]
        
        # Prepare for next iteration
        if U[0] != 0:
            ind = np.where(Ci == U[0])[0].astype(np.int32)
            Ng = len(ind)
            Bg = _extract_submatrix(B, ind, Ng)
    
    # Compute final modularity using JIT-compiled function
    Q = _compute_final_Q(Ci.astype(np.int32), B, m, N)
    
    return Ci, Q


# Convenience function for compatibility
def modularity(A, gamma=1.0):
    """Alias for modularity_und."""
    return modularity_und(A, gamma)


# Pre-compile JIT functions on module load
def _warmup():
    """Warmup JIT compilation with small arrays."""
    small_B = np.random.randn(4, 4)
    small_B = (small_B + small_B.T) / 2
    np.fill_diagonal(small_B, 0)
    small_S = np.array([1.0, 1.0, -1.0, -1.0])
    _finetune_loop_optimized(small_B, small_S, 0.1, 4)
    
    small_Ci = np.array([1, 1, 2, 2], dtype=np.int32)
    _compute_final_Q(small_Ci, small_B, 1.0, 4)
    
    ind = np.array([0, 1], dtype=np.int32)
    _extract_submatrix(small_B, ind, 2)


# Run warmup on import
_warmup()


if __name__ == '__main__':
    import time
    from scipy.io import loadmat
    
    # Load test data
    print("Loading MATLAB test data...")
    mat_data = loadmat('modularity_test_data.mat', squeeze_me=True)
    A = mat_data['A']
    gamma = float(mat_data['gamma'])
    mat_Q = float(mat_data['Q'])
    mat_Ci = mat_data['Ci_final']
    
    print(f"Matrix size: {A.shape}")
    print(f"MATLAB Q: {mat_Q:.10f}")
    print(f"MATLAB Ci: {mat_Ci}")
    
    # Run optimized version
    print("\n--- Running optimized version ---")
    Ci, Q = modularity_und(A, gamma)
    
    print(f"Python Q: {Q:.10f}")
    print(f"Python Ci: {Ci}")
    print(f"Q difference: {abs(Q - mat_Q):.2e}")
    
    # Check equivalence
    def check_Ci_equivalent(py_Ci, mat_Ci):
        mapping = {}
        for p, m in zip(py_Ci, mat_Ci):
            if p in mapping:
                if mapping[p] != m:
                    return False
            else:
                mapping[p] = m
        return len(set(mapping.values())) == len(mapping)
    
    ci_match = check_Ci_equivalent(Ci, mat_Ci)
    q_match = abs(Q - mat_Q) < 1e-10
    
    print(f"\nResults match MATLAB: Q={q_match}, Ci={ci_match}")
    
    # Benchmark
    print("\n--- Benchmarking ---")
    
    # Warmup runs
    for _ in range(3):
        modularity_und(A, gamma)
    
    # Timing runs
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        modularity_und(A, gamma)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000
    print(f"Average time per call ({n_runs} runs): {avg_time:.3f} ms")
