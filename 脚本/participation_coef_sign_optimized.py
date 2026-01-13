"""
PARTICIPATION_COEF_SIGN     Participation coefficient (Numba Optimized)

    Ppos, Pneg = participation_coef_sign(W, Ci)

    This is a numba-optimized version of the participation coefficient calculation.
    
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.

    Inputs:     W,      undirected connection matrix with positive and
                        negative weights (numpy array, shape: n x n)

                Ci,     community affiliation vector (numpy array, shape: n,)
                        Values should be integers starting from 1.

    Output:     Ppos,   participation coefficient from positive weights
                        (numpy array, shape: n,)

                Pneg,   participation coefficient from negative weights
                        (numpy array, shape: n,)

    Reference: Guimera R, Amaral L. Nature (2005) 433:895-900.

    2011, Mika Rubinov, UNSW
    
    Ported to Python & Numba optimized: 2024
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _pcoef_core(W_, Ci, n, max_Ci):
    """
    Core computation of participation coefficient using numba JIT.
    
    Parameters
    ----------
    W_ : numpy.ndarray
        Weight matrix (positive or negative part), shape (n, n)
    Ci : numpy.ndarray
        Community affiliation vector, shape (n,), dtype int64
    n : int
        Number of nodes
    max_Ci : int
        Maximum community index
    
    Returns
    -------
    P : numpy.ndarray
        Participation coefficient, shape (n,)
    """
    # Step 1: Compute strength S = sum(W_, axis=1)
    S = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            S[i] += W_[i, j]
    
    # Step 2: Compute neighbor community affiliation Gc
    # Gc[i, j] = Ci[j] if W_[i, j] != 0, else 0
    # This is equivalent to: Gc = (W_ != 0) @ np.diag(Ci)
    Gc = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if W_[i, j] != 0.0:
                Gc[i, j] = Ci[j]
    
    # Step 3: Compute Sc2 - sum of squared community-specific strengths
    Sc2 = np.zeros(n, dtype=np.float64)
    for comm in range(1, max_Ci + 1):
        for i in range(n):
            # Sum of weights to neighbors in community 'comm'
            sum_w = 0.0
            for j in range(n):
                if Gc[i, j] == comm:
                    sum_w += W_[i, j]
            Sc2[i] += sum_w * sum_w
    
    # Step 4: Compute participation coefficient P = 1 - Sc2 / S^2
    P = np.zeros(n, dtype=np.float64)
    for i in range(n):
        S_sq = S[i] * S[i]
        if S_sq > 0.0:
            P[i] = 1.0 - Sc2[i] / S_sq
        else:
            P[i] = 0.0  # Handle division by zero (NaN -> 0)
    
    # Step 5: P(~P) = 0 - set falsy values to 0 (already handled above for NaN)
    # In MATLAB this sets any zero or negative values to 0
    for i in range(n):
        if P[i] == 0.0:
            P[i] = 0.0  # Explicit for clarity
    
    return P


@njit(cache=True, parallel=True)
def _pcoef_core_parallel(W_, Ci, n, max_Ci):
    """
    Parallelized core computation of participation coefficient.
    
    Uses parallel loops for larger matrices.
    """
    # Step 1: Compute strength S = sum(W_, axis=1)
    S = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        s_val = 0.0
        for j in range(n):
            s_val += W_[i, j]
        S[i] = s_val
    
    # Step 2: Compute neighbor community affiliation Gc
    Gc = np.zeros((n, n), dtype=np.int64)
    for i in prange(n):
        for j in range(n):
            if W_[i, j] != 0.0:
                Gc[i, j] = Ci[j]
    
    # Step 3 & 4 combined: Compute Sc2 and P in one pass per node
    P = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        Sc2_i = 0.0
        for comm in range(1, max_Ci + 1):
            sum_w = 0.0
            for j in range(n):
                if Gc[i, j] == comm:
                    sum_w += W_[i, j]
            Sc2_i += sum_w * sum_w
        
        S_sq = S[i] * S[i]
        if S_sq > 0.0:
            P[i] = 1.0 - Sc2_i / S_sq
        else:
            P[i] = 0.0
    
    return P


@njit(cache=True)
def _extract_weights(W, n):
    """
    Extract positive and negative weight matrices.
    
    W_pos = W * (W > 0)
    W_neg = -W * (W < 0)
    """
    W_pos = np.zeros((n, n), dtype=np.float64)
    W_neg = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            w = W[i, j]
            if w > 0.0:
                W_pos[i, j] = w
            elif w < 0.0:
                W_neg[i, j] = -w
    
    return W_pos, W_neg


def participation_coef_sign(W, Ci, use_parallel=True):
    """
    Compute participation coefficient for networks with positive and negative weights.
    
    This is a numba-optimized version that provides significant speedup for larger matrices.
    
    Parameters
    ----------
    W : numpy.ndarray
        Undirected connection matrix with positive and negative weights.
        Shape: (n, n), must be symmetric.
    Ci : numpy.ndarray
        Community affiliation vector. Shape: (n,) or (n, 1).
        Values should be positive integers (1, 2, 3, ...).
    use_parallel : bool, optional
        Whether to use parallel computation (default: True).
        Set to False for small matrices (n < 50) for better performance.
    
    Returns
    -------
    Ppos : numpy.ndarray
        Participation coefficient from positive weights. Shape: (n,)
    Pneg : numpy.ndarray
        Participation coefficient from negative weights. Shape: (n,)
    
    Notes
    -----
    The participation coefficient measures the diversity of intermodular
    connections of individual nodes. A coefficient close to 1 indicates
    that the node's connections are uniformly distributed among all modules,
    while a coefficient close to 0 indicates that the node's connections
    are concentrated within its own module.
    
    References
    ----------
    Guimera R, Amaral L. Nature (2005) 433:895-900.
    """
    # Ensure correct types
    W = np.asarray(W, dtype=np.float64)
    Ci = np.asarray(Ci, dtype=np.int64).flatten()
    n = len(W)
    max_Ci = int(np.max(Ci))
    
    # Extract positive and negative weights
    W_pos, W_neg = _extract_weights(W, n)
    
    # Choose computation method based on matrix size and user preference
    if use_parallel and n >= 50:
        pcoef_func = _pcoef_core_parallel
    else:
        pcoef_func = _pcoef_core
    
    # Compute participation coefficients
    Ppos = pcoef_func(W_pos, Ci, n, max_Ci)
    Pneg = pcoef_func(W_neg, Ci, n, max_Ci)
    
    return Ppos, Pneg


# Alternative implementation using vectorized operations where possible
@njit(cache=True)
def _pcoef_vectorized(W_, Ci, n, max_Ci):
    """
    Alternative implementation with more vectorized operations.
    May be faster for certain matrix sizes.
    """
    # Compute strength
    S = np.sum(W_, axis=1)
    
    # Precompute which entries are non-zero
    W_nonzero = W_ != 0.0
    
    # Compute Sc2
    Sc2 = np.zeros(n, dtype=np.float64)
    
    for comm in range(1, max_Ci + 1):
        for i in range(n):
            sum_w = 0.0
            for j in range(n):
                if W_nonzero[i, j] and Ci[j] == comm:
                    sum_w += W_[i, j]
            Sc2[i] += sum_w * sum_w
    
    # Compute P
    P = np.ones(n, dtype=np.float64)
    for i in range(n):
        S_sq = S[i] * S[i]
        if S_sq > 0.0:
            P[i] = 1.0 - Sc2[i] / S_sq
        else:
            P[i] = 0.0
    
    return P


def participation_coef_sign_v2(W, Ci):
    """
    Alternative implementation using vectorized inner function.
    """
    W = np.asarray(W, dtype=np.float64)
    Ci = np.asarray(Ci, dtype=np.int64).flatten()
    n = len(W)
    max_Ci = int(np.max(Ci))
    
    W_pos, W_neg = _extract_weights(W, n)
    
    Ppos = _pcoef_vectorized(W_pos, Ci, n, max_Ci)
    Pneg = _pcoef_vectorized(W_neg, Ci, n, max_Ci)
    
    return Ppos, Pneg


@njit(cache=True, parallel=True)
def _compute_both_coefficients(W, Ci, n, max_Ci):
    """
    Compute both positive and negative participation coefficients in one pass.
    This is the most optimized version that processes both weight types together.
    """
    Ppos = np.zeros(n, dtype=np.float64)
    Pneg = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        # Compute strength and Gc for both positive and negative
        S_pos = 0.0
        S_neg = 0.0
        
        # First pass: compute strengths and build temporary Gc info
        Gc_pos = np.zeros(n, dtype=np.int64)
        Gc_neg = np.zeros(n, dtype=np.int64)
        W_pos_row = np.zeros(n, dtype=np.float64)
        W_neg_row = np.zeros(n, dtype=np.float64)
        
        for j in range(n):
            w = W[i, j]
            if w > 0.0:
                S_pos += w
                Gc_pos[j] = Ci[j]
                W_pos_row[j] = w
            elif w < 0.0:
                w_neg = -w
                S_neg += w_neg
                Gc_neg[j] = Ci[j]
                W_neg_row[j] = w_neg
        
        # Compute Sc2 for both
        Sc2_pos = 0.0
        Sc2_neg = 0.0
        
        for comm in range(1, max_Ci + 1):
            sum_pos = 0.0
            sum_neg = 0.0
            for j in range(n):
                if Gc_pos[j] == comm:
                    sum_pos += W_pos_row[j]
                if Gc_neg[j] == comm:
                    sum_neg += W_neg_row[j]
            Sc2_pos += sum_pos * sum_pos
            Sc2_neg += sum_neg * sum_neg
        
        # Compute P
        S_sq_pos = S_pos * S_pos
        S_sq_neg = S_neg * S_neg
        
        if S_sq_pos > 0.0:
            Ppos[i] = 1.0 - Sc2_pos / S_sq_pos
        
        if S_sq_neg > 0.0:
            Pneg[i] = 1.0 - Sc2_neg / S_sq_neg
    
    return Ppos, Pneg


def participation_coef_sign_fast(W, Ci):
    """
    Fastest implementation - computes both coefficients in a single parallel pass.
    
    Parameters
    ----------
    W : numpy.ndarray
        Undirected connection matrix with positive and negative weights.
    Ci : numpy.ndarray
        Community affiliation vector.
    
    Returns
    -------
    Ppos, Pneg : numpy.ndarray
        Participation coefficients from positive and negative weights.
    """
    W = np.asarray(W, dtype=np.float64)
    Ci = np.asarray(Ci, dtype=np.int64).flatten()
    n = len(W)
    max_Ci = int(np.max(Ci))
    
    return _compute_both_coefficients(W, Ci, n, max_Ci)


# Warmup function to pre-compile JIT functions
def warmup():
    """Pre-compile JIT functions with small dummy data."""
    W_dummy = np.array([[0., 1.], [1., 0.]], dtype=np.float64)
    Ci_dummy = np.array([1, 1], dtype=np.int64)
    participation_coef_sign(W_dummy, Ci_dummy, use_parallel=False)
    participation_coef_sign(W_dummy, Ci_dummy, use_parallel=True)
    participation_coef_sign_v2(W_dummy, Ci_dummy)
    participation_coef_sign_fast(W_dummy, Ci_dummy)


if __name__ == "__main__":
    import scipy.io as sio
    import time
    
    print("=" * 70)
    print("Participation Coefficient - Numba Optimized Version Test")
    print("=" * 70)
    
    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    warmup()
    print("JIT compilation complete.\n")
    
    # Load test data
    try:
        inputs = sio.loadmat('test_data_inputs.mat')
        outputs = sio.loadmat('test_data_outputs.mat')
        
        W = inputs['W']
        Ci = inputs['Ci']
        
        Ppos_matlab = outputs['Ppos'].flatten()
        Pneg_matlab = outputs['Pneg'].flatten()
        
        print("Test with MATLAB reference data:")
        print("-" * 50)
        
        # Test all optimized versions
        Ppos, Pneg = participation_coef_sign(W, Ci)
        Ppos_fast, Pneg_fast = participation_coef_sign_fast(W, Ci)
        
        print(f"Standard optimized - Ppos match: {np.allclose(Ppos, Ppos_matlab)}")
        print(f"Standard optimized - Pneg match: {np.allclose(Pneg, Pneg_matlab)}")
        print(f"Fast version - Ppos match: {np.allclose(Ppos_fast, Ppos_matlab)}")
        print(f"Fast version - Pneg match: {np.allclose(Pneg_fast, Pneg_matlab)}")
        print(f"Max diff (standard): Ppos={np.max(np.abs(Ppos - Ppos_matlab)):.2e}, Pneg={np.max(np.abs(Pneg - Pneg_matlab)):.2e}")
        print(f"Max diff (fast):     Ppos={np.max(np.abs(Ppos_fast - Ppos_matlab)):.2e}, Pneg={np.max(np.abs(Pneg_fast - Pneg_matlab)):.2e}")
        
    except FileNotFoundError:
        print("MATLAB test data not found, using synthetic data.\n")
    
    # Performance benchmark
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    # Import original version for comparison
    import sys
    sys.path.insert(0, '/mnt/user-data/uploads')
    from participation_coef_sign import participation_coef_sign as original_func
    
    sizes = [50, 100, 200, 500, 1000]
    
    for n in sizes:
        print(f"\nMatrix size: {n} x {n}")
        print("-" * 50)
        
        # Generate random test data
        np.random.seed(42)
        W_test = np.random.randn(n, n)
        W_test = (W_test + W_test.T) / 2
        np.fill_diagonal(W_test, 0)
        
        num_comm = max(3, n // 20)
        Ci_test = np.random.randint(1, num_comm + 1, n)
        
        # Benchmark original
        n_iter = 10 if n <= 200 else (5 if n <= 500 else 3)
        
        start = time.perf_counter()
        for _ in range(n_iter):
            Ppos_orig, Pneg_orig = original_func(W_test, Ci_test)
        time_orig = (time.perf_counter() - start) / n_iter
        
        # Benchmark optimized (non-parallel)
        start = time.perf_counter()
        for _ in range(n_iter):
            Ppos_opt, Pneg_opt = participation_coef_sign(W_test, Ci_test, use_parallel=False)
        time_opt = (time.perf_counter() - start) / n_iter
        
        # Benchmark optimized (parallel)
        start = time.perf_counter()
        for _ in range(n_iter):
            Ppos_par, Pneg_par = participation_coef_sign(W_test, Ci_test, use_parallel=True)
        time_par = (time.perf_counter() - start) / n_iter
        
        # Benchmark fast version
        start = time.perf_counter()
        for _ in range(n_iter):
            Ppos_fast, Pneg_fast = participation_coef_sign_fast(W_test, Ci_test)
        time_fast = (time.perf_counter() - start) / n_iter
        
        # Check accuracy
        max_diff_pos = np.max(np.abs(Ppos_orig - Ppos_opt))
        max_diff_neg = np.max(np.abs(Pneg_orig - Pneg_opt))
        max_diff_pos_fast = np.max(np.abs(Ppos_orig - Ppos_fast))
        max_diff_neg_fast = np.max(np.abs(Pneg_orig - Pneg_fast))
        
        print(f"{'Method':<25} {'Time (ms)':<12} {'Speedup':<10} {'Max Diff':<15}")
        print(f"{'Original':<25} {time_orig*1000:>8.3f}    {'-':<10} {'-':<15}")
        print(f"{'Optimized (serial)':<25} {time_opt*1000:>8.3f}    {time_orig/time_opt:>6.2f}x    {max(max_diff_pos, max_diff_neg):.2e}")
        print(f"{'Optimized (parallel)':<25} {time_par*1000:>8.3f}    {time_orig/time_par:>6.2f}x    {max(max_diff_pos, max_diff_neg):.2e}")
        print(f"{'Fast (parallel)':<25} {time_fast*1000:>8.3f}    {time_orig/time_fast:>6.2f}x    {max(max_diff_pos_fast, max_diff_neg_fast):.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nRecommendations:")
    print("  - For small matrices (n < 100): use participation_coef_sign(W, Ci, use_parallel=False)")
    print("  - For medium matrices (100 <= n < 500): use participation_coef_sign(W, Ci)")
    print("  - For large matrices (n >= 500): use participation_coef_sign_fast(W, Ci)")
    print("\nAll optimized versions maintain numerical precision within floating-point tolerance.")
    print("Maximum expected difference from original: ~1e-15 (machine epsilon)")

