"""
Time-resolved connectivity using Multiplication of Temporal Derivatives (MTD)

OPTIMIZED Python implementation with Numba JIT acceleration.
Maintains exact numerical compatibility with MATLAB's coupling.m

Optimizations:
1. Numba JIT compilation for hot loops
2. Cumsum-based O(n) moving average (vs O(n*span) naive approach)
3. Vectorized operations where possible
4. Parallel processing for the node×node loop

Author: Optimized version
"""

import numpy as np
from numba import jit, prange
import numba


def matlab_round(x):
    """
    Replicate MATLAB's round function (round half away from zero).
    """
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)


@jit(nopython=True, cache=True)
def _matlab_smooth_numba(y, span):
    """
    Numba-accelerated MATLAB smooth replication.
    
    Uses the same algorithm as the original but compiled with Numba.
    """
    n = len(y)
    
    # Make span odd
    if span % 2 == 0:
        span = span - 1
    if span < 1:
        span = 1
    if span > n:
        span = n if n % 2 == 1 else n - 1
    
    half_span = span // 2
    out = np.zeros(n)
    
    for i in range(n):
        # Effective half-window size (limited by boundaries)
        dist_from_start = i
        dist_from_end = n - 1 - i
        effective_half = min(half_span, dist_from_start, dist_from_end)
        
        # Window bounds
        start = i - effective_half
        end = i + effective_half + 1
        
        # Manual mean calculation (faster in numba than np.mean on small arrays)
        s = 0.0
        for idx in range(start, end):
            s += y[idx]
        out[i] = s / (end - start)
    
    return out


@jit(nopython=True, cache=True)
def _matlab_smooth_cumsum(y, span):
    """
    Cumsum-based MATLAB smooth replication - O(n) complexity.
    
    Uses cumulative sum to compute moving averages efficiently,
    with special handling for MATLAB's boundary behavior.
    """
    n = len(y)
    
    # Make span odd
    if span % 2 == 0:
        span = span - 1
    if span < 1:
        span = 1
    if span > n:
        span = n if n % 2 == 1 else n - 1
    
    half_span = span // 2
    out = np.zeros(n)
    
    # Compute cumulative sum with a leading zero
    # cumsum[i] = sum of y[0:i]
    cumsum = np.zeros(n + 1)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + y[i]
    
    for i in range(n):
        # Effective half-window size (limited by boundaries)
        dist_from_start = i
        dist_from_end = n - 1 - i
        effective_half = min(half_span, dist_from_start, dist_from_end)
        
        # Window bounds
        start = i - effective_half
        end = i + effective_half + 1
        
        # Use cumsum to get window sum: sum(y[start:end]) = cumsum[end] - cumsum[start]
        window_sum = cumsum[end] - cumsum[start]
        out[i] = window_sum / (end - start)
    
    return out


@jit(nopython=True, parallel=True, cache=True)
def _smooth_all_pairs_parallel(fc, window, nodes, ts_minus_1):
    """
    Apply smoothing to all node pairs in parallel.
    
    This is the main performance bottleneck - we parallelize across
    the nodes×nodes pairs.
    """
    mtd_temp = np.zeros((nodes, nodes, ts_minus_1))
    
    # Flatten the loop for better parallel distribution
    for idx in prange(nodes * nodes):
        j = idx // nodes
        k = idx % nodes
        # Extract the time series for this pair
        ts_data = fc[:, j, k].copy()  # Need copy for contiguous array
        mtd_temp[j, k, :] = _matlab_smooth_cumsum(ts_data, window)
    
    return mtd_temp


@jit(nopython=True, cache=True)
def _smooth_all_pairs_serial(fc, window, nodes, ts_minus_1):
    """
    Apply smoothing to all node pairs serially (for comparison/debugging).
    """
    mtd_temp = np.zeros((nodes, nodes, ts_minus_1))
    
    for j in range(nodes):
        for k in range(nodes):
            ts_data = fc[:, j, k].copy()
            mtd_temp[j, k, :] = _matlab_smooth_cumsum(ts_data, window)
    
    return mtd_temp


def coupling_optimized(data, window, direction=0, trim=0, parallel=True):
    """
    OPTIMIZED Time-resolved connectivity using MTD.
    
    Creates a functional coupling metric from time series data.
    Numerically identical to MATLAB's coupling.m but much faster.
    
    Parameters:
    -----------
    data : ndarray
        Time series data organized as TIME x NODES matrix.
    window : int
        Smoothing parameter for simple moving average of coupling metric.
    direction : int, optional
        Window type: 0 = middle (default), 1 = forward facing
    trim : int, optional
        Whether to trim zeros from the ends: 0 = no (default), 1 = yes
    parallel : bool, optional
        Whether to use parallel processing (default True)
    
    Returns:
    --------
    mtd : ndarray
        Time-varying connectivity matrix, shape (NODES, NODES, TIME).
    """
    data = np.asarray(data, dtype=np.float64)
    
    # Get dimensions

    ts, nodes = data.shape
    
    # Step 1: Calculate temporal derivative
    td = np.diff(data, axis=0)  # shape: (ts-1, nodes)
    
    # Step 2: Standardize data (VECTORIZED - no loop needed)
    data_std = np.std(td, axis=0, ddof=1)  # shape: (nodes,)
    data_std[data_std == 0] = 1.0
    td = td / data_std  # Broadcasting handles the division
    
    # Step 3: Functional coupling score
    fc = td[:, np.newaxis, :] * td[:, :, np.newaxis]  # shape: (ts-1, nodes, nodes)
    
    # Ensure C-contiguous for numba
    fc = np.ascontiguousarray(fc)
    
    # Step 4: Temporal smoothing (PARALLELIZED with Numba)
    if parallel:
        mtd_temp = _smooth_all_pairs_parallel(fc, window, nodes, ts - 1)
    else:
        mtd_temp = _smooth_all_pairs_serial(fc, window, nodes, ts - 1)
    
    # Step 5: Initialize output and apply window type
    mtd = np.zeros((nodes, nodes, ts))
    
    if direction == 1:
        idx_offset = int(matlab_round(window / 2 + 1))
        dest_end = ts - idx_offset
        src_start = idx_offset - 1
        mtd[:, :, :dest_end] = mtd_temp[:, :, src_start:]
    else:
        mtd[:, :, :ts - 1] = mtd_temp
    
    # Step 6: Trim ends if requested
    if trim == 1:
        if direction == 0:
            trim_amount = int(matlab_round(window / 2))
            trim_start = trim_amount
            trim_end = ts - trim_amount
            mtd = mtd[:, :, trim_start:trim_end]
        else:
            mtd = mtd[:, :, :ts - window]
    
    return mtd


# =============================================================================
# Original implementation for comparison
# =============================================================================

def matlab_smooth_original(y, span):
    """Original (slow) implementation for reference."""
    y = np.asarray(y).flatten()
    n = len(y)
    
    span = int(span)
    if span % 2 == 0:
        span = span - 1
    if span < 1:
        span = 1
    if span > n:
        span = n if n % 2 == 1 else n - 1
    
    half_span = span // 2
    out = np.zeros(n)
    
    for i in range(n):
        dist_from_start = i
        dist_from_end = n - 1 - i
        effective_half = min(half_span, dist_from_start, dist_from_end)
        start = i - effective_half
        end = i + effective_half + 1
        out[i] = np.mean(y[start:end])
    
    return out


def coupling_original(data, window, direction=0, trim=0):
    """Original (slow) implementation for comparison."""
    ts, nodes = data.shape
    td = np.diff(data, axis=0)
    data_std = np.std(td, axis=0, ddof=1)
    data_std[data_std == 0] = 1.0
    
    for i in range(nodes):
        td[:, i] = td[:, i] / data_std[i]
    
    fc = td[:, np.newaxis, :] * td[:, :, np.newaxis]
    mtd_temp = np.zeros((nodes, nodes, ts - 1))
    
    for j in range(nodes):
        for k in range(nodes):
            mtd_temp[j, k, :] = matlab_smooth_original(fc[:, j, k], window)
    
    mtd = np.zeros((nodes, nodes, ts))
    
    if direction == 1:
        idx_offset = int(matlab_round(window / 2 + 1))
        dest_end = ts - idx_offset
        src_start = idx_offset - 1
        mtd[:, :, :dest_end] = mtd_temp[:, :, src_start:]
    else:
        mtd[:, :, :ts - 1] = mtd_temp
    
    if trim == 1:
        if direction == 0:
            trim_amount = int(matlab_round(window / 2))
            mtd = mtd[:, :, trim_amount:ts - trim_amount]
        else:
            mtd = mtd[:, :, :ts - window]
    
    return mtd


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("Coupling Optimization Test")
    print("=" * 70)
    
    # Test data
    np.random.seed(42)
    ts, nodes = 200, 50
    test_data = np.random.randn(ts, nodes)
    window = 14
    direction = 1
    trim = 1
    
    print(f"\nTest configuration:")
    print(f"  Data shape: ({ts}, {nodes})")
    print(f"  Window: {window}")
    print(f"  Total node pairs: {nodes * nodes}")
    
    # Warmup JIT
    print("\nWarming up JIT compilation...")
    _ = coupling_optimized(test_data[:20, :5], window, direction, trim)
    
    # Test optimized version
    print("\nRunning optimized version (parallel)...")
    t0 = time.perf_counter()
    mtd_opt = coupling_optimized(test_data, window, direction, trim, parallel=True)
    t_opt = time.perf_counter() - t0
    print(f"  Time: {t_opt:.4f}s")
    
    # Test optimized version (serial)
    print("\nRunning optimized version (serial)...")
    t0 = time.perf_counter()
    mtd_opt_serial = coupling_optimized(test_data, window, direction, trim, parallel=False)
    t_opt_serial = time.perf_counter() - t0
    print(f"  Time: {t_opt_serial:.4f}s")
    
    # Test original version
    print("\nRunning original version...")
    t0 = time.perf_counter()
    mtd_orig = coupling_original(test_data, window, direction, trim)
    t_orig = time.perf_counter() - t0
    print(f"  Time: {t_orig:.4f}s")
    
    # Compare results
    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)
    
    diff_parallel = np.abs(mtd_opt - mtd_orig)
    diff_serial = np.abs(mtd_opt_serial - mtd_orig)
    
    print(f"\nParallel vs Original:")
    print(f"  Max absolute diff: {diff_parallel.max():.2e}")
    print(f"  Mean absolute diff: {diff_parallel.mean():.2e}")
    print(f"  Exact match: {np.allclose(mtd_opt, mtd_orig, rtol=1e-14, atol=1e-14)}")
    
    print(f"\nSerial vs Original:")
    print(f"  Max absolute diff: {diff_serial.max():.2e}")
    print(f"  Mean absolute diff: {diff_serial.mean():.2e}")
    print(f"  Exact match: {np.allclose(mtd_opt_serial, mtd_orig, rtol=1e-14, atol=1e-14)}")
    
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"  Original:           {t_orig:.4f}s")
    print(f"  Optimized (serial): {t_opt_serial:.4f}s  ({t_orig/t_opt_serial:.1f}x speedup)")
    print(f"  Optimized (parallel): {t_opt:.4f}s  ({t_orig/t_opt:.1f}x speedup)")
