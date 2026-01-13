"""tSNR计算模块 - 精简版"""

import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _compute_tsnr_numba(data):
        n_timepoints, n_features = data.shape
        tsnr = np.zeros(n_features, dtype=np.float64)
        for i in prange(n_features):
            mean_val = 0.0
            for t in range(n_timepoints):
                mean_val += data[t, i]
            mean_val /= n_timepoints
            var_val = 0.0
            for t in range(n_timepoints):
                diff = data[t, i] - mean_val
                var_val += diff * diff
            std_val = np.sqrt(var_val / (n_timepoints - 1))
            if std_val > 1e-10:
                tsnr[i] = mean_val / std_val
        return tsnr


def compute_tsnr(data):
    """计算tSNR: mean / std"""
    data = np.asarray(data, dtype=np.float64)
    if HAS_NUMBA:
        return _compute_tsnr_numba(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    tsnr = np.zeros_like(mean)
    valid = std > 1e-10
    tsnr[valid] = mean[valid] / std[valid]
    return tsnr


def filter_by_tsnr(data, threshold_sd=2.0):
    """过滤tSNR偏离均值超过threshold_sd个标准差的特征"""
    tsnr = compute_tsnr(data)
    valid_tsnr = tsnr[np.isfinite(tsnr)]
    mean, std = np.mean(valid_tsnr), np.std(valid_tsnr, ddof=1)
    mask = (tsnr >= mean - threshold_sd * std) & (tsnr <= mean + threshold_sd * std) & np.isfinite(tsnr)
    return data[:, mask], mask, {'tsnr': tsnr, 'mean': mean, 'std': std, 'n_excluded': np.sum(~mask)}


# ============ 添加以下两个函数 ============

def get_tsnr_mask(tsnr_values, threshold_sd=2.0):
    """根据已计算的tSNR值获取有效特征掩码"""
    tsnr = np.asarray(tsnr_values)
    valid_tsnr = tsnr[np.isfinite(tsnr)]
    
    if len(valid_tsnr) == 0:
        return np.zeros(len(tsnr), dtype=bool), {
            'mean_tsnr': np.nan, 'std_tsnr': np.nan,
            'threshold_low': np.nan, 'threshold_high': np.nan,
            'n_excluded': len(tsnr), 'n_valid': 0
        }
    
    mean_tsnr = np.mean(valid_tsnr)
    std_tsnr = np.std(valid_tsnr, ddof=1)
    threshold_low = mean_tsnr - threshold_sd * std_tsnr
    threshold_high = mean_tsnr + threshold_sd * std_tsnr
    
    valid_mask = (tsnr >= threshold_low) & (tsnr <= threshold_high) & np.isfinite(tsnr)
    
    return valid_mask, {
        'mean_tsnr': mean_tsnr,
        'std_tsnr': std_tsnr,
        'threshold_low': threshold_low,
        'threshold_high': threshold_high,
        'n_excluded': np.sum(~valid_mask),
        'n_valid': np.sum(valid_mask)
    }


def warmup_jit():
    """预热JIT编译器"""
    if HAS_NUMBA:
        test_data = np.random.randn(50, 20).astype(np.float64)
        _ = compute_tsnr(test_data)