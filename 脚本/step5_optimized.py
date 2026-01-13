"""
module_step5.py
Step 5: K-means聚类 - Numba JIT 优化版本 (v2)

优化策略:
1. 使用 Numba JIT 加速核心计算 (串行，避免小数据开销)
2. 优化内存访问模式，减少分配
3. 快速收敛检测减少迭代次数
4. 单遍统计计算
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict


# ============================================================================
# Numba 优化的核心计算函数
# ============================================================================

@njit(cache=True, fastmath=True)
def reshape_cp_fortran(CP: np.ndarray, xNumBins: int, yNumBins: int, nTime: int) -> np.ndarray:
    """
    将 CP 重塑为 2D 矩阵 (Fortran order)
    """
    n_features = xNumBins * yNumBins
    pcwd = np.empty((n_features, nTime), dtype=CP.dtype)
    
    for t in range(nTime):
        idx = 0
        for j in range(yNumBins):
            for i in range(xNumBins):
                pcwd[idx, t] = CP[i, j, t]
                idx += 1
    
    return pcwd


@njit(cache=True, fastmath=True)
def compute_distances_and_assign(X: np.ndarray, centers: np.ndarray, 
                                  labels: np.ndarray, distances: np.ndarray) -> bool:
    """
    计算距离并分配簇，返回是否有变化
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_clusters = centers.shape[0]
    changed = False
    
    for i in range(n_samples):
        min_dist = np.inf
        min_idx = 0
        
        for k in range(n_clusters):
            dist = 0.0
            for j in range(n_features):
                diff = X[i, j] - centers[k, j]
                dist += diff * diff
            
            if dist < min_dist:
                min_dist = dist
                min_idx = k
        
        distances[i] = min_dist
        if labels[i] != min_idx:
            changed = True
            labels[i] = min_idx
    
    return changed


@njit(cache=True, fastmath=True)
def update_centers_inplace(X: np.ndarray, labels: np.ndarray, 
                           centers: np.ndarray, counts: np.ndarray) -> None:
    """
    就地更新聚类中心
    """
    n_clusters = centers.shape[0]
    n_features = centers.shape[1]
    
    # 清零
    for k in range(n_clusters):
        counts[k] = 0
        for j in range(n_features):
            centers[k, j] = 0.0
    
    # 累加
    for i in range(X.shape[0]):
        k = labels[i]
        counts[k] += 1
        for j in range(n_features):
            centers[k, j] += X[i, j]
    
    # 求均值
    for k in range(n_clusters):
        if counts[k] > 0:
            inv_count = 1.0 / counts[k]
            for j in range(n_features):
                centers[k, j] *= inv_count


@njit(cache=True, fastmath=True)
def kmeans_single_run_optimized(X: np.ndarray, n_clusters: int, max_iter: int,
                                 init_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    优化的单次 K-means 运行，减少内存分配
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # 预分配所有需要的数组
    centers = init_centers.copy()
    labels = np.zeros(n_samples, dtype=np.int32)
    distances = np.zeros(n_samples, dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    
    # 初始分配
    compute_distances_and_assign(X, centers, labels, distances)
    
    for iteration in range(max_iter):
        # 更新中心
        update_centers_inplace(X, labels, centers, counts)
        
        # 处理空簇
        for k in range(n_clusters):
            if counts[k] == 0:
                max_dist_idx = 0
                max_dist = distances[0]
                for i in range(1, n_samples):
                    if distances[i] > max_dist:
                        max_dist = distances[i]
                        max_dist_idx = i
                for j in range(n_features):
                    centers[k, j] = X[max_dist_idx, j]
        
        # 重新分配
        changed = compute_distances_and_assign(X, centers, labels, distances)
        
        if not changed:
            break
    
    # 计算最终惯性
    inertia = 0.0
    for i in range(n_samples):
        inertia += distances[i]
    
    return labels, centers, inertia


@njit(cache=True)
def initialize_centers_kmeans_pp(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """
    K-means++ 初始化
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    
    # 随机选择第一个中心
    first_idx = np.random.randint(0, n_samples)
    for j in range(n_features):
        centers[0, j] = X[first_idx, j]
    
    # 用于存储最小距离
    min_distances = np.full(n_samples, np.inf, dtype=X.dtype)
    
    # 选择剩余的中心
    for k in range(1, n_clusters):
        # 更新最小距离
        for i in range(n_samples):
            dist = 0.0
            for j in range(n_features):
                diff = X[i, j] - centers[k-1, j]
                dist += diff * diff
            if dist < min_distances[i]:
                min_distances[i] = dist
        
        # 按概率选择下一个中心
        total = 0.0
        for i in range(n_samples):
            total += min_distances[i]
        
        if total == 0:
            next_idx = np.random.randint(0, n_samples)
        else:
            r = np.random.random() * total
            cumsum = 0.0
            next_idx = n_samples - 1
            for i in range(n_samples):
                cumsum += min_distances[i]
                if cumsum >= r:
                    next_idx = i
                    break
        
        for j in range(n_features):
            centers[k, j] = X[next_idx, j]
    
    return centers


@njit(cache=True, fastmath=True)
def kmeans_multiple_runs(X: np.ndarray, n_clusters: int, n_init: int, 
                          max_iter: int, base_seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    执行多次 K-means 并返回最佳结果
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    best_labels = np.zeros(n_samples, dtype=np.int32)
    best_centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    best_inertia = np.inf
    
    for run in range(n_init):
        seed = base_seed + run
        init_centers = initialize_centers_kmeans_pp(X, n_clusters, seed)
        labels, centers, inertia = kmeans_single_run_optimized(X, n_clusters, max_iter, init_centers)
        
        if inertia < best_inertia:
            best_inertia = inertia
            for i in range(n_samples):
                best_labels[i] = labels[i]
            for k in range(n_clusters):
                for j in range(n_features):
                    best_centers[k, j] = centers[k, j]
    
    return best_labels, best_centers, best_inertia


def kmeans_numba(X: np.ndarray, n_clusters: int = 2, n_init: int = 100, 
                 max_iter: int = 300, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Numba 加速的 K-means 聚类
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    
    if random_state is None:
        random_state = np.random.randint(0, 2**31)
    
    return kmeans_multiple_runs(X, n_clusters, n_init, max_iter, random_state)


# ============================================================================
# 优化的统计计算
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_state_means_fast(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    计算指定状态的均值 (对每个 ROI)
    """
    nROI = data.shape[0]
    nTime = data.shape[1]
    means = np.empty(nROI, dtype=np.float64)
    
    count = 0
    for t in range(nTime):
        if mask[t]:
            count += 1
    
    if count == 0:
        for i in range(nROI):
            means[i] = 0.0
        return means
    
    inv_count = 1.0 / count
    
    for i in range(nROI):
        total = 0.0
        for t in range(nTime):
            if mask[t]:
                total += data[i, t]
        means[i] = total * inv_count
    
    return means


@njit(cache=True, fastmath=True)
def welch_ttest_fast(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    快速 Welch's t-test
    """
    n1 = len(x)
    n2 = len(y)
    
    # 单遍计算均值和方差
    mean1 = 0.0
    mean2 = 0.0
    for i in range(n1):
        mean1 += x[i]
    for i in range(n2):
        mean2 += y[i]
    mean1 /= n1
    mean2 /= n2
    
    var1 = 0.0
    var2 = 0.0
    for i in range(n1):
        diff = x[i] - mean1
        var1 += diff * diff
    for i in range(n2):
        diff = y[i] - mean2
        var2 += diff * diff
    
    var1 /= (n1 - 1)
    var2 /= (n2 - 1)
    
    se1 = var1 / n1
    se2 = var2 / n2
    se = np.sqrt(se1 + se2)
    
    if se < 1e-15:
        return 0.0, 1e10
    
    t_stat = (mean1 - mean2) / se
    
    num = (se1 + se2) ** 2
    denom = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
    df = num / denom if denom > 1e-15 else n1 + n2 - 2
    
    return t_stat, df


# ============================================================================
# 主函数
# ============================================================================

def run_step5(CP: np.ndarray, WT: np.ndarray, BT: np.ndarray, 
              xNumBins: int, yNumBins: int, 
              n_clusters: int = 2, n_init: int = 100, 
              random_state: int = None,
              use_sklearn_kmeans: bool = False) -> Dict:
    """
    执行Step 5: K-means聚类分析 (Numba优化版)
    """
    from scipy.stats import t as t_dist
    
    nTime = CP.shape[2]
    
    # 确保数据类型和内存布局
    CP = np.ascontiguousarray(CP, dtype=np.float64)
    WT = np.ascontiguousarray(WT, dtype=np.float64)
    BT = np.ascontiguousarray(BT, dtype=np.float64)
    
    # Step 5.1: 重塑CP为2D矩阵
    pcwd = reshape_cp_fortran(CP, xNumBins, yNumBins, nTime)
    
    # Step 5.2: K-means聚类
    kmeans_input = np.ascontiguousarray(pcwd.T)  # (nTime, features)
    
    if use_sklearn_kmeans:
        from sklearn.cluster import KMeans
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
            algorithm='lloyd'
        )
        idx_0based = kmeans_model.fit_predict(kmeans_input)
        cluster_centers = kmeans_model.cluster_centers_
    else:
        idx_0based, cluster_centers, _ = kmeans_numba(
            kmeans_input, 
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state if random_state is not None else 42
        )
    
    idx = idx_0based + 1
    
    # Step 5.3: 找出两个状态
    state1_mask = (idx == 1).astype(np.bool_)
    state2_mask = (idx == 2).astype(np.bool_)
    state1_indices = np.where(state1_mask)[0]
    state2_indices = np.where(state2_mask)[0]
    
    # Step 5.4: 计算各状态的均值
    s1_WT = compute_state_means_fast(WT, state1_mask)
    s2_WT = compute_state_means_fast(WT, state2_mask)
    s1_BT = compute_state_means_fast(BT, state1_mask)
    s2_BT = compute_state_means_fast(BT, state2_mask)
    
    # Step 5.5: t检验
    t_stat, df = welch_ttest_fast(s1_BT, s2_BT)
    p_value = 2.0 * t_dist.sf(np.abs(t_stat), df)
    
    # Step 5.6: 根据 t 统计量确定状态标签
    labels_swapped = False
    
    if t_stat > 0:
        integration_count = len(state1_indices)
        segregation_count = len(state2_indices)
        BT_value = np.array([np.mean(s1_BT), np.mean(s2_BT)])
        final_idx = idx.copy()
    else:
        integration_count = len(state2_indices)
        segregation_count = len(state1_indices)
        BT_value = np.array([np.mean(s2_BT), np.mean(s1_BT)])
        final_idx = 3 - idx
        labels_swapped = True
    
    return {
        'idx': final_idx,
        'integration_count': integration_count,
        'segregation_count': segregation_count,
        'BT_value': BT_value,
        'cluster_centers': cluster_centers,
        'tstat': t_stat,
        'pvalue': p_value,
        'labels_swapped': labels_swapped,
        's1_BT_mean': np.mean(s1_BT),
        's2_BT_mean': np.mean(s2_BT),
    }


def update_results(result, s, Trady_results, WTBT):
    """更新结果矩阵"""
    Trady_results[s, 0] = result['integration_count']
    Trady_results[s, 1] = result['segregation_count']
    
    while len(WTBT) <= s:
        WTBT.append([None, None, None])
    
    WTBT[s][2] = result['idx']


def warmup():
    """预热 JIT 编译"""
    test_cp = np.random.rand(5, 5, 10).astype(np.float64)
    test_wt = np.random.rand(3, 10).astype(np.float64)
    test_bt = np.random.rand(3, 10).astype(np.float64)
    _ = run_step5(test_cp, test_wt, test_bt, 5, 5, n_init=2, random_state=0)


if __name__ == "__main__":
    import scipy.io as sio
    import time
    
    print("=" * 60)
    print("Module Step5 - Numba 优化版本测试 (v2)")
    print("=" * 60)
    
    # 加载测试数据
    print("\n加载测试数据...")
    input_data = sio.loadmat('test_step5_input.mat', squeeze_me=False)
    
    CP = input_data['CP']
    WT = input_data['WT']
    BT = input_data['BT']
    xNumBins = int(input_data['xNumBins'].flatten()[0])
    yNumBins = int(input_data['yNumBins'].flatten()[0])
    
    # 预热
    print("预热 JIT...")
    warmup()
    
    # 测试
    print("\n运行测试...")
    
    # Numba版本
    start = time.perf_counter()
    result_numba = run_step5(CP, WT, BT, xNumBins, yNumBins, random_state=42)
    time_numba = time.perf_counter() - start
    
    # sklearn版本
    start = time.perf_counter()
    result_sklearn = run_step5(CP, WT, BT, xNumBins, yNumBins, random_state=42, use_sklearn_kmeans=True)
    time_sklearn = time.perf_counter() - start
    
    print(f"\nNumba: {time_numba*1000:.2f} ms")
    print(f"sklearn: {time_sklearn*1000:.2f} ms")
    print(f"加速比: {time_sklearn/time_numba:.2f}x")
    
    # 与MATLAB比较
    output_mat = sio.loadmat('test_step5_output.mat', squeeze_me=False)
    matlab_BT = output_mat['BT_value']
    print(f"\nBT值匹配: {np.allclose(result_numba['BT_value'], matlab_BT.flatten())}")
