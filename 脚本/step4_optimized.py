"""
step4_optimized.py
Step 4: 2-dimensional Cartographic Profile (CP) - Numba JIT优化版本

使用Numba JIT编译加速计算，保持与原始MATLAB实现完全一致的数值精度。

输入:
    BT: (nNodes, nTime) - 参与系数，范围 [0, 1]
    WT: (nNodes, nTime) - 模块度z分数，范围约 [-5, 5]

输出:
    CP: (101, 101, nTime) - 2D直方图，每个时间点的节点分布
    xbins: (101,) - 参与系数的bin边界 [0, 0.01, ..., 1.0]
    ybins: (101,) - 模块度z分数的bin边界 [5, 4.9, ..., -5]
"""

import numpy as np
from numba import njit, prange


# 预计算的bins常量（避免重复创建）
_XBINS = np.arange(0, 1.01, 0.01)    # 101个值
_YBINS = np.arange(5, -5.1, -0.1)    # 101个值（递减）
_X_NUM_BINS = 101
_Y_NUM_BINS = 101


@njit(cache=True)
def _compute_indices_single_time(BT_t, WT_t, nNodes):
    """
    计算单个时间点的bin索引 (0-based)
    
    参数:
        BT_t: 1D数组，第t个时间点的参与系数
        WT_t: 1D数组，第t个时间点的模块度z分数
        nNodes: 节点数量
    
    返回:
        Xi_idx: 0-based x索引数组
        Yi_idx: 0-based y索引数组
    """
    Xi_idx = np.empty(nNodes, dtype=np.int64)
    Yi_idx = np.empty(nNodes, dtype=np.int64)
    
    for i in range(nNodes):
        # 计算索引 (1-based)
        # Xi = round(1 + BT * 100)
        # Yi = round(1 + (5 - WT) * 10)
        xi = round(1.0 + BT_t[i] * 100.0)
        yi = round(1.0 + (5.0 - WT_t[i]) * 10.0)
        
        # 裁剪到有效范围 [1, 101]
        if xi < 1:
            xi = 1
        elif xi > 101:
            xi = 101
        
        if yi < 1:
            yi = 1
        elif yi > 101:
            yi = 101
        
        # 转换为0-based索引
        Xi_idx[i] = int(xi) - 1
        Yi_idx[i] = int(yi) - 1
    
    return Xi_idx, Yi_idx


@njit(cache=True)
def _accumulate_histogram(CP_t, Yi_idx, Xi_idx, nNodes):
    """
    累积直方图（单个时间点）
    
    参数:
        CP_t: 2D数组 (101, 101)，将被原地修改
        Yi_idx: 0-based y索引数组
        Xi_idx: 0-based x索引数组
        nNodes: 节点数量
    """
    for i in range(nNodes):
        CP_t[Yi_idx[i], Xi_idx[i]] += 1.0


@njit(cache=True)
def _compute_cp_sequential(BT, WT, nNodes, nTime):
    """
    顺序计算CP（单线程版本）
    """
    CP = np.zeros((101, 101, nTime))
    
    for t in range(nTime):
        Xi_idx, Yi_idx = _compute_indices_single_time(BT[:, t], WT[:, t], nNodes)
        _accumulate_histogram(CP[:, :, t], Yi_idx, Xi_idx, nNodes)
    
    return CP


@njit(parallel=True, cache=True)
def _compute_cp_parallel(BT, WT, nNodes, nTime):
    """
    并行计算CP（多线程版本）
    每个时间点独立计算，可以并行
    """
    CP = np.zeros((101, 101, nTime))
    
    for t in prange(nTime):
        # 计算索引
        Xi_idx = np.empty(nNodes, dtype=np.int64)
        Yi_idx = np.empty(nNodes, dtype=np.int64)
        
        for i in range(nNodes):
            xi = round(1.0 + BT[i, t] * 100.0)
            yi = round(1.0 + (5.0 - WT[i, t]) * 10.0)
            
            # 裁剪
            if xi < 1:
                xi = 1
            elif xi > 101:
                xi = 101
            
            if yi < 1:
                yi = 1
            elif yi > 101:
                yi = 101
            
            Xi_idx[i] = int(xi) - 1
            Yi_idx[i] = int(yi) - 1
        
        # 累积直方图
        for i in range(nNodes):
            CP[Yi_idx[i], Xi_idx[i], t] += 1.0
    
    return CP


def compute_cartographic_profile(BT, WT, parallel=True):
    """
    计算2D Cartographic Profile (CP) - Numba JIT优化版本
    
    参数:
        BT: ndarray, shape (nNodes, nTime)
            参与系数矩阵，值范围 [0, 1]
        WT: ndarray, shape (nNodes, nTime)
            模块度z分数矩阵，值范围约 [-5, 5]
        parallel: bool, 默认True
            是否使用并行计算（对于大数据集效果更好）
    
    返回:
        CP: ndarray, shape (101, 101, nTime)
            2D直方图，CP[y, x, t] 表示第t个时间点落入bin (x,y) 的节点数
        xbins: ndarray, shape (101,)
            参与系数的bin边界
        ybins: ndarray, shape (101,)
            模块度z分数的bin边界（递减顺序）
    """
    BT = np.asarray(BT, dtype=np.float64)
    WT = np.asarray(WT, dtype=np.float64)
    
    # 确保是2D数组
    if BT.ndim == 1:
        BT = BT[:, np.newaxis]
        WT = WT[:, np.newaxis]
    
    # 确保C连续内存布局以获得最佳性能
    if not BT.flags['C_CONTIGUOUS']:
        BT = np.ascontiguousarray(BT)
    if not WT.flags['C_CONTIGUOUS']:
        WT = np.ascontiguousarray(WT)
    
    nNodes, nTime = BT.shape
    
    # 选择计算方式
    if parallel and nTime > 1:
        CP = _compute_cp_parallel(BT, WT, nNodes, nTime)
    else:
        CP = _compute_cp_sequential(BT, WT, nNodes, nTime)
    
    return CP, _XBINS.copy(), _YBINS.copy()


def warmup():
    """
    预热JIT编译器
    调用一次以确保后续调用不会包含编译时间
    """
    BT_dummy = np.random.rand(10, 2)
    WT_dummy = np.random.randn(10, 2)
    compute_cartographic_profile(BT_dummy, WT_dummy, parallel=False)
    compute_cartographic_profile(BT_dummy, WT_dummy, parallel=True)


# 可选：模块导入时自动预热
# warmup()
