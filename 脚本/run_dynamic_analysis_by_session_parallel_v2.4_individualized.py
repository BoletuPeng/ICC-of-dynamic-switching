#!/usr/bin/env python3
"""
run_dynamic_analysis_by_session_parallel_v2.4_individualized.py
DMN-ECN动态功能分析主脚本 - 个体化分割版本

本脚本与 run_dynamic_analysis_by_session_parallel_v2.4.py 完全一致的分析管线，
唯一的区别是数据读取部分适配了 extract_roi_timeseries_v2_individualized.py 的输出格式。

数据格式差异：
- 原始脚本：按session组织 (rsfc_ses-01.mat, rsfc_ses-02.mat, ...)
  每个文件包含该session所有被试的数据
- 本脚本：按subject组织 (sub-001.mat, sub-002.mat, ...)
  每个文件包含该被试所有session的连接数据

空间差异：
- 原始脚本：fsaverage6, MNI
- 本脚本：Schaefer, Kong

使用方法：
    python run_dynamic_analysis_by_session_parallel_v2.4_individualized.py \\
        --roi_dir "C:/path/to/analysis_output_individualized" \\
        --behavior_file "C:/path/to/behavior_data.csv" \\
        --methods_dir "C:/path/to/Methods" \\
        --output_dir "C:/path/to/dynamic_analysis_individualized" \\
        --level ROI \\
        --Louv 500 --LouvTau 0.1 --LouvReps 500 \\
        --Kmns 100 \\
        --tSNR --tSNR_threshold 2.0 \\
        --Motion --motion_file "C:/path/to/excluded_sessions.txt" \\
        --n_jobs 23

Author: Adapted from Chen et al. (2025) MATLAB code
Version: 2.4 individualized (with corrected Louvain consensus implementation)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import savemat, loadmat
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 并行处理相关
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def add_methods_to_path(methods_dir):
    """将优化方法模块添加到系统路径"""
    if methods_dir not in sys.path:
        sys.path.insert(0, methods_dir)


def load_motion_exclusion_list(motion_file):
    """
    加载motion排除列表
    
    Parameters
    ----------
    motion_file : str
        排除列表文件路径，格式为每行一个 (subject_num, session_num)
    
    Returns
    -------
    excluded_sessions : set
        排除的 (sub_id, session_id) 元组集合
    """
    excluded = set()
    
    if not os.path.exists(motion_file):
        print(f"  Warning: Motion exclusion file not found: {motion_file}")
        return excluded
    
    with open(motion_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析 (subject, session) 格式
            try:
                # 处理 (3,5) 格式
                line = line.replace('(', '').replace(')', '').replace(' ', '')
                parts = line.split(',')
                if len(parts) == 2:
                    sub_num = int(parts[0])
                    ses_num = int(parts[1])
                    # 转换为标准格式: sub-003, ses-05
                    sub_id = f"sub-{sub_num:03d}"
                    ses_id = f"ses-{ses_num:02d}"
                    excluded.add((sub_id, ses_id))
            except (ValueError, IndexError):
                continue
    
    return excluded


def load_roi_data_by_session_individualized(roi_dir, space='Schaefer', level='ROI'):
    """
    按session加载ROI时间序列数据 - 个体化版本
    
    从按subject组织的文件中读取数据，重新组织为按session格式
    
    输入文件格式（extract_roi_timeseries_v2_individualized.py 输出）：
    - roi_dir/Schaefer/ROI/sub-001.mat (或.npz)
    - 每个文件包含: timeseries, sessions, n_timepoints_per_session
    
    Returns
    -------
    session_data : dict
        {session_id: {'rsfc': list of arrays, 'sub_ids': list}}
    all_sessions : list
        所有session ID列表
    """
    import glob
    
    # 构建数据目录路径
    data_dir = os.path.join(roi_dir, space, level)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # 查找所有subject文件
    subject_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*.mat')))
    
    if not subject_files:
        subject_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*.npz')))
    
    if not subject_files:
        raise FileNotFoundError(f"No subject files found in {data_dir}")
    
    # 首先收集所有数据，按subject组织
    subject_data = {}  # {sub_id: {ses_id: timeseries}}
    all_sessions_set = set()
    
    for f in subject_files:
        basename = os.path.basename(f)
        # 提取被试ID (sub-001.mat -> sub-001)
        sub_id = basename.replace('.mat', '').replace('.npz', '')
        
        try:
            if f.endswith('.mat'):
                mat = loadmat(f, squeeze_me=True)
                timeseries = mat['timeseries']
                sessions_raw = mat['sessions']
                n_tp_per_session = mat['n_timepoints_per_session']
            else:
                data = np.load(f, allow_pickle=True)
                timeseries = data['timeseries']
                sessions_raw = data['sessions']
                n_tp_per_session = data['n_timepoints_per_session']
            
            # 处理sessions格式
            if isinstance(sessions_raw, np.ndarray):
                if sessions_raw.dtype.kind in ['U', 'S', 'O']:
                    sessions = [str(s) for s in sessions_raw.flatten()]
                else:
                    sessions = [str(s) for s in sessions_raw]
            else:
                sessions = [str(sessions_raw)]
            
            # 处理n_tp_per_session格式
            if isinstance(n_tp_per_session, np.ndarray):
                n_tp_per_session = n_tp_per_session.flatten().astype(int)
            else:
                n_tp_per_session = np.array([int(n_tp_per_session)])
            
            # 分割时间序列
            subject_data[sub_id] = {}
            start_idx = 0
            
            for ses, n_tp in zip(sessions, n_tp_per_session):
                end_idx = start_idx + n_tp
                subject_data[sub_id][ses] = timeseries[start_idx:end_idx, :]
                all_sessions_set.add(ses)
                start_idx = end_idx
                
        except Exception as e:
            print(f"  Warning: Failed to load {f}: {e}")
            continue
    
    # 重新组织为按session格式
    all_sessions = sorted(all_sessions_set)
    session_data = {}
    
    for ses in all_sessions:
        rsfc_list = []
        sub_ids = []
        
        for sub_id in sorted(subject_data.keys()):
            if ses in subject_data[sub_id]:
                rsfc_list.append(subject_data[sub_id][ses])
                sub_ids.append(sub_id)
        
        if rsfc_list:
            session_data[ses] = {
                'rsfc': rsfc_list,
                'sub_ids': sub_ids
            }
    
    return session_data, all_sessions


def load_roi_info_individualized(roi_dir, space='Schaefer', level='ROI'):
    """加载ROI信息 - 个体化版本"""
    info_file = os.path.join(roi_dir, 'roi_info.json')
    
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            full_info = json.load(f)
        
        # 根据空间选择对应的定义
        if space == 'Schaefer':
            definition_key = 'schaefer_definition'
        else:  # Kong
            definition_key = 'kong_definition'
        
        space_info = full_info.get(definition_key, {})
        
        roi_info = {
            'n_dmn': space_info.get('n_dmn', 0),
            'n_ecn': space_info.get('n_ecn', 0),
            'space': space,
            'level': level,
            'network_order': space_info.get('network_order', [])
        }
        return roi_info
    
    return None


def louvain_consensus(W, n_iter=500, tau=0.1, reps=500, random_state=None, methods_dir=None):
    """
    正确实现的Louvain共识方法，复制MATLAB代码中被注释的版本
    
    对应MATLAB代码 (rsfc_DynamicIndex.m 中的 Dynamic Modularity method #2):
    ```matlab
    for loop = 1:500
        [cc(:,loop),qq(loop,1)] = modularity_louvain_und_sign(mtd(:,:,t));  
    end
    ci(:,t) = consensus_und(agreement_weighted(cc,qq),0.1,500);
    q(t,1) = mean(qq);
    ```
    
    Parameters
    ----------
    W : ndarray
        连接矩阵 (MTD矩阵，可能包含正负权重)
    n_iter : int
        运行Louvain算法的次数 (论文声称500次)
    tau : float
        consensus_und的阈值参数 (默认0.1)
    reps : int
        consensus_und的重复次数参数 (默认500)
    random_state : int, optional
        随机种子
    methods_dir : str, optional
        方法模块路径
    
    Returns
    -------
    ci_consensus : ndarray
        共识社区分配向量 (1-indexed)
    q_mean : float
        平均模块度
    """
    if methods_dir:
        add_methods_to_path(methods_dir)
    
    from modularity_louvain_und_sign_optimized import modularity_louvain_und_sign_optimized
    from agreement_weighted_optimized import agreement_weighted
    from consensus_und_optimized import consensus_und_single
    
    nNodes = W.shape[0]
    
    # 当n_iter=1时，单次运行（与OSF实际代码行为一致）
    if n_iter == 1:
        return modularity_louvain_und_sign_optimized(W, qtype='sta', seed=random_state)
    
    # 存储所有分区结果和Q值
    all_ci = np.zeros((nNodes, n_iter), dtype=np.int64)
    all_q = np.zeros(n_iter, dtype=np.float64)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # 运行n_iter次Louvain算法
    for i in range(n_iter):
        # 使用modularity_louvain_und_sign处理有正负权重的MTD矩阵
        ci, q = modularity_louvain_und_sign_optimized(W, qtype='sta')
        all_ci[:, i] = ci
        all_q[i] = q
    
    # 使用agreement_weighted计算加权协议矩阵（权重是Q值）
    # 对应MATLAB: agreement_weighted(cc, qq)
    D = agreement_weighted(all_ci, all_q, parallel=False)
    
    # 使用consensus_und进行共识聚类
    # 对应MATLAB: consensus_und(D, 0.1, 500)
    ci_consensus = consensus_und_single(D, tau=tau, reps=reps, seed=random_state)
    
    return ci_consensus, np.mean(all_q)


def process_single_subject(ts_data, window=14, direction=1, trim=1, 
                            n_clusters=2, n_init=100, 
                            n_louvain_iter=1, louvain_tau=0.1, louvain_reps=500,
                            random_state=None, methods_dir=None, verbose=False):
    """
    处理单个被试单个session的动态功能分析
    
    Parameters
    ----------
    n_louvain_iter : int
        每个时间点Louvain的迭代次数
        - 1: 单次运行 (OSF实际代码行为)
        - >1: 共识聚类 (论文声称方法)
    louvain_tau : float
        consensus_und的阈值参数 (默认0.1)
    louvain_reps : int
        consensus_und的重复次数参数 (默认500)
    n_init : int
        K-means的初始化次数
    """
    if methods_dir and methods_dir not in sys.path:
        sys.path.insert(0, methods_dir)
    
    from coupling_optimized import coupling_optimized
    from modularity_louvain_und_sign_optimized import modularity_louvain_und_sign_optimized
    from module_degree_zscore_optimized import module_degree_zscore
    from participation_coef_sign_optimized import participation_coef_sign
    from step4_optimized import compute_cartographic_profile
    from step5_optimized import run_step5
    from final_step_optimized import compute_switch_frequency
    
    n_timepoints, n_rois = ts_data.shape
    
    if n_timepoints < window + 10:
        raise ValueError(f"Time series too short: {n_timepoints} < {window + 10}")
    
    # Step 2: Time-resolved Functional Connectivity (MTD)
    mtd = coupling_optimized(ts_data, window, direction, trim, parallel=False)
    nNodes, _, nTime = mtd.shape
    
    # Step 3: Graph Theoretical Measures
    ci = np.zeros((nNodes, nTime), dtype=np.int32)
    q = np.zeros(nTime)
    WT = np.zeros((nNodes, nTime))
    BT = np.zeros((nNodes, nTime))
    
    for t in range(nTime):
        mtd_t = mtd[:, :, t]
        # 确保对称性
        mtd_t = (mtd_t + mtd_t.T) / 2
        
        # 根据n_louvain_iter决定是否使用共识
        if n_louvain_iter > 1:
            # 共识聚类模式（论文声称的方法）
            ci_t, q_t = louvain_consensus(
                mtd_t, 
                n_iter=n_louvain_iter, 
                tau=louvain_tau,
                reps=louvain_reps,
                random_state=random_state + t if random_state else None,
                methods_dir=methods_dir
            )
            ci[:, t] = ci_t
            q[t] = q_t
        else:
            # 单次运行模式（OSF实际代码行为）
            # 使用modularity_louvain_und_sign因为MTD可能有负权重
            ci[:, t], q[t] = modularity_louvain_und_sign_optimized(
                mtd_t, qtype='sta', 
                seed=random_state + t if random_state else None
            )
        
        # Module Degree Z-score
        WT[:, t] = module_degree_zscore(mtd_t, ci[:, t], flag=0)
        
        # Participation Coefficient (for signed networks)
        Ppos, _ = participation_coef_sign(mtd_t, ci[:, t], use_parallel=False)
        BT[:, t] = Ppos
    
    # Step 4: 2D Cartographic Profile
    CP, xbins, ybins = compute_cartographic_profile(BT, WT, parallel=False)
    xNumBins = len(xbins)
    yNumBins = len(ybins)
    
    # Step 5: K-means clustering
    step5_result = run_step5(
        CP, WT, BT, 
        xNumBins, yNumBins,
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state
    )
    
    final_idx = step5_result['idx']
    integration_count = step5_result['integration_count']
    segregation_count = step5_result['segregation_count']
    
    # Step 6: Compute switch frequency
    switch_count, switch_f = compute_switch_frequency(final_idx)
    
    switch_rate = switch_count / nTime * 100
    
    total_time = integration_count + segregation_count
    balance = (integration_count - segregation_count) / total_time if total_time > 0 else 0
    
    return {
        'integration_count': integration_count,
        'segregation_count': segregation_count,
        'switch_count': switch_count,
        'switch_rate': switch_rate,
        'balance': balance,
        'BT_value': step5_result['BT_value'],
        'tstat': step5_result['tstat'],
        'nTime': nTime
    }


def process_single_task(task_info):
    """并行任务包装函数"""
    try:
        ts_data = task_info['ts_data']
        sub_id = task_info['sub_id']
        session = task_info['session']
        params = task_info['params']
        methods_dir = task_info['methods_dir']
        
        result = process_single_subject(
            ts_data,
            window=params['window'],
            direction=params['direction'],
            trim=params['trim'],
            n_init=params['n_init'],
            n_louvain_iter=params['n_louvain_iter'],
            louvain_tau=params['louvain_tau'],
            louvain_reps=params['louvain_reps'],
            random_state=params['random_state'],
            methods_dir=methods_dir,
            verbose=False
        )
        
        return {
            'sub_id': sub_id,
            'session': session,
            'integration_count': result['integration_count'],
            'segregation_count': result['segregation_count'],
            'switch_count': result['switch_count'],
            'switch_rate': result['switch_rate'],
            'balance': result['balance'],
            'nTime': result['nTime'],
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'sub_id': task_info['sub_id'],
            'session': task_info['session'],
            'error': str(e),
            'status': 'error'
        }


def compute_icc(data_matrix, icc_type='ICC(2,1)'):
    """计算组内相关系数"""
    valid_mask = ~np.any(np.isnan(data_matrix), axis=1)
    data = data_matrix[valid_mask, :]
    
    n, k = data.shape
    
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    grand_mean = np.mean(data)
    subject_means = np.mean(data, axis=1)
    session_means = np.mean(data, axis=0)
    
    SS_total = np.sum((data - grand_mean) ** 2)
    SS_between = k * np.sum((subject_means - grand_mean) ** 2)
    SS_within = SS_total - SS_between
    SS_session = n * np.sum((session_means - grand_mean) ** 2)
    SS_error = SS_within - SS_session
    
    df_between = n - 1
    df_session = k - 1
    df_error = (n - 1) * (k - 1)
    
    MS_between = SS_between / df_between
    MS_session = SS_session / df_session
    MS_error = SS_error / df_error if df_error > 0 else 1e-10
    
    icc_num = MS_between - MS_error
    icc_denom = MS_between + (k - 1) * MS_error + k * (MS_session - MS_error) / n
    
    if icc_denom == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    icc = icc_num / icc_denom
    
    f_value = MS_between / MS_error if MS_error > 0 else np.inf
    p_value = 1 - stats.f.cdf(f_value, df_between, df_error) if df_error > 0 else np.nan
    
    if np.isfinite(f_value) and df_error > 0:
        f_lower = f_value / stats.f.ppf(0.975, df_between, df_error)
        f_upper = f_value * stats.f.ppf(0.975, df_error, df_between)
        ci_lower = (f_lower - 1) / (f_lower + k - 1)
        ci_upper = (f_upper - 1) / (f_upper + k - 1)
    else:
        ci_lower, ci_upper = np.nan, np.nan
    
    return icc, f_value, p_value, ci_lower, ci_upper


def apply_tsnr_filter(session_data, all_sessions, methods_dir, threshold_sd=2.0, verbose=True):
    """对session数据应用tSNR过滤"""
    if methods_dir not in sys.path:
        sys.path.insert(0, methods_dir)
    
    from tSNR_optimized import compute_tsnr, get_tsnr_mask, warmup_jit
    
    warmup_jit()
    
    filtered_session_data = {}
    tsnr_stats = {'by_session': {}, 'global': {}}
    all_tsnr_values = []
    
    if verbose:
        print(f"\n  Computing tSNR for each session...")
    
    for ses in all_sessions:
        rsfc_list = session_data[ses]['rsfc']
        sub_ids = session_data[ses]['sub_ids']
        
        filtered_rsfc_list = []
        session_tsnr_all = []
        
        for s, ts_data in enumerate(rsfc_list):
            tsnr = compute_tsnr(ts_data)
            session_tsnr_all.append(tsnr)
            all_tsnr_values.append(tsnr)
        
        session_mean_tsnr = np.nanmean(np.vstack(session_tsnr_all), axis=0)
        valid_mask, mask_stats = get_tsnr_mask(session_mean_tsnr, threshold_sd=threshold_sd)
        
        for ts_data in rsfc_list:
            filtered_ts = ts_data[:, valid_mask]
            filtered_rsfc_list.append(filtered_ts)
        
        filtered_session_data[ses] = {
            'rsfc': filtered_rsfc_list,
            'sub_ids': sub_ids
        }
        
        tsnr_stats['by_session'][ses] = {
            'mean_tsnr': float(mask_stats['mean_tsnr']),
            'std_tsnr': float(mask_stats['std_tsnr']),
            'n_original': int(len(session_mean_tsnr)),
            'n_valid': int(mask_stats['n_valid']),
            'n_excluded': int(mask_stats['n_excluded']),
            'exclusion_rate': float(mask_stats['n_excluded'] / len(session_mean_tsnr) * 100)
        }
        
        if verbose:
            print(f"    {ses}: {mask_stats['n_excluded']}/{len(session_mean_tsnr)} features excluded "
                  f"({mask_stats['n_excluded']/len(session_mean_tsnr)*100:.1f}%)")
    
    all_tsnr_concat = np.concatenate(all_tsnr_values)
    valid_tsnr = all_tsnr_concat[np.isfinite(all_tsnr_concat)]
    
    tsnr_stats['global'] = {
        'mean_tsnr': float(np.mean(valid_tsnr)),
        'std_tsnr': float(np.std(valid_tsnr, ddof=1)),
        'threshold_sd': threshold_sd
    }
    
    if verbose:
        print(f"\n  Global tSNR statistics:")
        print(f"    Mean ± SD: {tsnr_stats['global']['mean_tsnr']:.2f} ± {tsnr_stats['global']['std_tsnr']:.2f}")
    
    return filtered_session_data, tsnr_stats


def generate_output_dirname(louv, louv_tau, louv_reps, kmns, motion, tsnr, space, level):
    """
    生成扁平化输出目录名称
    
    Examples:
        Louv1_Kmns100_noMotion_notSNR_Schaefer_ROI
        Louv500_tau0.1_reps500_Kmns200_yesMotion_yestSNR_Kong_VoxVert
    """
    motion_str = "yesMotion" if motion else "noMotion"
    tsnr_str = "yestSNR" if tsnr else "notSNR"
    space_str = space  # Schaefer 或 Kong
    level_str = level
    
    if louv > 1:
        # 共识模式: 包含tau和reps参数
        return f"Louv{louv}_tau{louv_tau}_reps{louv_reps}_Kmns{kmns}_{motion_str}_{tsnr_str}_{space_str}_{level_str}"
    else:
        # 单次运行模式
        return f"Louv{louv}_Kmns{kmns}_{motion_str}_{tsnr_str}_{space_str}_{level_str}"


def run_single_analysis(roi_dir, behavior_file, methods_dir, output_dir,
                        space='Schaefer', level='ROI',
                        window=14, direction=1, trim=1, 
                        n_init=100, n_louvain_iter=1, 
                        louvain_tau=0.1, louvain_reps=500,
                        random_state=42,
                        n_jobs=None, 
                        apply_tsnr=False, tsnr_threshold_sd=2.0,
                        apply_motion=False, excluded_sessions=None,
                        skip_frames=5, log_buffer=None):
    """
    运行单个空间+级别的动态功能分析
    """
    os.makedirs(output_dir, exist_ok=True)
    
    add_methods_to_path(methods_dir)
    
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    elif n_jobs <= 0:
        n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)
    
    level_desc = "Vertex" if level == "Vox_Vert" else "ROI"
    
    def log(msg):
        print(msg)
        if log_buffer is not None:
            log_buffer.append(msg)
    
    log("=" * 70)
    log(f"Dynamic Functional Analysis: {space} - {level_desc} Level")
    log(f"Replicating: Chen et al. (2025) Communications Biology")
    log(f"Data format: Individualized parcellation")
    log("=" * 70)
    log(f"\nConfiguration:")
    log(f"  Parallel jobs: {n_jobs}")
    log(f"  Louvain iterations per timepoint: {n_louvain_iter}")
    if n_louvain_iter > 1:
        log(f"    Consensus tau: {louvain_tau}")
        log(f"    Consensus reps: {louvain_reps}")
        log(f"    (Using agreement_weighted + consensus_und)")
    else:
        log(f"    (Single run mode - OSF actual behavior)")
    log(f"  K-means n_init: {n_init}")
    log(f"  Skip frames: {skip_frames}")
    log(f"  tSNR filtering: {'ENABLED' if apply_tsnr else 'DISABLED'}")
    log(f"  Motion filtering: {'ENABLED' if apply_motion else 'DISABLED'}")
    if apply_motion and excluded_sessions:
        log(f"    Excluded sessions: {len(excluded_sessions)}")
    
    # === 1. 加载数据 ===
    log(f"\n[1/6] Loading data by session from {space}/{level}...")
    
    session_data, all_sessions = load_roi_data_by_session_individualized(roi_dir, space, level)
    n_sessions = len(all_sessions)
    
    log(f"  Found {n_sessions} sessions: {all_sessions}")
    
    for ses in all_sessions:
        n_subs = len(session_data[ses]['rsfc'])
        if n_subs > 0:
            n_tp = session_data[ses]['rsfc'][0].shape[0]
            n_features = session_data[ses]['rsfc'][0].shape[1]
            log(f"    {ses}: {n_subs} subjects, {n_tp} timepoints, {n_features} features")
    
    # === 1.1 应用skip_frames ===
    if skip_frames > 0:
        log(f"\n  Skipping first {skip_frames} frames from each session...")
        for ses in all_sessions:
            rsfc_list = session_data[ses]['rsfc']
            skipped_rsfc_list = []
            for ts_data in rsfc_list:
                if ts_data.shape[0] > skip_frames:
                    skipped_rsfc_list.append(ts_data[skip_frames:, :])
                else:
                    skipped_rsfc_list.append(ts_data)
            session_data[ses]['rsfc'] = skipped_rsfc_list
        
        log(f"  After skipping {skip_frames} frames:")
        for ses in all_sessions:
            n_subs = len(session_data[ses]['rsfc'])
            if n_subs > 0:
                n_tp_after = session_data[ses]['rsfc'][0].shape[0]
                log(f"    {ses}: {n_tp_after} timepoints remaining")
    
    # 加载行为数据
    behavior_df = pd.read_csv(behavior_file)
    log(f"  Behavior data: {len(behavior_df)} subjects")
    
    # 加载ROI信息
    roi_info = load_roi_info_individualized(roi_dir, space, level)
    if roi_info:
        n_dmn = roi_info.get('n_dmn', 0)
        n_ecn = roi_info.get('n_ecn', 0)
        log(f"  ROIs: {n_dmn} DMN + {n_ecn} ECN = {n_dmn + n_ecn} total")
    
    # === 1.5 应用tSNR过滤 ===
    tsnr_stats = None
    if apply_tsnr:
        log(f"\n[1.5/6] Applying tSNR filtering (threshold: ±{tsnr_threshold_sd}SD)...")
        session_data, tsnr_stats = apply_tsnr_filter(
            session_data, all_sessions, methods_dir, 
            threshold_sd=tsnr_threshold_sd, verbose=True
        )
        
        log(f"\n  After tSNR filtering:")
        for ses in all_sessions:
            if len(session_data[ses]['rsfc']) > 0:
                n_features_after = session_data[ses]['rsfc'][0].shape[1]
                log(f"    {ses}: {n_features_after} features remaining")
    
    # === 2. 预热JIT ===
    log("\n[2/6] Warming up JIT compilation...")
    
    first_ses = all_sessions[0]
    if len(session_data[first_ses]['rsfc']) > 0:
        test_data = session_data[first_ses]['rsfc'][0][:50, :min(50, session_data[first_ses]['rsfc'][0].shape[1])]
        _ = process_single_subject(
            test_data, window=14, n_init=2, 
            n_louvain_iter=1, louvain_tau=0.1, louvain_reps=10,
            random_state=0, methods_dir=methods_dir, verbose=False
        )
    log("  JIT warmup complete")
    
    # === 3. 准备并行任务 ===
    log(f"\n[3/6] Processing all subjects by session (parallel)...")
    log(f"  Parameters: window={window}, direction={direction}, trim={trim}, "
        f"n_init={n_init}, n_louvain_iter={n_louvain_iter}")
    if n_louvain_iter > 1:
        log(f"  Consensus params: tau={louvain_tau}, reps={louvain_reps}")
    
    # 创建任务列表（考虑motion排除）
    tasks = []
    n_excluded_by_motion = 0
    
    for ses in all_sessions:
        rsfc_list = session_data[ses]['rsfc']
        sub_ids = session_data[ses]['sub_ids']
        
        for s, (ts_data, sub_id) in enumerate(zip(rsfc_list, sub_ids)):
            # 检查是否需要排除
            if apply_motion and excluded_sessions:
                if (sub_id, ses) in excluded_sessions:
                    n_excluded_by_motion += 1
                    continue
            
            task = {
                'ts_data': ts_data,
                'sub_id': sub_id,
                'session': ses,
                'params': {
                    'window': window,
                    'direction': direction,
                    'trim': trim,
                    'n_init': n_init,
                    'n_louvain_iter': n_louvain_iter,
                    'louvain_tau': louvain_tau,
                    'louvain_reps': louvain_reps,
                    'random_state': random_state + hash(sub_id + ses) % 10000
                },
                'methods_dir': methods_dir
            }
            tasks.append(task)
    
    if apply_motion:
        log(f"  Motion filtering: {n_excluded_by_motion} session-runs excluded")
    
    total_runs = len(tasks)
    log(f"  Total tasks: {total_runs}")
    
    # === 执行并行处理 ===
    all_results = []
    start_time = time.time()
    
    if n_jobs == 1:
        log("  Running in single-process mode...")
        for task in tqdm(tasks, desc="  Processing"):
            result = process_single_task(task)
            all_results.append(result)
    else:
        log(f"  Running with {n_jobs} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(process_single_task, task): i for i, task in enumerate(tasks)}
            
            with tqdm(total=total_runs, desc="  Processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)
                    pbar.update(1)
    
    elapsed = time.time() - start_time
    
    n_success = sum(1 for r in all_results if r.get('status') == 'success')
    n_error = sum(1 for r in all_results if r.get('status') == 'error')
    
    log(f"\n  Completed in {elapsed:.1f}s ({elapsed/max(1,total_runs):.2f}s per run)")
    log(f"  Success: {n_success}, Errors: {n_error}")
    
    if n_error > 0:
        log("  Errors:")
        for r in all_results:
            if r.get('status') == 'error':
                log(f"    {r['sub_id']} {r['session']}: {r.get('error', 'Unknown error')}")
    
    # === 4. 整理结果 ===
    log("\n[4/6] Aggregating results...")
    
    session_df = pd.DataFrame([r for r in all_results if r.get('status') == 'success'])
    
    log(f"  Total valid session-runs: {len(session_df)}")
    log(f"  Unique subjects: {session_df['sub_id'].nunique()}")
    
    # 计算每个被试的平均值
    subject_means = session_df.groupby('sub_id').agg({
        'switch_count': 'mean',
        'switch_rate': 'mean',
        'balance': 'mean',
        'integration_count': 'mean',
        'segregation_count': 'mean',
        'nTime': 'mean'
    }).reset_index()
    
    subject_stds = session_df.groupby('sub_id').agg({
        'switch_count': 'std',
        'switch_rate': 'std',
        'balance': 'std'
    }).reset_index()
    subject_stds.columns = ['sub_id', 'switch_count_std', 'switch_rate_std', 'balance_std']
    
    session_counts = session_df.groupby('sub_id').size().reset_index(name='n_sessions')
    
    subject_df = subject_means.merge(subject_stds, on='sub_id').merge(session_counts, on='sub_id')
    
    subject_df = subject_df.rename(columns={
        'switch_count': 'switch',
        'switch_rate': 'switch_rate',
        'balance': 'dytrad'
    })
    
    # 与行为数据合并
    behavior_cols = ['sub_id']
    optional_cols = ['gender', 'age', 'crea', 'IQ', 'CAQ_log', 'CBQ', 'O']
    for col in optional_cols:
        if col in behavior_df.columns:
            behavior_cols.append(col)
    
    merged_df = pd.merge(subject_df, behavior_df[behavior_cols], on='sub_id', how='inner')
    log(f"  Matched with behavior: {len(merged_df)} subjects")
    
    # === 5. ICC分析 ===
    log("\n[5/6] Computing ICC (test-retest reliability)...")
    
    unique_subs = sorted(session_df['sub_id'].unique())
    unique_sessions = sorted(session_df['session'].unique())
    
    icc_results = {}
    
    for metric in ['switch_count', 'switch_rate', 'balance']:
        metric_matrix = np.full((len(unique_subs), len(unique_sessions)), np.nan)
        
        for i, sub in enumerate(unique_subs):
            for j, ses in enumerate(unique_sessions):
                mask = (session_df['sub_id'] == sub) & (session_df['session'] == ses)
                if mask.sum() > 0:
                    metric_matrix[i, j] = session_df.loc[mask, metric].values[0]
        
        icc, f_val, p_val, ci_lower, ci_upper = compute_icc(metric_matrix)
        
        icc_results[metric] = {
            'ICC': float(icc) if np.isfinite(icc) else None,
            'F': float(f_val) if np.isfinite(f_val) else None,
            'p': float(p_val) if np.isfinite(p_val) else None,
            'CI_lower': float(ci_lower) if np.isfinite(ci_lower) else None,
            'CI_upper': float(ci_upper) if np.isfinite(ci_upper) else None
        }
        
        log(f"  {metric}:")
        log(f"    ICC(2,1) = {icc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        log(f"    F = {f_val:.2f}, p = {p_val:.4f}")
    
    # === 6. 统计分析 ===
    log("\n[6/6] Statistical analysis...")
    
    stats_results = {
        'icc': icc_results,
        'space': space,
        'level': level,
        'n_louvain_iter': n_louvain_iter,
        'louvain_tau': louvain_tau if n_louvain_iter > 1 else None,
        'louvain_reps': louvain_reps if n_louvain_iter > 1 else None,
        'n_init': n_init,
        'tsnr_filtering': apply_tsnr,
        'motion_filtering': apply_motion,
        'n_excluded_by_motion': n_excluded_by_motion if apply_motion else 0,
        'processing_time_seconds': elapsed,
        'data_format': 'individualized_parcellation'
    }
    
    if tsnr_stats is not None:
        stats_results['tsnr_stats'] = tsnr_stats
    
    if 'crea' in merged_df.columns:
        from scipy.stats import pearsonr
        
        valid_mask = merged_df['crea'].notna() & merged_df['switch'].notna()
        crea_vals = merged_df.loc[valid_mask, 'crea'].values
        switch_vals = merged_df.loc[valid_mask, 'switch'].values
        switch_rate_vals = merged_df.loc[valid_mask, 'switch_rate'].values
        balance_vals = merged_df.loc[valid_mask, 'dytrad'].values
        
        r_switch, p_switch = pearsonr(switch_vals, crea_vals)
        log(f"\n  === Switch Count vs Creativity ===")
        log(f"  Pearson r = {r_switch:.4f}, p = {p_switch:.4f}, N = {len(crea_vals)}")
        
        stats_results['switch_creativity'] = {
            'r': float(r_switch),
            'p': float(p_switch),
            'n': int(len(crea_vals))
        }
        
        r_rate, p_rate = pearsonr(switch_rate_vals, crea_vals)
        log(f"\n  === Switch Rate vs Creativity ===")
        log(f"  Pearson r = {r_rate:.4f}, p = {p_rate:.4f}")
        
        stats_results['switch_rate_creativity'] = {
            'r': float(r_rate),
            'p': float(p_rate)
        }
        
        r_balance, p_balance = pearsonr(balance_vals, crea_vals)
        log(f"\n  === Balance vs Creativity ===")
        log(f"  Pearson r = {r_balance:.4f}, p = {p_balance:.4f}")
        
        stats_results['balance_creativity'] = {
            'r': float(r_balance),
            'p': float(p_balance)
        }
        
        # 二次项检验
        coeffs = np.polyfit(balance_vals, crea_vals, 2)
        y_pred = np.polyval(coeffs, balance_vals)
        ss_res = np.sum((crea_vals - y_pred) ** 2)
        ss_tot = np.sum((crea_vals - np.mean(crea_vals)) ** 2)
        r2_quad = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        _, _, r_lin, _, _ = stats.linregress(balance_vals, crea_vals)
        
        log(f"\n  === Quadratic Relationship (Balance vs Creativity) ===")
        log(f"  Linear R² = {r_lin**2:.4f}")
        log(f"  Quadratic R² = {r2_quad:.4f}")
        log(f"  Quadratic coefficients: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}")
        
        if coeffs[0] < 0:
            log(f"  → Inverted-U shape detected (negative quadratic term)")
        
        stats_results['quadratic_fit'] = {
            'linear_r2': float(r_lin**2),
            'quadratic_r2': float(r2_quad),
            'coefficients': [float(c) for c in coeffs]
        }
    
    # 与智力的相关
    if 'IQ' in merged_df.columns:
        valid_iq = merged_df['IQ'].notna() & merged_df['switch'].notna()
        if valid_iq.sum() > 10:
            iq_vals = merged_df.loc[valid_iq, 'IQ'].values
            switch_iq = merged_df.loc[valid_iq, 'switch'].values
            r_iq, p_iq = pearsonr(switch_iq, iq_vals)
            log(f"\n  === Switch Frequency vs Intelligence ===")
            log(f"  Pearson r = {r_iq:.4f}, p = {p_iq:.4f}, N = {valid_iq.sum()}")
            
            stats_results['switch_intelligence'] = {
                'r': float(r_iq),
                'p': float(p_iq),
                'n': int(valid_iq.sum())
            }
    
    # === 保存结果 ===
    log("\n" + "=" * 70)
    log("Saving results...")
    
    session_df.to_csv(os.path.join(output_dir, 'dynamic_results_by_session.csv'), index=False)
    log(f"  Saved: dynamic_results_by_session.csv")
    
    merged_df.to_csv(os.path.join(output_dir, 'dynamic_results_subject_mean.csv'), index=False)
    log(f"  Saved: dynamic_results_subject_mean.csv")
    
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_results, f, indent=2)
    log(f"  Saved: statistics.json")
    
    Trady_results = merged_df[['integration_count', 'segregation_count', 'switch']].values
    savemat(os.path.join(output_dir, 'Trady_results_mtd.mat'), {
        'Trady_results': Trady_results,
        'sub_ids': np.array(merged_df['sub_id'].tolist(), dtype=object)
    })
    log(f"  Saved: Trady_results_mtd.mat")
    
    # === 打印汇总 ===
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    
    log(f"\nConfiguration:")
    log(f"  Space: {space}")
    log(f"  Level: {level} ({level_desc})")
    log(f"  Louvain iterations: {n_louvain_iter}")
    if n_louvain_iter > 1:
        log(f"    Consensus tau: {louvain_tau}")
        log(f"    Consensus reps: {louvain_reps}")
    log(f"  K-means n_init: {n_init}")
    log(f"  tSNR filtering: {'Yes (±' + str(tsnr_threshold_sd) + 'SD)' if apply_tsnr else 'No'}")
    log(f"  Motion filtering: {'Yes' if apply_motion else 'No'}")
    
    log(f"\nDataset:")
    log(f"  Total subjects: {len(merged_df)}")
    log(f"  Sessions per subject: {merged_df['n_sessions'].mean():.1f} ± {merged_df['n_sessions'].std():.1f}")
    log(f"  Time windows per session: {session_df['nTime'].mean():.0f}")
    
    log(f"\nDynamic indices (subject means ± SD):")
    log(f"  Switch count: {merged_df['switch'].mean():.1f} ± {merged_df['switch'].std():.1f}")
    log(f"  Switch rate: {merged_df['switch_rate'].mean():.2f} ± {merged_df['switch_rate'].std():.2f} per 100 windows")
    log(f"  Balance: {merged_df['dytrad'].mean():.4f} ± {merged_df['dytrad'].std():.4f}")
    
    log(f"\nICC (test-retest reliability):")
    for metric, res in icc_results.items():
        icc_val = res['ICC'] if res['ICC'] is not None else np.nan
        interpretation = "Excellent" if icc_val > 0.75 else "Good" if icc_val > 0.5 else "Moderate" if icc_val > 0.25 else "Poor"
        log(f"  {metric}: ICC = {icc_val:.3f} ({interpretation})")
    
    if 'switch_creativity' in stats_results:
        log(f"\nKey findings:")
        r = stats_results['switch_creativity']['r']
        p = stats_results['switch_creativity']['p']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        log(f"  Switch-Creativity correlation: r={r:.3f}, p={p:.4f} {sig}")
    
    log(f"\nProcessing time: {elapsed:.1f}s ({elapsed/max(1,total_runs):.2f}s per subject-session)")
    log(f"\nOutput saved to: {output_dir}")
    log("=" * 70)
    
    return merged_df, session_df, stats_results, elapsed


def run_analysis(roi_dir, behavior_file, methods_dir, output_dir,
                 space=None, level=None,
                 window=14, direction=1, trim=1, 
                 n_init=200, n_louvain_iter=1, 
                 louvain_tau=0.1, louvain_reps=500,
                 random_state=42,
                 n_jobs=None, 
                 enable_tsnr=False, tsnr_threshold_sd=2.0,
                 enable_motion=False, motion_file=None,
                 skip_frames=5):
    """
    运行动态功能分析（v2.4 个体化版本）
    """
    # 加载motion排除列表
    excluded_sessions = None
    if enable_motion and motion_file:
        excluded_sessions = load_motion_exclusion_list(motion_file)
        print(f"\nLoaded motion exclusion list: {len(excluded_sessions)} sessions to exclude")
        for sub_id, ses_id in sorted(excluded_sessions)[:5]:
            print(f"  {sub_id} {ses_id}")
        if len(excluded_sessions) > 5:
            print(f"  ... and {len(excluded_sessions) - 5} more")
    
    # 定义所有可能的组合 (个体化版本使用 Schaefer/Kong)
    all_combinations = [
        ('Schaefer', 'ROI'),
        ('Schaefer', 'Vox_Vert'),
        ('Kong', 'ROI'),
        ('Kong', 'Vox_Vert')
    ]
    
    # 确定要分析的组合
    if space is not None and level is not None:
        combinations = [(space, level)]
    elif space is not None:
        combinations = [(space, 'ROI'), (space, 'Vox_Vert')]
    elif level is not None:
        combinations = [('Schaefer', level), ('Kong', level)]
    else:
        combinations = []
        for sp, lv in all_combinations:
            data_dir = os.path.join(roi_dir, sp, lv)
            if os.path.exists(data_dir):
                combinations.append((sp, lv))
        
        if not combinations:
            raise FileNotFoundError(f"No valid data directories found in {roi_dir}")
    
    # 确定要分析的tSNR模式
    if enable_tsnr:
        tsnr_modes = [(False, False), (True, True)]
    else:
        tsnr_modes = [(False, False)]
    
    # 确定要分析的Motion模式
    if enable_motion:
        motion_modes = [(False, False), (True, True)]
    else:
        motion_modes = [(False, False)]
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 主日志缓冲区
    master_log = []
    
    master_log.append("=" * 70)
    master_log.append(f"Dynamic Functional Analysis - v2.4 Individualized")
    master_log.append(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    master_log.append("=" * 70)
    master_log.append(f"\nGlobal Configuration:")
    master_log.append(f"  Data format: Individualized parcellation (per-subject files)")
    master_log.append(f"  Louvain iterations: {n_louvain_iter}")
    if n_louvain_iter > 1:
        master_log.append(f"    Consensus tau: {louvain_tau}")
        master_log.append(f"    Consensus reps: {louvain_reps}")
        master_log.append(f"    Method: agreement_weighted + consensus_und (MATLAB-equivalent)")
    else:
        master_log.append(f"    Mode: Single run (OSF actual behavior)")
    master_log.append(f"  K-means n_init: {n_init}")
    master_log.append(f"  tSNR filtering: {'ENABLED' if enable_tsnr else 'DISABLED'}")
    master_log.append(f"  Motion filtering: {'ENABLED' if enable_motion else 'DISABLED'}")
    master_log.append(f"  Skip frames: {skip_frames}")
    master_log.append(f"\nWill analyze {len(combinations)} space/level combination(s):")
    for sp, lv in combinations:
        master_log.append(f"  - {sp}/{lv}")
    
    if enable_tsnr:
        master_log.append(f"\nFor each combination, will run:")
        master_log.append(f"  - Without tSNR filtering")
        master_log.append(f"  - With tSNR filtering (±{tsnr_threshold_sd}SD)")
    
    if enable_motion:
        if not enable_tsnr:
            master_log.append(f"\nFor each combination, will run:")
        master_log.append(f"  - Without Motion exclusion")
        master_log.append(f"  - With Motion exclusion")
    
    for msg in master_log:
        print(msg)
    
    all_results = {}
    pipeline_times = []
    
    for sp, lv in combinations:
        for apply_tsnr, tsnr_in_name in tsnr_modes:
            for apply_motion, motion_in_name in motion_modes:
                # 生成输出目录名
                dirname = generate_output_dirname(
                    louv=n_louvain_iter,
                    louv_tau=louvain_tau,
                    louv_reps=louvain_reps,
                    kmns=n_init,
                    motion=apply_motion,
                    tsnr=apply_tsnr,
                    space=sp,
                    level=lv
                )
                
                combo_output_dir = os.path.join(output_dir, dirname)
                
                print("\n\n" + "#" * 70)
                print(f"# Analyzing: {dirname}")
                print("#" * 70)
                
                pipeline_log = []
                pipeline_start = time.time()
                
                try:
                    merged_df, session_df, stats_results, elapsed = run_single_analysis(
                        roi_dir=roi_dir,
                        behavior_file=behavior_file,
                        methods_dir=methods_dir,
                        output_dir=combo_output_dir,
                        space=sp,
                        level=lv,
                        window=window,
                        direction=direction,
                        trim=trim,
                        n_init=n_init,
                        n_louvain_iter=n_louvain_iter,
                        louvain_tau=louvain_tau,
                        louvain_reps=louvain_reps,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        apply_tsnr=apply_tsnr,
                        tsnr_threshold_sd=tsnr_threshold_sd,
                        apply_motion=apply_motion,
                        excluded_sessions=excluded_sessions,
                        skip_frames=skip_frames,
                        log_buffer=pipeline_log
                    )
                    
                    pipeline_elapsed = time.time() - pipeline_start
                    
                    all_results[dirname] = {
                        'merged_df': merged_df,
                        'session_df': session_df,
                        'stats': stats_results
                    }
                    
                    pipeline_times.append({
                        'pipeline': dirname,
                        'elapsed_seconds': pipeline_elapsed,
                        'success': True,
                        'switch_creativity_r': stats_results.get('switch_creativity', {}).get('r', None),
                        'switch_creativity_p': stats_results.get('switch_creativity', {}).get('p', None)
                    })
                    
                except Exception as e:
                    pipeline_elapsed = time.time() - pipeline_start
                    print(f"\n  ERROR: Failed to analyze {dirname}")
                    print(f"  {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    pipeline_times.append({
                        'pipeline': dirname,
                        'elapsed_seconds': pipeline_elapsed,
                        'success': False,
                        'error': str(e)
                    })
                    continue
    
    # === 打印最终汇总 ===
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY - All Pipelines")
    print("=" * 70)
    
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("FINAL SUMMARY - All Pipelines")
    summary_lines.append("=" * 70)
    
    total_time = sum(pt['elapsed_seconds'] for pt in pipeline_times)
    summary_lines.append(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
    summary_lines.append(f"\nPipeline Results:")
    
    for pt in pipeline_times:
        status = "✓" if pt['success'] else "✗"
        line = f"\n{status} {pt['pipeline']}"
        summary_lines.append(line)
        print(line)
        
        time_line = f"    Time: {pt['elapsed_seconds']:.1f}s"
        summary_lines.append(time_line)
        print(time_line)
        
        if pt['success']:
            if pt['switch_creativity_r'] is not None:
                r = pt['switch_creativity_r']
                p = pt['switch_creativity_p']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                result_line = f"    Switch-Creativity: r={r:.4f}, p={p:.4f} {sig}"
                summary_lines.append(result_line)
                print(result_line)
        else:
            error_line = f"    Error: {pt.get('error', 'Unknown')}"
            summary_lines.append(error_line)
            print(error_line)
    
    summary_lines.append(f"\nOutput directory: {output_dir}")
    summary_lines.append("=" * 70)
    
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)
    
    # === 保存日志文件 ===
    log_filename = f"output_log_{timestamp}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(log_filepath, 'w', encoding='utf-8') as f:
        # 写入主日志
        for line in master_log:
            f.write(line + '\n')
        
        f.write('\n\n')
        
        # 写入汇总
        for line in summary_lines:
            f.write(line + '\n')
        
        # 写入详细时间记录
        f.write('\n\n' + '=' * 70 + '\n')
        f.write('DETAILED TIMING\n')
        f.write('=' * 70 + '\n')
        
        for pt in pipeline_times:
            f.write(f"\n{pt['pipeline']}:\n")
            f.write(f"  Elapsed: {pt['elapsed_seconds']:.2f}s\n")
            f.write(f"  Success: {pt['success']}\n")
            if pt['success'] and pt['switch_creativity_r'] is not None:
                f.write(f"  Switch-Creativity r: {pt['switch_creativity_r']:.4f}\n")
                f.write(f"  Switch-Creativity p: {pt['switch_creativity_p']:.4f}\n")
            elif not pt['success']:
                f.write(f"  Error: {pt.get('error', 'Unknown')}\n")
        
        f.write(f'\n\nFinished at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    print(f"\nLog saved to: {log_filepath}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='DMN-ECN Dynamic Functional Analysis v2.4 - Individualized Parcellation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v2.4 Individualized Key Changes:
  - Adapted for individualized parcellation data format (per-subject files)
  - Uses Schaefer/Kong network definitions instead of fsaverage6/MNI
  - Same analysis pipeline as the original v2.4 script

Data format difference:
  - Original: rsfc_ses-01.mat (per-session files, all subjects)
  - Individualized: sub-001.mat (per-subject files, all sessions)

Examples:
  # Analyze Schaefer ROI level only
  python run_dynamic_analysis_by_session_parallel_v2.4_individualized.py \\
      --roi_dir /path/to/analysis_output_individualized \\
      --behavior_file /path/to/behavior_data.csv \\
      --methods_dir /path/to/Methods \\
      --output_dir /path/to/dynamic_analysis_individualized \\
      --space Schaefer --level ROI

  # Full analysis with all options
  python run_dynamic_analysis_by_session_parallel_v2.4_individualized.py \\
      --roi_dir /path/to/analysis_output_individualized \\
      --behavior_file /path/to/behavior_data.csv \\
      --methods_dir /path/to/Methods \\
      --output_dir /path/to/dynamic_analysis_individualized \\
      --level ROI \\
      --Louv 500 --LouvTau 0.1 --LouvReps 500 \\
      --Kmns 100 \\
      --tSNR --tSNR_threshold 2.0 \\
      --Motion --motion_file /path/to/excluded_sessions.txt \\
      --n_jobs 23

Output folder naming:
  Louv1_Kmns100_noMotion_notSNR_Schaefer_ROI
  Louv500_tau0.1_reps500_Kmns200_yesMotion_yestSNR_Kong_VoxVert
        """
    )
    
    parser.add_argument('--roi_dir', required=True,
                       help='Directory containing individualized ROI timeseries data')
    parser.add_argument('--behavior_file', required=True,
                       help='Path to behavior data CSV file')
    parser.add_argument('--methods_dir', required=True,
                       help='Directory containing optimized method modules')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    
    parser.add_argument('--space', choices=['Schaefer', 'Kong'], default=None,
                       help='Space to analyze (default: all available)')
    parser.add_argument('--level', choices=['ROI', 'Vox_Vert'], default=None,
                       help='Level to analyze (default: all available)')
    
    parser.add_argument('--window', type=int, default=14,
                       help='MTD window size (default: 14)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--n_jobs', type=int, default=None,
                       help='Number of parallel jobs (default: CPU count - 1)')
    parser.add_argument('--skip_frames', type=int, default=5,
                       help='Number of frames to skip at the beginning (default: 5)')
    
    # Louvain共识参数
    parser.add_argument('--Louv', type=int, default=1,
                       help='Number of Louvain iterations per timepoint (default: 1). '
                            '1=single run (OSF actual), >1=consensus clustering (paper claimed).')
    parser.add_argument('--LouvTau', type=float, default=0.1,
                       help='Consensus_und tau threshold (default: 0.1, as in MATLAB code)')
    parser.add_argument('--LouvReps', type=int, default=500,
                       help='Consensus_und repetitions (default: 500, as in MATLAB code)')
    
    # K-means参数
    parser.add_argument('--Kmns', type=int, default=200,
                       help='Number of K-means initializations (default: 200)')
    
    # Motion过滤
    parser.add_argument('--Motion', action='store_true', default=False,
                       help='Enable motion-based session exclusion')
    parser.add_argument('--motion_file', type=str, default=None,
                       help='Path to motion exclusion list file. Required if --Motion is set.')
    
    # tSNR过滤
    parser.add_argument('--tSNR', action='store_true', default=False,
                       help='Enable tSNR filtering')
    parser.add_argument('--tSNR_threshold', type=float, default=2.0,
                       help='tSNR filtering threshold in standard deviations (default: 2.0)')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.Motion and not args.motion_file:
        parser.error("--motion_file is required when --Motion is set")
    
    run_analysis(
        roi_dir=args.roi_dir,
        behavior_file=args.behavior_file,
        methods_dir=args.methods_dir,
        output_dir=args.output_dir,
        space=args.space,
        level=args.level,
        window=args.window,
        n_init=args.Kmns,
        n_louvain_iter=args.Louv,
        louvain_tau=args.LouvTau,
        louvain_reps=args.LouvReps,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        skip_frames=args.skip_frames,
        enable_tsnr=args.tSNR,
        tsnr_threshold_sd=args.tSNR_threshold,
        enable_motion=args.Motion,
        motion_file=args.motion_file
    )


if __name__ == '__main__':
    main()
