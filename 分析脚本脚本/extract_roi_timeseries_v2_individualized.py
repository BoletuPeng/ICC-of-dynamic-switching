#!/usr/bin/env python3
"""
ROI时间序列提取脚本 v2 - 个体化分割版本
从DeepPrep输出中提取DMN+ECN区域的时间序列，使用个体化分割的atlas

支持两种Networks定义:
- Schaefer: 按Schaefer 2018的网络定义筛选DMN/ECN，然后映射到Kong索引
- Kong: 按Kong 2022的网络定义筛选DMN/ECN

输出目录结构:
    output_dir/
    ├── Schaefer/
    │   ├── ROI/           # ROI级别时间序列 (每个被试一个文件)
    │   └── Vox_Vert/      # 顶点级别时间序列 (每个被试一个文件)
    ├── Kong/
    │   ├── ROI/           # ROI级别时间序列 (每个被试一个文件)
    │   └── Vox_Vert/      # 顶点级别时间序列 (每个被试一个文件)
    └── roi_info.json      # ROI元数据

使用方法:
    python extract_roi_timeseries_v2_individualized.py \
        --bold_dir /path/to/postproc/BOLD \
        --atlas_dir /path/to/atlas_files \
        --ind_parc_dir /path/to/MSHBM_output \
        --output_dir /path/to/output

依赖:
    pip install nibabel numpy pandas scipy
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from nibabel.freesurfer.io import read_annot
import pandas as pd
from scipy.io import savemat, loadmat
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =============================================================================
# 模板文件名定义
# =============================================================================

# Schaefer 2018 annot文件名
SCHAEFER_ANNOT_LH = 'lh.Schaefer2018_300Parcels_17Networks_order.annot'
SCHAEFER_ANNOT_RH = 'rh.Schaefer2018_300Parcels_17Networks_order.annot'

# Kong 2022 annot文件名
KONG_ANNOT_LH = 'lh.Schaefer2018_300Parcels_Kong2022_17Networks_order.annot'
KONG_ANNOT_RH = 'rh.Schaefer2018_300Parcels_Kong2022_17Networks_order.annot'

# Kong到Schaefer的映射文件
MAPPING_FILE = 'Kong2022_to_Schaefer2018_mapping.csv'

# 个体化分割文件模板
IND_PARC_PATTERN = 'sub-{sub}/ind_parcellation_gMSHBM/test_set/6_sess/beta5/Ind_parcellation_MSHBM_sub1_w50_MRF10_beta5.mat'

# 默认并行数
DEFAULT_N_JOBS = 8


# =============================================================================
# 工具函数
# =============================================================================

def convert_to_native_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_annot(annot_path):
    """加载annot文件"""
    labels, ctab, names = read_annot(annot_path)
    names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in names]
    return labels, ctab, names


def get_dmn_ecn_indices_from_names(names, exclude_background=True):
    """
    从名称列表中筛选DMN和ECN的索引
    
    Returns:
        dmn_indices: DMN区域的索引列表
        ecn_indices: ECN区域的索引列表
    """
    dmn_indices = []
    ecn_indices = []
    
    for i, name in enumerate(names):
        if exclude_background and (name == 'Unknown' or name == '???' or 'Background' in name):
            continue
        if 'Default' in name:
            dmn_indices.append(i)
        elif 'Cont' in name:
            ecn_indices.append(i)
    
    return dmn_indices, ecn_indices


def load_individualized_parcellation(mat_path):
    """
    加载个体化分割结果
    
    Returns:
        lh_labels: 左脑顶点标签 (40962,)
        rh_labels: 右脑顶点标签 (40962,)
    """
    mat = loadmat(mat_path)
    lh_labels = mat['lh_labels'].flatten()
    rh_labels = mat['rh_labels'].flatten()
    return lh_labels, rh_labels


def load_bold_surface(lh_path, rh_path):
    """
    加载左右半球的BOLD表面数据
    
    Returns:
        lh_data: (n_timepoints, n_vertices_lh)
        rh_data: (n_timepoints, n_vertices_rh)
    """
    if lh_path.endswith('.gii'):
        lh_img = nib.load(lh_path)
        rh_img = nib.load(rh_path)
        lh_data = np.column_stack([d.data for d in lh_img.darrays])
        rh_data = np.column_stack([d.data for d in rh_img.darrays])
    else:
        lh_data = nib.load(lh_path).get_fdata()
        rh_data = nib.load(rh_path).get_fdata()
        
        if lh_data.ndim == 4:
            lh_data = lh_data.squeeze()
        if rh_data.ndim == 4:
            rh_data = rh_data.squeeze()
            
        # 确保是 (timepoints, vertices) 格式
        if lh_data.shape[0] > lh_data.shape[-1]:
            lh_data = lh_data.T
            rh_data = rh_data.T
    
    return lh_data, rh_data


# =============================================================================
# Schaefer定义的提取函数
# =============================================================================

def build_schaefer_to_kong_mapping(mapping_df, schaefer_lh_names, schaefer_rh_names):
    """
    构建Schaefer ROI名称到Kong个体化分割标签的映射
    
    个体化分割标签规则:
    - 左脑: label = mapping Kong_Index (1-150)
    - 右脑: label = mapping Kong_Index - 1 (151-300, 对应mapping 152-301)
    
    Returns:
        schaefer_name_to_kong_label: {Schaefer名称: Kong个体化分割标签}
    """
    schaefer_name_to_kong_label = {}
    
    for _, row in mapping_df.iterrows():
        kong_idx = row['Kong_Index']
        schaefer_name = row['Schaefer2018_Best_Match']
        
        if 'Background' in str(schaefer_name) or kong_idx == 0 or kong_idx == 151:
            continue
            
        # 确定个体化分割中的标签
        if kong_idx <= 150:
            # 左脑: label = Kong_Index
            ind_label = kong_idx
        else:
            # 右脑: label = Kong_Index - 1
            ind_label = kong_idx - 1
            
        schaefer_name_to_kong_label[schaefer_name] = ind_label
    
    return schaefer_name_to_kong_label


def get_schaefer_dmn_ecn_labels(schaefer_lh_names, schaefer_rh_names, schaefer_name_to_kong_label):
    """
    获取Schaefer定义下DMN/ECN对应的个体化分割标签
    
    Returns:
        lh_dmn_labels: 左脑DMN的个体化分割标签
        lh_ecn_labels: 左脑ECN的个体化分割标签
        rh_dmn_labels: 右脑DMN的个体化分割标签
        rh_ecn_labels: 右脑ECN的个体化分割标签
        label_order: 所有选中标签的顺序 (DMN在前, ECN在后)
        roi_names: 对应的ROI名称
    """
    lh_dmn_labels = []
    lh_ecn_labels = []
    rh_dmn_labels = []
    rh_ecn_labels = []
    
    lh_dmn_names = []
    lh_ecn_names = []
    rh_dmn_names = []
    rh_ecn_names = []
    
    # 处理左脑
    for name in schaefer_lh_names:
        if name in schaefer_name_to_kong_label:
            label = schaefer_name_to_kong_label[name]
            if 'Default' in name:
                lh_dmn_labels.append(label)
                lh_dmn_names.append(name)
            elif 'Cont' in name:
                lh_ecn_labels.append(label)
                lh_ecn_names.append(name)
    
    # 处理右脑
    for name in schaefer_rh_names:
        if name in schaefer_name_to_kong_label:
            label = schaefer_name_to_kong_label[name]
            if 'Default' in name:
                rh_dmn_labels.append(label)
                rh_dmn_names.append(name)
            elif 'Cont' in name:
                rh_ecn_labels.append(label)
                rh_ecn_names.append(name)
    
    # 组合顺序: LH_DMN, RH_DMN, LH_ECN, RH_ECN
    label_order = lh_dmn_labels + rh_dmn_labels + lh_ecn_labels + rh_ecn_labels
    roi_names = lh_dmn_names + rh_dmn_names + lh_ecn_names + rh_ecn_names
    
    return (lh_dmn_labels, lh_ecn_labels, rh_dmn_labels, rh_ecn_labels,
            label_order, roi_names)


# =============================================================================
# Kong定义的提取函数
# =============================================================================

def get_kong_dmn_ecn_labels(kong_lh_names, kong_rh_names):
    """
    获取Kong定义下DMN/ECN对应的个体化分割标签
    
    个体化分割标签规则:
    - 左脑: label = annot索引 (1-150)
    - 右脑: label = annot索引 + 150 (151-300)
    
    Returns:
        类似于get_schaefer_dmn_ecn_labels
    """
    lh_dmn_labels = []
    lh_ecn_labels = []
    rh_dmn_labels = []
    rh_ecn_labels = []
    
    lh_dmn_names = []
    lh_ecn_names = []
    rh_dmn_names = []
    rh_ecn_names = []
    
    # 处理左脑 (annot索引 = 个体化分割标签)
    for annot_idx, name in enumerate(kong_lh_names):
        if annot_idx == 0 or 'Background' in name:
            continue
        if 'Default' in name:
            lh_dmn_labels.append(annot_idx)
            lh_dmn_names.append(name)
        elif 'Cont' in name:
            lh_ecn_labels.append(annot_idx)
            lh_ecn_names.append(name)
    
    # 处理右脑 (个体化分割标签 = annot索引 + 150)
    for annot_idx, name in enumerate(kong_rh_names):
        if annot_idx == 0 or 'Background' in name:
            continue
        ind_label = annot_idx + 150
        if 'Default' in name:
            rh_dmn_labels.append(ind_label)
            rh_dmn_names.append(name)
        elif 'Cont' in name:
            rh_ecn_labels.append(ind_label)
            rh_ecn_names.append(name)
    
    # 组合顺序: LH_DMN, RH_DMN, LH_ECN, RH_ECN
    label_order = lh_dmn_labels + rh_dmn_labels + lh_ecn_labels + rh_ecn_labels
    roi_names = lh_dmn_names + rh_dmn_names + lh_ecn_names + rh_ecn_names
    
    return (lh_dmn_labels, lh_ecn_labels, rh_dmn_labels, rh_ecn_labels,
            label_order, roi_names)


# =============================================================================
# 时间序列提取函数
# =============================================================================

def extract_roi_timeseries(lh_data, rh_data, ind_lh_labels, ind_rh_labels, 
                           lh_parcel_labels, rh_parcel_labels, label_order):
    """
    从表面BOLD数据中提取ROI级别的平均时间序列
    
    Args:
        lh_data, rh_data: BOLD数据 (n_timepoints, n_vertices)
        ind_lh_labels, ind_rh_labels: 个体化分割标签 (n_vertices,)
        lh_parcel_labels: 要提取的左脑parcel标签列表
        rh_parcel_labels: 要提取的右脑parcel标签列表
        label_order: 所有标签的输出顺序
        
    Returns:
        timeseries: (n_timepoints, n_parcels)
    """
    n_timepoints = lh_data.shape[0]
    n_parcels = len(label_order)
    timeseries = np.zeros((n_timepoints, n_parcels))
    
    for i, label in enumerate(label_order):
        if label <= 150:
            # 左脑
            mask = ind_lh_labels == label
            if np.sum(mask) > 0:
                timeseries[:, i] = np.mean(lh_data[:, mask], axis=1)
            else:
                timeseries[:, i] = np.nan
        else:
            # 右脑
            mask = ind_rh_labels == label
            if np.sum(mask) > 0:
                timeseries[:, i] = np.mean(rh_data[:, mask], axis=1)
            else:
                timeseries[:, i] = np.nan
    
    return timeseries


def extract_vertex_timeseries(lh_data, rh_data, ind_lh_labels, ind_rh_labels,
                              lh_parcel_labels, rh_parcel_labels):
    """
    从表面BOLD数据中提取顶点级别的时间序列
    
    Returns:
        timeseries: (n_timepoints, n_selected_vertices)
        vertex_info: 顶点元数据字典
    """
    # 左脑掩码
    lh_mask = np.zeros(len(ind_lh_labels), dtype=bool)
    for label in lh_parcel_labels:
        lh_mask |= (ind_lh_labels == label)
    
    # 右脑掩码
    rh_mask = np.zeros(len(ind_rh_labels), dtype=bool)
    for label in rh_parcel_labels:
        rh_mask |= (ind_rh_labels == label)
    
    # 提取时间序列
    lh_ts = lh_data[:, lh_mask]
    rh_ts = rh_data[:, rh_mask]
    timeseries = np.hstack([lh_ts, rh_ts])
    
    # 记录顶点信息
    vertex_info = {
        'lh_original_indices': np.where(lh_mask)[0].tolist(),
        'lh_parcel_labels': ind_lh_labels[lh_mask].tolist(),
        'rh_original_indices': np.where(rh_mask)[0].tolist(),
        'rh_parcel_labels': ind_rh_labels[rh_mask].tolist(),
        'n_lh_vertices': int(np.sum(lh_mask)),
        'n_rh_vertices': int(np.sum(rh_mask)),
        'hemi_split_index': int(np.sum(lh_mask))
    }
    
    return timeseries, vertex_info


# =============================================================================
# Session处理函数
# =============================================================================

def process_session(bold_dir, sub, ses, ind_lh_labels, ind_rh_labels,
                    schaefer_info, kong_info, desc='regression'):
    """
    处理单个session
    
    Returns:
        results: {
            'schaefer': {'roi': roi_ts, 'vertex': vertex_ts, 'vertex_info': info},
            'kong': {'roi': roi_ts, 'vertex': vertex_ts, 'vertex_info': info}
        }
    """
    func_dir = os.path.join(bold_dir, sub, ses, 'func')
    
    # 查找BOLD文件
    lh_pattern = os.path.join(func_dir, 
        f'{sub}_{ses}_task-rest_hemi-L_space-fsaverage6_desc-{desc}_bold.nii.gz')
    rh_pattern = os.path.join(func_dir,
        f'{sub}_{ses}_task-rest_hemi-R_space-fsaverage6_desc-{desc}_bold.nii.gz')
    
    if not os.path.exists(lh_pattern):
        lh_pattern = os.path.join(func_dir,
            f'{sub}_{ses}_task-rest_hemi-L_space-fsaverage6_bold.nii.gz')
        rh_pattern = os.path.join(func_dir,
            f'{sub}_{ses}_task-rest_hemi-R_space-fsaverage6_bold.nii.gz')
    
    if not os.path.exists(lh_pattern):
        return None
    
    # 加载BOLD数据
    lh_data, rh_data = load_bold_surface(lh_pattern, rh_pattern)
    
    results = {}
    
    # === Schaefer定义 ===
    s_lh_dmn, s_lh_ecn, s_rh_dmn, s_rh_ecn, s_label_order, _ = schaefer_info
    s_lh_all = s_lh_dmn + s_lh_ecn
    s_rh_all = s_rh_dmn + s_rh_ecn
    
    s_roi_ts = extract_roi_timeseries(
        lh_data, rh_data, ind_lh_labels, ind_rh_labels,
        s_lh_all, s_rh_all, s_label_order
    )
    s_vertex_ts, s_vertex_info = extract_vertex_timeseries(
        lh_data, rh_data, ind_lh_labels, ind_rh_labels,
        s_lh_all, s_rh_all
    )
    
    results['schaefer'] = {
        'roi': s_roi_ts,
        'vertex': s_vertex_ts,
        'vertex_info': s_vertex_info
    }
    
    # === Kong定义 ===
    k_lh_dmn, k_lh_ecn, k_rh_dmn, k_rh_ecn, k_label_order, _ = kong_info
    k_lh_all = k_lh_dmn + k_lh_ecn
    k_rh_all = k_rh_dmn + k_rh_ecn
    
    k_roi_ts = extract_roi_timeseries(
        lh_data, rh_data, ind_lh_labels, ind_rh_labels,
        k_lh_all, k_rh_all, k_label_order
    )
    k_vertex_ts, k_vertex_info = extract_vertex_timeseries(
        lh_data, rh_data, ind_lh_labels, ind_rh_labels,
        k_lh_all, k_rh_all
    )
    
    results['kong'] = {
        'roi': k_roi_ts,
        'vertex': k_vertex_ts,
        'vertex_info': k_vertex_info
    }
    
    return results


# =============================================================================
# 数据保存函数
# =============================================================================

def save_subject_data(output_dir, sub, sessions_data, data_type='roi'):
    """
    保存单个被试的数据
    
    Args:
        output_dir: 输出目录
        sub: 被试ID
        sessions_data: {ses: timeseries} 字典
        data_type: 'roi' 或 'vertex'
    """
    if not sessions_data:
        return
    
    # 合并所有session
    all_sessions = sorted(sessions_data.keys())
    ts_list = [sessions_data[ses] for ses in all_sessions]
    concat_ts = np.vstack(ts_list)
    
    # 保存为npz格式 (每个被试一个文件)
    np.savez(os.path.join(output_dir, f'{sub}.npz'),
             timeseries=concat_ts,
             sessions=np.array(all_sessions),
             n_timepoints_per_session=np.array([ts.shape[0] for ts in ts_list]))
    
    # 同时保存mat格式 (兼容性)
    savemat(os.path.join(output_dir, f'{sub}.mat'), {
        'timeseries': concat_ts,
        'sessions': np.array(all_sessions, dtype=object),
        'n_timepoints_per_session': np.array([ts.shape[0] for ts in ts_list])
    })


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract ROI timeseries using individualized parcellation')
    parser.add_argument('--bold_dir', required=True,
                        help='Path to postproc/BOLD directory')
    parser.add_argument('--atlas_dir', required=True,
                        help='Path to directory containing atlas files (Schaefer and Kong annot files, mapping CSV)')
    parser.add_argument('--ind_parc_dir', required=True,
                        help='Path to MSHBM output directory (containing sub-* folders)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory')
    parser.add_argument('--desc', default='regression',
                        help='Which preprocessing to use: bandpass or regression')
    parser.add_argument('--skip_vertex', action='store_true',
                        help='Skip vertex level extraction (saves time and space)')
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS,
                        help=f'Number of parallel jobs (default: {DEFAULT_N_JOBS})')
    args = parser.parse_args()
    
    # 构建文件路径
    schaefer_lh_path = os.path.join(args.atlas_dir, SCHAEFER_ANNOT_LH)
    schaefer_rh_path = os.path.join(args.atlas_dir, SCHAEFER_ANNOT_RH)
    kong_lh_path = os.path.join(args.atlas_dir, KONG_ANNOT_LH)
    kong_rh_path = os.path.join(args.atlas_dir, KONG_ANNOT_RH)
    mapping_path = os.path.join(args.atlas_dir, MAPPING_FILE)
    
    # 检查文件
    required_files = [
        (schaefer_lh_path, SCHAEFER_ANNOT_LH),
        (schaefer_rh_path, SCHAEFER_ANNOT_RH),
        (kong_lh_path, KONG_ANNOT_LH),
        (kong_rh_path, KONG_ANNOT_RH),
        (mapping_path, MAPPING_FILE)
    ]
    
    missing = [name for path, name in required_files if not os.path.exists(path)]
    if missing:
        print("ERROR: Missing files:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nPlease ensure all files are in: {args.atlas_dir}")
        return
    
    # 创建输出目录
    dirs = {
        'schaefer_roi': os.path.join(args.output_dir, 'Schaefer', 'ROI'),
        'schaefer_vert': os.path.join(args.output_dir, 'Schaefer', 'Vox_Vert'),
        'kong_roi': os.path.join(args.output_dir, 'Kong', 'ROI'),
        'kong_vert': os.path.join(args.output_dir, 'Kong', 'Vox_Vert'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    print("=" * 70)
    print("ROI Timeseries Extraction v2 - Individualized Parcellation")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载Atlas文件
    # =========================================================================
    print("\n[1/5] Loading atlas files...")
    
    # Schaefer 2018
    print(f"  Loading Schaefer 2018 annot files...")
    _, _, schaefer_lh_names = load_annot(schaefer_lh_path)
    _, _, schaefer_rh_names = load_annot(schaefer_rh_path)
    
    # Kong 2022
    print(f"  Loading Kong 2022 annot files...")
    _, _, kong_lh_names = load_annot(kong_lh_path)
    _, _, kong_rh_names = load_annot(kong_rh_path)
    
    # Mapping
    print(f"  Loading mapping file...")
    mapping_df = pd.read_csv(mapping_path)
    
    # =========================================================================
    # 2. 构建DMN/ECN定义
    # =========================================================================
    print("\n[2/5] Building DMN/ECN definitions...")
    
    # Schaefer到Kong的映射
    schaefer_name_to_kong_label = build_schaefer_to_kong_mapping(
        mapping_df, schaefer_lh_names, schaefer_rh_names
    )
    
    # Schaefer定义的DMN/ECN
    schaefer_info = get_schaefer_dmn_ecn_labels(
        schaefer_lh_names, schaefer_rh_names, schaefer_name_to_kong_label
    )
    s_lh_dmn, s_lh_ecn, s_rh_dmn, s_rh_ecn, s_label_order, s_roi_names = schaefer_info
    
    n_schaefer_dmn = len(s_lh_dmn) + len(s_rh_dmn)
    n_schaefer_ecn = len(s_lh_ecn) + len(s_rh_ecn)
    print(f"  Schaefer definition: {n_schaefer_dmn} DMN + {n_schaefer_ecn} ECN = {n_schaefer_dmn + n_schaefer_ecn} parcels")
    
    # Kong定义的DMN/ECN
    kong_info = get_kong_dmn_ecn_labels(kong_lh_names, kong_rh_names)
    k_lh_dmn, k_lh_ecn, k_rh_dmn, k_rh_ecn, k_label_order, k_roi_names = kong_info
    
    n_kong_dmn = len(k_lh_dmn) + len(k_rh_dmn)
    n_kong_ecn = len(k_lh_ecn) + len(k_rh_ecn)
    print(f"  Kong definition: {n_kong_dmn} DMN + {n_kong_ecn} ECN = {n_kong_dmn + n_kong_ecn} parcels")
    
    # =========================================================================
    # 3. 保存ROI元数据
    # =========================================================================
    print("\n[3/5] Saving ROI metadata...")
    
    roi_info = {
        'schaefer_definition': {
            'n_dmn': n_schaefer_dmn,
            'n_ecn': n_schaefer_ecn,
            'n_total': n_schaefer_dmn + n_schaefer_ecn,
            'lh_dmn_labels': s_lh_dmn,
            'lh_ecn_labels': s_lh_ecn,
            'rh_dmn_labels': s_rh_dmn,
            'rh_ecn_labels': s_rh_ecn,
            'label_order': s_label_order,
            'roi_names': s_roi_names,
            'network_order': ['DMN'] * n_schaefer_dmn + ['ECN'] * n_schaefer_ecn,
        },
        'kong_definition': {
            'n_dmn': n_kong_dmn,
            'n_ecn': n_kong_ecn,
            'n_total': n_kong_dmn + n_kong_ecn,
            'lh_dmn_labels': k_lh_dmn,
            'lh_ecn_labels': k_lh_ecn,
            'rh_dmn_labels': k_rh_dmn,
            'rh_ecn_labels': k_rh_ecn,
            'label_order': k_label_order,
            'roi_names': k_roi_names,
            'network_order': ['DMN'] * n_kong_dmn + ['ECN'] * n_kong_ecn,
        },
        'description': {
            'DMN': 'Default Mode Network (parcels with "Default" in name)',
            'ECN': 'Executive Control Network (parcels with "Cont" in name)',
            'note': 'Same parcel boundaries but different network assignments between Schaefer and Kong'
        }
    }
    
    with open(os.path.join(args.output_dir, 'roi_info.json'), 'w') as f:
        json.dump(convert_to_native_types(roi_info), f, indent=2)
    print(f"  Saved: {os.path.join(args.output_dir, 'roi_info.json')}")
    
    # =========================================================================
    # 4. 获取被试列表
    # =========================================================================
    print("\n[4/5] Scanning subjects...")
    
    # 从BOLD目录获取被试列表
    bold_subjects = sorted([d for d in os.listdir(args.bold_dir)
                           if d.startswith('sub-') and os.path.isdir(os.path.join(args.bold_dir, d))])
    
    # 检查哪些被试有个体化分割结果
    valid_subjects = []
    for sub in bold_subjects:
        # 从sub-001提取数字部分
        sub_num = sub.replace('sub-', '').lstrip('0')
        if not sub_num:
            sub_num = '0'
        
        # 检查个体化分割文件
        ind_parc_path = os.path.join(args.ind_parc_dir, 
            IND_PARC_PATTERN.format(sub=sub_num.zfill(3)))
        
        # 也尝试其他可能的格式
        alt_patterns = [
            os.path.join(args.ind_parc_dir, f'sub-{sub_num.zfill(3)}', 
                        'ind_parcellation_gMSHBM', 'test_set', '6_sess', 'beta5',
                        'Ind_parcellation_MSHBM_sub1_w50_MRF10_beta5.mat'),
            os.path.join(args.ind_parc_dir, sub,
                        'ind_parcellation_gMSHBM', 'test_set', '6_sess', 'beta5',
                        'Ind_parcellation_MSHBM_sub1_w50_MRF10_beta5.mat'),
        ]
        
        found = False
        for pattern in [ind_parc_path] + alt_patterns:
            if os.path.exists(pattern):
                valid_subjects.append((sub, pattern))
                found = True
                break
        
        if not found:
            print(f"  Warning: No individualized parcellation found for {sub}")
    
    print(f"  Found {len(bold_subjects)} subjects in BOLD dir")
    print(f"  Found {len(valid_subjects)} subjects with individualized parcellation")
    
    # =========================================================================
    # 5. 提取时间序列
    # =========================================================================
    print(f"\n[5/5] Extracting timeseries (parallel with {args.n_jobs} workers)...")
    
    # 收集所有任务
    all_tasks = []
    for sub, ind_parc_path in valid_subjects:
        sub_dir = os.path.join(args.bold_dir, sub)
        sessions = sorted([d for d in os.listdir(sub_dir)
                          if d.startswith('ses-') and os.path.isdir(os.path.join(sub_dir, d))])
        for ses in sessions:
            all_tasks.append((sub, ses, ind_parc_path))
    
    print(f"  Total tasks: {len(all_tasks)} (subjects × sessions)")
    
    # 存储结果
    schaefer_roi_data = {}
    schaefer_vert_data = {}
    kong_roi_data = {}
    kong_vert_data = {}
    
    # 记录顶点信息 (只保存一次)
    schaefer_vert_info = None
    kong_vert_info = None
    
    completed = 0
    lock = threading.Lock()
    
    # 缓存个体化分割
    ind_parc_cache = {}
    cache_lock = threading.Lock()
    
    def process_task(task):
        nonlocal schaefer_vert_info, kong_vert_info
        sub, ses, ind_parc_path = task
        
        # 加载个体化分割 (带缓存)
        with cache_lock:
            if ind_parc_path not in ind_parc_cache:
                ind_lh, ind_rh = load_individualized_parcellation(ind_parc_path)
                ind_parc_cache[ind_parc_path] = (ind_lh, ind_rh)
            else:
                ind_lh, ind_rh = ind_parc_cache[ind_parc_path]
        
        # 处理session
        results = process_session(
            args.bold_dir, sub, ses, ind_lh, ind_rh,
            schaefer_info, kong_info, desc=args.desc
        )
        
        return sub, ses, results
    
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_task, task): task for task in all_tasks}
        
        for future in as_completed(futures):
            sub, ses, results = future.result()
            
            with lock:
                completed += 1
                
                if results is not None:
                    # Schaefer结果
                    if sub not in schaefer_roi_data:
                        schaefer_roi_data[sub] = {}
                        schaefer_vert_data[sub] = {}
                    schaefer_roi_data[sub][ses] = results['schaefer']['roi']
                    if not args.skip_vertex:
                        schaefer_vert_data[sub][ses] = results['schaefer']['vertex']
                        if schaefer_vert_info is None:
                            schaefer_vert_info = results['schaefer']['vertex_info']
                    
                    # Kong结果
                    if sub not in kong_roi_data:
                        kong_roi_data[sub] = {}
                        kong_vert_data[sub] = {}
                    kong_roi_data[sub][ses] = results['kong']['roi']
                    if not args.skip_vertex:
                        kong_vert_data[sub][ses] = results['kong']['vertex']
                        if kong_vert_info is None:
                            kong_vert_info = results['kong']['vertex_info']
                    
                    status = f"ROI:S{results['schaefer']['roi'].shape[1]}/K{results['kong']['roi'].shape[1]}"
                else:
                    status = "SKIP"
                
                print(f"\r  Progress: {completed}/{len(all_tasks)} | {sub} {ses} ({status})          ", end='', flush=True)
    
    print(f"\n  Completed {completed} tasks")
    
    # 保存数据
    print("\n  Saving Schaefer ROI data...")
    for sub in schaefer_roi_data:
        save_subject_data(dirs['schaefer_roi'], sub, schaefer_roi_data[sub], 'roi')
    print(f"    Saved {len(schaefer_roi_data)} subjects")
    
    if not args.skip_vertex:
        print("  Saving Schaefer vertex data...")
        for sub in schaefer_vert_data:
            save_subject_data(dirs['schaefer_vert'], sub, schaefer_vert_data[sub], 'vertex')
        if schaefer_vert_info:
            with open(os.path.join(dirs['schaefer_vert'], 'vertex_info.json'), 'w') as f:
                json.dump(convert_to_native_types(schaefer_vert_info), f)
        print(f"    Saved {len(schaefer_vert_data)} subjects")
    
    print("  Saving Kong ROI data...")
    for sub in kong_roi_data:
        save_subject_data(dirs['kong_roi'], sub, kong_roi_data[sub], 'roi')
    print(f"    Saved {len(kong_roi_data)} subjects")
    
    if not args.skip_vertex:
        print("  Saving Kong vertex data...")
        for sub in kong_vert_data:
            save_subject_data(dirs['kong_vert'], sub, kong_vert_data[sub], 'vertex')
        if kong_vert_info:
            with open(os.path.join(dirs['kong_vert'], 'vertex_info.json'), 'w') as f:
                json.dump(convert_to_native_types(kong_vert_info), f)
        print(f"    Saved {len(kong_vert_data)} subjects")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nSchaefer definition (Networks defined by Schaefer 2018):")
    print(f"  ROI level: {n_schaefer_dmn + n_schaefer_ecn} parcels (DMN: {n_schaefer_dmn}, ECN: {n_schaefer_ecn})")
    print(f"\nKong definition (Networks defined by Kong 2022):")
    print(f"  ROI level: {n_kong_dmn + n_kong_ecn} parcels (DMN: {n_kong_dmn}, ECN: {n_kong_ecn})")
    print(f"\nTotal subjects processed: {len(schaefer_roi_data)}")
    print("\nDirectory structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── Schaefer/")
    print(f"  │   ├── ROI/           # {n_schaefer_dmn + n_schaefer_ecn} ROIs × {len(schaefer_roi_data)} subjects")
    print(f"  │   └── Vox_Vert/      # vertex-level × {len(schaefer_vert_data)} subjects")
    print(f"  ├── Kong/")
    print(f"  │   ├── ROI/           # {n_kong_dmn + n_kong_ecn} ROIs × {len(kong_roi_data)} subjects")
    print(f"  │   └── Vox_Vert/      # vertex-level × {len(kong_vert_data)} subjects")
    print(f"  └── roi_info.json")


if __name__ == '__main__':
    main()
