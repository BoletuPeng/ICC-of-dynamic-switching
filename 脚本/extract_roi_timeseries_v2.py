#!/usr/bin/env python3
"""
ROI时间序列提取脚本 v2
从DeepPrep输出中提取Schaefer300 DMN+ECN区域的时间序列

支持:
- fsaverage6空间 (表面): ROI级别 + 顶点级别
- MNI空间 (体积): ROI级别 + 体素级别

输出目录结构:
    output_dir/
    ├── fsaverage6/
    │   ├── ROI/           # 106个ROI的平均时间序列
    │   └── Vox_Vert/      # 所有DMN/ECN顶点的时间序列
    ├── MNI/
    │   ├── ROI/           # 106个ROI的平均时间序列
    │   └── Vox_Vert/      # 所有DMN/ECN体素的时间序列
    └── roi_info.json      # ROI元数据

使用方法:
    python extract_roi_timeseries_v2.py \
        --bold_dir /path/to/postproc/BOLD \
        --atlas_dir /path/to/atlas_files \
        --output_dir /path/to/output

依赖:
    pip install nibabel numpy pandas scipy
"""

import os

# =============================================================================
# 模板文件名定义 (在此处修改模板文件名)
# =============================================================================

# fsaverage6 空间的 Schaefer annot 文件名
SCHAEFER_ANNOT_LH = 'lh.Schaefer2018_300Parcels_17Networks_order.annot'
SCHAEFER_ANNOT_RH = 'rh.Schaefer2018_300Parcels_17Networks_order.annot'

# MNI 空间的 Schaefer atlas 文件名
SCHAEFER_MNI_ATLAS = 'Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'

# 默认并行数
DEFAULT_N_JOBS = 8

# =============================================================================
# 导入库
# =============================================================================
import glob
import argparse
import numpy as np
import nibabel as nib
from nibabel.freesurfer.io import read_annot
import pandas as pd
from scipy.io import savemat
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def convert_to_native_types(obj):
    """递归转换numpy类型为Python原生类型，用于JSON序列化"""
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


# =============================================================================
# 通用工具函数
# =============================================================================

def load_schaefer_annot(annot_path):
    """
    加载Schaefer annot文件
    
    Returns:
        labels: 每个顶点的标签 (0表示medial wall)
        ctab: 颜色表
        names: 区域名称列表
    """
    labels, ctab, names = read_annot(annot_path)
    names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in names]
    return labels, ctab, names


def get_dmn_ecn_indices(names):
    """
    从Schaefer 17Networks中筛选DMN和ECN (Control)的parcel索引
    
    Returns:
        dmn_indices: DMN区域的索引列表
        ecn_indices: ECN区域的索引列表  
        selected_names: 选中区域的名称
    """
    dmn_indices = []
    ecn_indices = []
    selected_names = []
    
    for i, name in enumerate(names):
        if name == 'Unknown' or name == '???':
            continue
        if 'Default' in name:
            dmn_indices.append(i)
            selected_names.append(name)
        elif 'Cont' in name:
            ecn_indices.append(i)
            selected_names.append(name)
    
    return dmn_indices, ecn_indices, selected_names


def get_dmn_ecn_parcel_ids_from_mni(atlas_data, parcel_names_map):
    """
    从MNI atlas中获取DMN和ECN的parcel IDs
    
    Args:
        atlas_data: atlas volume数据
        parcel_names_map: {parcel_id: parcel_name} 的映射
        
    Returns:
        dmn_ids: DMN的parcel IDs
        ecn_ids: ECN的parcel IDs
    """
    dmn_ids = []
    ecn_ids = []
    
    for pid, name in parcel_names_map.items():
        if 'Default' in name:
            dmn_ids.append(pid)
        elif 'Cont' in name:
            ecn_ids.append(pid)
    
    return dmn_ids, ecn_ids


# =============================================================================
# fsaverage6 空间处理函数
# =============================================================================

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


def extract_surface_roi_timeseries(bold_data, vertex_labels, parcel_indices):
    """
    从表面BOLD数据中提取ROI级别的平均时间序列
    
    Returns:
        timeseries: (n_timepoints, n_parcels)
    """
    n_timepoints = bold_data.shape[0]
    n_parcels = len(parcel_indices)
    timeseries = np.zeros((n_timepoints, n_parcels))
    
    for i, idx in enumerate(parcel_indices):
        mask = vertex_labels == idx
        if np.sum(mask) > 0:
            timeseries[:, i] = np.mean(bold_data[:, mask], axis=1)
        else:
            timeseries[:, i] = np.nan
            
    return timeseries


def extract_surface_vertex_timeseries(bold_data, vertex_labels, parcel_indices):
    """
    从表面BOLD数据中提取所有选定parcel的顶点级别时间序列
    
    Returns:
        timeseries: (n_timepoints, n_selected_vertices)
        vertex_info: 每个顶点的信息 (原始索引, parcel索引)
    """
    # 创建选定顶点的掩码
    selected_mask = np.zeros(len(vertex_labels), dtype=bool)
    for idx in parcel_indices:
        selected_mask |= (vertex_labels == idx)
    
    # 提取选定顶点的时间序列
    timeseries = bold_data[:, selected_mask]
    
    # 记录每个选定顶点的信息
    selected_indices = np.where(selected_mask)[0]
    vertex_parcel = vertex_labels[selected_mask]
    
    vertex_info = {
        'original_indices': selected_indices.tolist(),
        'parcel_labels': vertex_parcel.tolist()
    }
    
    return timeseries, vertex_info


def process_fsaverage6_session(bold_dir, sub, ses, lh_labels, rh_labels,
                                lh_indices, rh_indices, desc='regression'):
    """
    处理单个被试单个session的fsaverage6数据
    
    Returns:
        roi_ts: ROI级别时间序列 (n_timepoints, n_rois)
        vertex_ts: 顶点级别时间序列 (n_timepoints, n_vertices)
        vertex_info: 顶点元数据
    """
    func_dir = os.path.join(bold_dir, sub, ses, 'func')
    
    # 构建文件路径
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
        return None, None, None
    
    # 加载BOLD数据
    lh_data, rh_data = load_bold_surface(lh_pattern, rh_pattern)
    
    # === ROI级别 ===
    lh_roi_ts = extract_surface_roi_timeseries(lh_data, lh_labels, lh_indices)
    rh_roi_ts = extract_surface_roi_timeseries(rh_data, rh_labels, rh_indices)
    roi_ts = np.hstack([lh_roi_ts, rh_roi_ts])
    
    # === 顶点级别 ===
    lh_vert_ts, lh_vert_info = extract_surface_vertex_timeseries(lh_data, lh_labels, lh_indices)
    rh_vert_ts, rh_vert_info = extract_surface_vertex_timeseries(rh_data, rh_labels, rh_indices)
    vertex_ts = np.hstack([lh_vert_ts, rh_vert_ts])
    
    # 合并顶点信息
    n_lh_vert = lh_vert_ts.shape[1]
    vertex_info = {
        'lh_original_indices': lh_vert_info['original_indices'],
        'lh_parcel_labels': lh_vert_info['parcel_labels'],
        'rh_original_indices': rh_vert_info['original_indices'],
        'rh_parcel_labels': rh_vert_info['parcel_labels'],
        'n_lh_vertices': n_lh_vert,
        'n_rh_vertices': rh_vert_ts.shape[1],
        'hemi_split_index': n_lh_vert  # 前n_lh_vert个是左半球
    }
    
    return roi_ts, vertex_ts, vertex_info


# =============================================================================
# MNI 空间处理函数
# =============================================================================

def load_mni_atlas(atlas_path):
    """
    加载MNI空间的Schaefer atlas
    
    Returns:
        atlas_data: 3D array of parcel labels
        affine: affine transformation matrix
        parcel_ids: unique parcel IDs (excluding 0)
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    affine = atlas_img.affine
    
    parcel_ids = np.unique(atlas_data)
    parcel_ids = parcel_ids[parcel_ids != 0]  # 排除背景
    
    return atlas_data, affine, parcel_ids


def get_schaefer_mni_parcel_mapping(lh_names, rh_names, lh_indices, rh_indices):
    """
    基于fsaverage6的parcel名称，建立MNI atlas parcel ID的映射
    
    Schaefer MNI atlas的parcel ID规则:
    - 1-150: 左半球 (按照fsaverage6 annot中的顺序，跳过Unknown)
    - 151-300: 右半球
    
    Args:
        lh_names, rh_names: 从annot文件读取的parcel名称
        lh_indices, rh_indices: 选定的DMN/ECN parcel在annot中的索引
        
    Returns:
        mni_dmn_ids: MNI atlas中DMN的parcel IDs
        mni_ecn_ids: MNI atlas中ECN的parcel IDs
        parcel_id_to_name: {parcel_id: name}
    """
    # 建立MNI parcel ID到名称的映射
    # 注意: annot中的索引0通常是'Unknown'，对应MNI atlas中的0(背景)
    # MNI atlas的ID从1开始
    
    parcel_id_to_name = {}
    
    # 左半球: annot索引1-150 对应 MNI ID 1-150
    lh_valid_names = [n for n in lh_names if n not in ['Unknown', '???', '']]
    for i, name in enumerate(lh_valid_names):
        parcel_id_to_name[i + 1] = name  # MNI ID从1开始
    
    # 右半球: annot索引1-150 对应 MNI ID 151-300
    rh_valid_names = [n for n in rh_names if n not in ['Unknown', '???', '']]
    for i, name in enumerate(rh_valid_names):
        parcel_id_to_name[i + 151] = name
    
    # 筛选DMN和ECN
    mni_dmn_ids = []
    mni_ecn_ids = []
    mni_selected_names = []
    
    for pid, name in parcel_id_to_name.items():
        if 'Default' in name:
            mni_dmn_ids.append(pid)
            mni_selected_names.append(name)
        elif 'Cont' in name:
            mni_ecn_ids.append(pid)
            mni_selected_names.append(name)
    
    # 排序以保持一致性
    mni_dmn_ids = sorted(mni_dmn_ids)
    mni_ecn_ids = sorted(mni_ecn_ids)
    
    return mni_dmn_ids, mni_ecn_ids, parcel_id_to_name


def extract_mni_roi_timeseries(bold_data, atlas_data, parcel_ids):
    """
    从MNI空间BOLD数据中提取ROI级别的平均时间序列
    
    Args:
        bold_data: (x, y, z, t) 4D BOLD数据
        atlas_data: (x, y, z) 3D atlas标签
        parcel_ids: 要提取的parcel ID列表
        
    Returns:
        timeseries: (n_timepoints, n_parcels)
    """
    n_timepoints = bold_data.shape[3]
    n_parcels = len(parcel_ids)
    timeseries = np.zeros((n_timepoints, n_parcels))
    
    for i, pid in enumerate(parcel_ids):
        mask = atlas_data == pid
        if np.sum(mask) > 0:
            # 提取该parcel内所有体素的平均值
            voxel_ts = bold_data[mask, :]  # (n_voxels, n_timepoints)
            timeseries[:, i] = np.mean(voxel_ts, axis=0)
        else:
            timeseries[:, i] = np.nan
    
    return timeseries


def extract_mni_voxel_timeseries(bold_data, atlas_data, parcel_ids, affine):
    """
    从MNI空间BOLD数据中提取所有选定parcel的体素级别时间序列
    
    Returns:
        timeseries: (n_timepoints, n_selected_voxels)
        voxel_info: 每个体素的信息 (坐标, parcel标签)
    """
    # 创建选定体素的掩码
    selected_mask = np.zeros(atlas_data.shape, dtype=bool)
    for pid in parcel_ids:
        selected_mask |= (atlas_data == pid)
    
    # 获取选定体素的索引
    voxel_indices = np.array(np.where(selected_mask)).T  # (n_voxels, 3)
    
    # 提取时间序列
    timeseries = bold_data[selected_mask, :].T  # (n_timepoints, n_voxels)
    
    # 计算MNI坐标
    # 添加齐次坐标
    voxel_coords_homo = np.hstack([voxel_indices, np.ones((len(voxel_indices), 1))])
    mni_coords = (affine @ voxel_coords_homo.T).T[:, :3]
    
    # 获取每个体素的parcel标签
    voxel_parcel_labels = atlas_data[selected_mask]
    
    voxel_info = {
        'voxel_indices': voxel_indices.tolist(),  # 体素索引 (i, j, k)
        'mni_coords': mni_coords.tolist(),        # MNI坐标 (x, y, z)
        'parcel_labels': voxel_parcel_labels.tolist()
    }
    
    return timeseries, voxel_info


def process_mni_session(bold_dir, sub, ses, atlas_data, affine, 
                        selected_parcel_ids, desc='regression'):
    """
    处理单个被试单个session的MNI空间数据
    
    Returns:
        roi_ts: ROI级别时间序列
        voxel_ts: 体素级别时间序列
        voxel_info: 体素元数据
    """
    func_dir = os.path.join(bold_dir, sub, ses, 'func')
    
    # 构建文件路径
    mni_pattern = os.path.join(func_dir, 
        f'{sub}_{ses}_task-rest_space-MNI152NLin6Asym_res-02_desc-{desc}_bold.nii.gz')
    
    if not os.path.exists(mni_pattern):
        mni_pattern = os.path.join(func_dir,
            f'{sub}_{ses}_task-rest_space-MNI152NLin6Asym_res-02_bold.nii.gz')
    
    if not os.path.exists(mni_pattern):
        return None, None, None
    
    # 加载BOLD数据
    bold_img = nib.load(mni_pattern)
    bold_data = bold_img.get_fdata()
    
    # === ROI级别 ===
    roi_ts = extract_mni_roi_timeseries(bold_data, atlas_data, selected_parcel_ids)
    
    # === 体素级别 ===
    voxel_ts, voxel_info = extract_mni_voxel_timeseries(
        bold_data, atlas_data, selected_parcel_ids, affine)
    
    return roi_ts, voxel_ts, voxel_info


# =============================================================================
# 数据保存函数
# =============================================================================

def save_session_data(output_dir, data_dict, sub_ids, prefix, save_mat=True):
    """
    保存session数据
    
    Args:
        output_dir: 输出目录
        data_dict: {sub: {ses: timeseries}}
        sub_ids: 有效的被试ID列表
        prefix: 文件名前缀
        save_mat: 是否保存.mat格式
    """
    sessions = ['ses-01', 'ses-02', 'ses-03', 'ses-04', 'ses-05', 'ses-06']
    
    # 每个session单独保存
    for ses in sessions:
        ts_list = []
        valid_subs = []
        for sub in sub_ids:
            if sub in data_dict and ses in data_dict[sub]:
                ts_list.append(data_dict[sub][ses])
                valid_subs.append(sub)
        
        if ts_list:
            if save_mat:
                mat_data = {
                    'rsfc': np.array(ts_list, dtype=object),
                    'sub_ids': np.array(valid_subs, dtype=object)
                }
                savemat(os.path.join(output_dir, f'{prefix}_{ses}.mat'), mat_data)
            
            # 也保存npz格式
            np.savez(os.path.join(output_dir, f'{prefix}_{ses}.npz'),
                     rsfc=np.array(ts_list, dtype=object),
                     sub_ids=np.array(valid_subs))
    
    # 合并所有session
    concat_list = []
    concat_subs = []
    for sub in sub_ids:
        if sub in data_dict:
            sessions_data = [data_dict[sub][ses] for ses in sorted(data_dict[sub].keys())]
            if sessions_data:
                concat_ts = np.vstack(sessions_data)
                concat_list.append(concat_ts)
                concat_subs.append(sub)
    
    if concat_list:
        if save_mat:
            mat_data = {
                'rsfc': np.array(concat_list, dtype=object),
                'sub_ids': np.array(concat_subs, dtype=object)
            }
            savemat(os.path.join(output_dir, f'{prefix}_all_sessions.mat'), mat_data)
        
        np.savez(os.path.join(output_dir, f'{prefix}_all_sessions.npz'),
                 rsfc=np.array(concat_list, dtype=object),
                 sub_ids=np.array(concat_subs))
    
    return len(concat_subs)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract ROI timeseries from fsaverage6 and MNI space BOLD data')
    parser.add_argument('--bold_dir', required=True, 
                        help='Path to postproc/BOLD directory')
    parser.add_argument('--atlas_dir', required=True, 
                        help='Path to directory containing Schaefer atlas files (annot + nii.gz)')
    parser.add_argument('--output_dir', required=True, 
                        help='Output directory')
    parser.add_argument('--desc', default='regression', 
                        help='Which preprocessing to use: bandpass or regression')
    parser.add_argument('--skip_voxel', action='store_true',
                        help='Skip voxel/vertex level extraction (saves time and space)')
    parser.add_argument('--n_jobs', type=int, default=DEFAULT_N_JOBS,
                        help=f'Number of parallel jobs (default: {DEFAULT_N_JOBS})')
    args = parser.parse_args()
    
    # 构建模板文件的完整路径
    lh_annot_path = os.path.join(args.atlas_dir, SCHAEFER_ANNOT_LH)
    rh_annot_path = os.path.join(args.atlas_dir, SCHAEFER_ANNOT_RH)
    mni_atlas_path = os.path.join(args.atlas_dir, SCHAEFER_MNI_ATLAS)
    
    # 检查文件是否存在
    missing_files = []
    for fpath, fname in [(lh_annot_path, SCHAEFER_ANNOT_LH), 
                          (rh_annot_path, SCHAEFER_ANNOT_RH),
                          (mni_atlas_path, SCHAEFER_MNI_ATLAS)]:
        if not os.path.exists(fpath):
            missing_files.append(fname)
    
    if missing_files:
        print("ERROR: The following atlas files are missing from the atlas directory:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease ensure all files are in: {args.atlas_dir}")
        return
    
    # 创建输出目录结构
    dirs = {
        'fs_roi': os.path.join(args.output_dir, 'fsaverage6', 'ROI'),
        'fs_vert': os.path.join(args.output_dir, 'fsaverage6', 'Vox_Vert'),
        'mni_roi': os.path.join(args.output_dir, 'MNI', 'ROI'),
        'mni_vox': os.path.join(args.output_dir, 'MNI', 'Vox_Vert'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    print("=" * 70)
    print("ROI Timeseries Extraction v2")
    print("=" * 70)
    
    # =========================================================================
    # 1. 加载fsaverage6 Schaefer模板
    # =========================================================================
    print("\n[1/6] Loading fsaverage6 Schaefer300 17Networks parcellation...")
    print(f"  LH annot: {lh_annot_path}")
    print(f"  RH annot: {rh_annot_path}")
    
    lh_labels, lh_ctab, lh_names = load_schaefer_annot(lh_annot_path)
    rh_labels, rh_ctab, rh_names = load_schaefer_annot(rh_annot_path)
    
    print(f"  LH parcels: {len(lh_names)}, RH parcels: {len(rh_names)}")
    print(f"  LH vertices: {len(lh_labels)}, RH vertices: {len(rh_labels)}")
    
    # 筛选DMN和ECN区域
    print("\n  Selecting DMN and ECN parcels (fsaverage6)...")
    lh_dmn, lh_ecn, lh_selected = get_dmn_ecn_indices(lh_names)
    rh_dmn, rh_ecn, rh_selected = get_dmn_ecn_indices(rh_names)
    
    lh_indices = lh_dmn + lh_ecn
    rh_indices = rh_dmn + rh_ecn
    
    n_dmn_fs = len(lh_dmn) + len(rh_dmn)
    n_ecn_fs = len(lh_ecn) + len(rh_ecn)
    n_total_fs = len(lh_indices) + len(rh_indices)
    
    print(f"    DMN parcels: {n_dmn_fs}")
    print(f"    ECN parcels: {n_ecn_fs}")
    print(f"    Total: {n_total_fs} parcels")
    
    # 计算顶点数量
    n_lh_vert = sum(1 for v in lh_labels if v in lh_indices)
    n_rh_vert = sum(1 for v in rh_labels if v in rh_indices)
    print(f"    Selected vertices: LH={n_lh_vert}, RH={n_rh_vert}, Total={n_lh_vert + n_rh_vert}")
    
    # =========================================================================
    # 2. 加载MNI Schaefer atlas
    # =========================================================================
    print("\n[2/6] Loading MNI Schaefer300 17Networks atlas...")
    print(f"  MNI atlas: {mni_atlas_path}")
    
    atlas_data, affine, all_parcel_ids = load_mni_atlas(mni_atlas_path)
    print(f"  Atlas shape: {atlas_data.shape}")
    print(f"  Total parcels in atlas: {len(all_parcel_ids)}")
    
    # 基于fsaverage6的annot名称建立MNI parcel映射
    mni_dmn_ids, mni_ecn_ids, parcel_id_to_name = get_schaefer_mni_parcel_mapping(
        lh_names, rh_names, lh_indices, rh_indices
    )
    
    mni_selected_ids = mni_dmn_ids + mni_ecn_ids
    mni_selected_names = [parcel_id_to_name.get(pid, f'Parcel_{pid}') for pid in mni_selected_ids]
    
    n_dmn_mni = len(mni_dmn_ids)
    n_ecn_mni = len(mni_ecn_ids)
    n_total_mni = len(mni_selected_ids)
    
    print(f"  Selecting DMN and ECN parcels (MNI)...")
    print(f"    DMN parcels: {n_dmn_mni}")
    print(f"    ECN parcels: {n_ecn_mni}")
    print(f"    Total: {n_total_mni} parcels")
    
    # 计算体素数量
    n_voxels = sum(np.sum(atlas_data == pid) for pid in mni_selected_ids)
    print(f"    Selected voxels: {n_voxels}")
    
    # =========================================================================
    # 3. 保存ROI元数据
    # =========================================================================
    print("\n[3/6] Saving ROI metadata...")
    
    roi_info = {
        'fsaverage6': {
            'lh_parcel_names': [lh_names[i] for i in lh_indices],
            'rh_parcel_names': [rh_names[i] for i in rh_indices],
            'lh_parcel_indices': lh_indices,
            'rh_parcel_indices': rh_indices,
            'n_dmn_lh': len(lh_dmn),
            'n_dmn_rh': len(rh_dmn),
            'n_ecn_lh': len(lh_ecn),
            'n_ecn_rh': len(rh_ecn),
            'n_lh_vertices': n_lh_vert,
            'n_rh_vertices': n_rh_vert,
        },
        'mni': {
            'parcel_ids': mni_selected_ids,
            'parcel_names': mni_selected_names,
            'dmn_parcel_ids': mni_dmn_ids,
            'ecn_parcel_ids': mni_ecn_ids,
            'n_dmn': n_dmn_mni,
            'n_ecn': n_ecn_mni,
            'n_voxels': n_voxels,
        },
        'network_order': ['DMN'] * n_dmn_fs + ['ECN'] * n_ecn_fs,
        'description': 'DMN = Default Mode Network, ECN = Executive Control Network (Cont in Schaefer)'
    }
    
    with open(os.path.join(args.output_dir, 'roi_info.json'), 'w') as f:
        json.dump(convert_to_native_types(roi_info), f, indent=2)
    print(f"  Saved: {os.path.join(args.output_dir, 'roi_info.json')}")
    
    # =========================================================================
    # 4. 获取被试列表
    # =========================================================================
    print("\n[4/6] Scanning subjects...")
    subjects = sorted([d for d in os.listdir(args.bold_dir) 
                      if d.startswith('sub-') and os.path.isdir(os.path.join(args.bold_dir, d))])
    print(f"  Found {len(subjects)} subjects")
    
    # =========================================================================
    # 5. 提取fsaverage6数据 (并行)
    # =========================================================================
    print(f"\n[5/6] Extracting fsaverage6 timeseries (parallel with {args.n_jobs} workers)...")
    
    fs_roi_data = {}
    fs_vert_data = {}
    fs_vert_info = None
    
    # 构建所有任务列表
    fs_tasks = []
    for sub in subjects:
        sub_dir = os.path.join(args.bold_dir, sub)
        sessions = sorted([d for d in os.listdir(sub_dir) 
                          if d.startswith('ses-') and os.path.isdir(os.path.join(sub_dir, d))])
        for ses in sessions:
            fs_tasks.append((sub, ses))
    
    print(f"  Total tasks: {len(fs_tasks)} (subjects × sessions)")
    
    # 并行处理
    fs_results = {}
    completed = 0
    lock = threading.Lock()
    
    def process_fs_task(task):
        sub, ses = task
        roi_ts, vert_ts, vert_info = process_fsaverage6_session(
            args.bold_dir, sub, ses, lh_labels, rh_labels,
            lh_indices, rh_indices, desc=args.desc
        )
        return sub, ses, roi_ts, vert_ts, vert_info
    
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_fs_task, task): task for task in fs_tasks}
        
        for future in as_completed(futures):
            sub, ses, roi_ts, vert_ts, vert_info = future.result()
            
            with lock:
                completed += 1
                if sub not in fs_roi_data:
                    fs_roi_data[sub] = {}
                    fs_vert_data[sub] = {}
                
                if roi_ts is not None:
                    fs_roi_data[sub][ses] = roi_ts
                    if not args.skip_voxel:
                        fs_vert_data[sub][ses] = vert_ts
                    
                    # 保存顶点信息 (只需要保存一次)
                    if fs_vert_info is None and vert_info is not None:
                        fs_vert_info = vert_info
                
                # 进度显示
                status = f"ROI:{roi_ts.shape[1]}" if roi_ts is not None else "SKIP"
                print(f"\r  Progress: {completed}/{len(fs_tasks)} | {sub} {ses} ({status})          ", end='', flush=True)
    
    print(f"\n  Completed {completed} tasks")
    
    # 保存顶点信息
    if fs_vert_info is not None:
        with open(os.path.join(dirs['fs_vert'], 'vertex_info.json'), 'w') as f:
            json.dump(convert_to_native_types(fs_vert_info), f)
    
    # 保存fsaverage6数据
    print("  Saving fsaverage6 ROI data...")
    n_fs_roi = save_session_data(dirs['fs_roi'], fs_roi_data, subjects, 'rsfc')
    print(f"    Saved {n_fs_roi} subjects")
    
    if not args.skip_voxel:
        print("  Saving fsaverage6 vertex data...")
        n_fs_vert = save_session_data(dirs['fs_vert'], fs_vert_data, subjects, 'rsfc_vertex', save_mat=False)
        print(f"    Saved {n_fs_vert} subjects")
    
    # =========================================================================
    # 6. 提取MNI数据 (并行)
    # =========================================================================
    print(f"\n[6/6] Extracting MNI timeseries (parallel with {args.n_jobs} workers)...")
    
    mni_roi_data = {}
    mni_vox_data = {}
    mni_vox_info = None
    
    # 构建所有任务列表 (复用fs_tasks的结构)
    mni_tasks = fs_tasks  # 相同的subject-session组合
    print(f"  Total tasks: {len(mni_tasks)} (subjects × sessions)")
    
    # 并行处理
    completed = 0
    
    def process_mni_task(task):
        sub, ses = task
        roi_ts, vox_ts, vox_info = process_mni_session(
            args.bold_dir, sub, ses, atlas_data, affine,
            mni_selected_ids, desc=args.desc
        )
        return sub, ses, roi_ts, vox_ts, vox_info
    
    with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(process_mni_task, task): task for task in mni_tasks}
        
        for future in as_completed(futures):
            sub, ses, roi_ts, vox_ts, vox_info = future.result()
            
            with lock:
                completed += 1
                if sub not in mni_roi_data:
                    mni_roi_data[sub] = {}
                    mni_vox_data[sub] = {}
                
                if roi_ts is not None:
                    mni_roi_data[sub][ses] = roi_ts
                    if not args.skip_voxel:
                        mni_vox_data[sub][ses] = vox_ts
                    
                    # 保存体素信息 (只需要保存一次)
                    if mni_vox_info is None and vox_info is not None:
                        mni_vox_info = vox_info
                
                # 进度显示
                status = f"ROI:{roi_ts.shape[1]}" if roi_ts is not None else "SKIP"
                print(f"\r  Progress: {completed}/{len(mni_tasks)} | {sub} {ses} ({status})          ", end='', flush=True)
    
    print(f"\n  Completed {completed} tasks")
    
    # 保存体素信息
    if mni_vox_info is not None:
        np.savez(os.path.join(dirs['mni_vox'], 'voxel_info.npz'),
                 voxel_indices=np.array(mni_vox_info['voxel_indices']),
                 mni_coords=np.array(mni_vox_info['mni_coords']),
                 parcel_labels=np.array(mni_vox_info['parcel_labels']))
    
    # 保存MNI数据
    print("  Saving MNI ROI data...")
    n_mni_roi = save_session_data(dirs['mni_roi'], mni_roi_data, subjects, 'rsfc')
    print(f"    Saved {n_mni_roi} subjects")
    
    if not args.skip_voxel:
        print("  Saving MNI voxel data...")
        n_mni_vox = save_session_data(dirs['mni_vox'], mni_vox_data, subjects, 'rsfc_voxel', save_mat=False)
        print(f"    Saved {n_mni_vox} subjects")
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nfsaverage6 space:")
    print(f"  ROI level: {n_total_fs} parcels (DMN: {n_dmn_fs}, ECN: {n_ecn_fs})")
    print(f"  Vertex level: {n_lh_vert + n_rh_vert} vertices")
    print(f"\nMNI space:")
    print(f"  ROI level: {n_total_mni} parcels (DMN: {n_dmn_mni}, ECN: {n_ecn_mni})")
    print(f"  Voxel level: {n_voxels} voxels")
    print(f"\nTotal subjects processed: {len(subjects)}")
    print("\nDirectory structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── fsaverage6/")
    print(f"  │   ├── ROI/           # {n_total_fs} ROIs × {len(subjects)} subjects")
    print(f"  │   └── Vox_Vert/      # {n_lh_vert + n_rh_vert} vertices × {len(subjects)} subjects")
    print(f"  ├── MNI/")
    print(f"  │   ├── ROI/           # {n_total_mni} ROIs × {len(subjects)} subjects")
    print(f"  │   └── Vox_Vert/      # {n_voxels} voxels × {len(subjects)} subjects")
    print(f"  └── roi_info.json")


if __name__ == '__main__':
    main()
