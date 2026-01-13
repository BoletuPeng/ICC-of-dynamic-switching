#!/usr/bin/env python3
"""
行为数据整合脚本
将多个Excel文件中的行为学数据整合为统一格式

使用方法:
    python prepare_behavior_data.py --behavior_dir D:/D/Back/fMRI/deepprep_output/Behavior \
                                    --output_dir /path/to/output

输入文件:
    - Demographic_Name_index_GenderMale1Female2_Age_HandRight2Left1.xlsx
    - BigFive_Name_N_E_O_A_C.xlsx
    - CAchievQ_Name_Raw_Log.xlsx
    - CBQ_Name_Total.xlsx
    - Raven_Name_Score.xlsx
    - subject_id_name_mapping.txt

输出:
    - behavior_data.csv: 整合后的行为数据，sub_id格式
"""

import os
import argparse
import pandas as pd
import numpy as np


def load_subject_mapping(mapping_file):
    """
    加载被试ID与姓名的映射关系
    
    支持多种格式:
    - ('001','李雨恒')
    - 001,李雨恒
    - 001\t李雨恒
    
    Returns:
        name_to_id: dict, 姓名 -> sub-ID
    """
    name_to_id = {}
    
    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig']
    content = None
    used_encoding = None
    
    for enc in encodings:
        try:
            with open(mapping_file, 'r', encoding=enc) as f:
                content = f.read()
            # 检查是否有乱码（乱码通常包含很多非常见字符）
            if '?' not in content[:100] and '锟' not in content[:100]:
                used_encoding = enc
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if content is None:
        print(f"  ERROR: Cannot read file with any encoding")
        return name_to_id
    
    print(f"  File encoding detected: {used_encoding}")
    # 打印文件内容的前几行用于调试
    print(f"  File preview: {content[:200]}")
    
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            # 尝试多种格式
            sub_num = None
            name = None
            
            # 格式1: ('001','李雨恒') 或 ("001","李雨恒")
            if '(' in line and ')' in line:
                # 移除括号
                inner = line.strip('() \t')
                # 分割 - 尝试多种分隔符
                if "', '" in inner:
                    parts = inner.split("', '")
                elif "','" in inner:
                    parts = inner.split("','")
                elif '", "' in inner:
                    parts = inner.split('", "')
                elif '","' in inner:
                    parts = inner.split('","')
                else:
                    parts = inner.split(',')
                
                if len(parts) >= 2:
                    sub_num = parts[0].strip("'\" ")
                    name = parts[1].strip("'\" )")
            
            # 格式2: 001,李雨恒 或 001\t李雨恒
            elif ',' in line or '\t' in line:
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split(',')
                
                if len(parts) >= 2:
                    sub_num = parts[0].strip()
                    name = parts[1].strip()
            
            if sub_num and name:
                # 转换为BIDS格式: sub-001
                sub_id = f"sub-{sub_num.zfill(3)}"
                name_to_id[name] = sub_id
                
        except Exception as e:
            print(f"  Warning: Cannot parse line: {repr(line)} - {e}")
            continue
    
    print(f"  Loaded {len(name_to_id)} subject mappings")
    if name_to_id:
        # 显示几个例子
        examples = list(name_to_id.items())[:3]
        print(f"  Examples: {examples}")
    
    return name_to_id


def load_demographic(filepath):
    """
    加载人口统计学数据
    列: 姓名, 编号, 性别, 年龄, 利手, 民族
    """
    df = pd.read_excel(filepath)
    
    # 标准化列名
    col_mapping = {
        '姓名': 'name',
        '编号': 'index',
        '性别': 'gender',  # 男1女2
        '年龄': 'age',
        '利手': 'handedness',  # 右2左1
        '民族': 'ethnicity'
    }
    
    # 尝试重命名列
    for old_name, new_name in col_mapping.items():
        for col in df.columns:
            if old_name in str(col):
                df = df.rename(columns={col: new_name})
                break
    
    # 确保有name列
    if 'name' not in df.columns:
        # 尝试第一列作为姓名
        df = df.rename(columns={df.columns[0]: 'name'})
    
    return df[['name', 'gender', 'age']].copy()


def load_bigfive(filepath):
    """
    加载大五人格数据
    列: 姓名, N, E, O, A, C
    """
    df = pd.read_excel(filepath)
    
    # 标准化列名
    if '姓名' in df.columns:
        df = df.rename(columns={'姓名': 'name'})
    elif df.columns[0] != 'name':
        df = df.rename(columns={df.columns[0]: 'name'})
    
    # 确保大五人格列存在
    personality_cols = ['N', 'E', 'O', 'A', 'C']
    cols_to_keep = ['name']
    
    for col in personality_cols:
        if col in df.columns:
            cols_to_keep.append(col)
        else:
            # 尝试找到包含该字母的列
            for c in df.columns:
                if col in str(c).upper():
                    df = df.rename(columns={c: col})
                    cols_to_keep.append(col)
                    break
    
    return df[cols_to_keep].copy()


def load_creativity_achievement(filepath):
    """
    加载创造成就问卷数据
    列: 姓名, 总分, log_总分
    """
    df = pd.read_excel(filepath)
    
    # 标准化列名
    col_mapping = {
        '姓名': 'name',
        '总分': 'CAQ_raw',
        'log': 'CAQ_log'
    }
    
    for old_pattern, new_name in col_mapping.items():
        for col in df.columns:
            if old_pattern in str(col):
                df = df.rename(columns={col: new_name})
                break
    
    if 'name' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'name'})
    
    cols = ['name']
    if 'CAQ_raw' in df.columns:
        cols.append('CAQ_raw')
    if 'CAQ_log' in df.columns:
        cols.append('CAQ_log')
    
    return df[cols].copy()


def load_creativity_behavior(filepath):
    """
    加载创造行为问卷数据
    列: 姓名, 总分
    """
    df = pd.read_excel(filepath)
    
    # 标准化列名
    if '姓名' in df.columns:
        df = df.rename(columns={'姓名': 'name'})
    elif df.columns[0] != 'name':
        df = df.rename(columns={df.columns[0]: 'name'})
    
    # 找到总分列
    for col in df.columns:
        if '总分' in str(col) or 'Total' in str(col) or 'total' in str(col):
            df = df.rename(columns={col: 'CBQ'})
            break
    
    if 'CBQ' not in df.columns and len(df.columns) > 1:
        df = df.rename(columns={df.columns[1]: 'CBQ'})
    
    return df[['name', 'CBQ']].copy()


def load_raven(filepath):
    """
    加载瑞文推理测试数据
    列: 姓名, 总分
    """
    df = pd.read_excel(filepath)
    
    # 标准化列名
    if '姓名' in df.columns:
        df = df.rename(columns={'姓名': 'name'})
    elif df.columns[0] != 'name':
        df = df.rename(columns={df.columns[0]: 'name'})
    
    # 找到分数列
    for col in df.columns:
        if '总分' in str(col) or 'Score' in str(col) or 'score' in str(col):
            df = df.rename(columns={col: 'IQ'})
            break
    
    if 'IQ' not in df.columns and len(df.columns) > 1:
        df = df.rename(columns={df.columns[1]: 'IQ'})
    
    return df[['name', 'IQ']].copy()


def main():
    parser = argparse.ArgumentParser(description='Prepare behavior data')
    parser.add_argument('--behavior_dir', required=True, 
                       help='Directory containing behavior data files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Preparing Behavior Data")
    print("="*60)
    
    # === 1. 加载被试ID映射 ===
    mapping_file = os.path.join(args.behavior_dir, 'subject_id_name_mapping.txt')
    if os.path.exists(mapping_file):
        name_to_id = load_subject_mapping(mapping_file)
    else:
        print(f"Warning: {mapping_file} not found!")
        print("Will use index-based subject IDs")
        name_to_id = {}
    
    # === 2. 加载各个数据文件 ===
    data_frames = []
    
    # 人口统计学
    demo_file = os.path.join(args.behavior_dir, 
        'Demographic_Name_index_GenderMale1Female2_Age_HandRight2Left1.xlsx')
    if os.path.exists(demo_file):
        print(f"\nLoading demographic data...")
        df_demo = load_demographic(demo_file)
        print(f"  Found {len(df_demo)} records")
        data_frames.append(('demographic', df_demo))
    
    # 大五人格
    bigfive_file = os.path.join(args.behavior_dir, 'BigFive_Name_N_E_O_A_C.xlsx')
    if os.path.exists(bigfive_file):
        print(f"\nLoading Big Five personality data...")
        df_bigfive = load_bigfive(bigfive_file)
        print(f"  Found {len(df_bigfive)} records")
        data_frames.append(('bigfive', df_bigfive))
    
    # 创造成就问卷
    caq_file = os.path.join(args.behavior_dir, 'CAchievQ_Name_Raw_Log.xlsx')
    if os.path.exists(caq_file):
        print(f"\nLoading Creative Achievement data...")
        df_caq = load_creativity_achievement(caq_file)
        print(f"  Found {len(df_caq)} records")
        data_frames.append(('CAQ', df_caq))
    
    # 创造行为问卷
    cbq_file = os.path.join(args.behavior_dir, 'CBQ_Name_Total.xlsx')
    if os.path.exists(cbq_file):
        print(f"\nLoading Creative Behavior data...")
        df_cbq = load_creativity_behavior(cbq_file)
        print(f"  Found {len(df_cbq)} records")
        data_frames.append(('CBQ', df_cbq))
    
    # 瑞文推理
    raven_file = os.path.join(args.behavior_dir, 'Raven_Name_Score.xlsx')
    if os.path.exists(raven_file):
        print(f"\nLoading Raven IQ data...")
        df_raven = load_raven(raven_file)
        print(f"  Found {len(df_raven)} records")
        data_frames.append(('Raven', df_raven))
    
    # === 3. 合并数据 ===
    print("\n" + "="*60)
    print("Merging data...")
    
    if not data_frames:
        print("Error: No data files found!")
        return
    
    # 从第一个数据框开始
    merged_df = data_frames[0][1].copy()
    
    for name, df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on='name', how='outer')
        print(f"  After merging {name}: {len(merged_df)} records")
    
    # === 4. 添加sub_id ===
    print("\nMapping names to subject IDs...")
    
    if name_to_id:
        merged_df['sub_id'] = merged_df['name'].map(name_to_id)
        n_mapped = merged_df['sub_id'].notna().sum()
        print(f"  Mapped {n_mapped} subjects with fMRI data")
        
        # 筛选有fMRI数据的被试
        df_fmri = merged_df[merged_df['sub_id'].notna()].copy()
    else:
        print("  WARNING: No subject mapping loaded!")
        print("  Please check subject_id_name_mapping.txt file format")
        print("  Expected format: one of the following")
        print("    ('001','姓名')")
        print("    001,姓名")
        print("    001\\t姓名")
        print("")
        print("  Creating fallback mapping based on '编号' column if available...")
        
        # 尝试使用编号列
        if 'index' in merged_df.columns:
            merged_df['sub_id'] = merged_df['index'].apply(
                lambda x: f"sub-{int(x):03d}" if pd.notna(x) else None
            )
            df_fmri = merged_df[merged_df['sub_id'].notna()].copy()
            print(f"  Created sub_id from index column for {len(df_fmri)} subjects")
        else:
            # 最后的fallback - 不推荐
            print("  WARNING: Using sequential IDs - this may not match fMRI data!")
            merged_df['sub_id'] = [f"sub-{i:03d}" for i in range(1, len(merged_df)+1)]
            df_fmri = merged_df.copy()
    
    print(f"\nSubjects with fMRI data: {len(df_fmri)}")
    
    # === 6. 创建创造力综合指标 ===
    # 标准化并组合CAQ和CBQ
    # 注意：使用transform而不是直接赋值，以保持索引对齐
    
    if 'CAQ_log' in df_fmri.columns and 'CBQ' in df_fmri.columns:
        # 计算z分数（保持原始索引，缺失值仍为NaN）
        df_fmri['CAQ_z'] = (df_fmri['CAQ_log'] - df_fmri['CAQ_log'].mean()) / df_fmri['CAQ_log'].std()
        df_fmri['CBQ_z'] = (df_fmri['CBQ'] - df_fmri['CBQ'].mean()) / df_fmri['CBQ'].std()
        
        # 综合创造力分数 (取平均，如果其中一个缺失则用另一个)
        df_fmri['crea'] = df_fmri[['CAQ_z', 'CBQ_z']].mean(axis=1)
        print("  Created composite creativity score (crea) from CAQ and CBQ")
        
    elif 'CAQ_log' in df_fmri.columns:
        df_fmri['crea'] = (df_fmri['CAQ_log'] - df_fmri['CAQ_log'].mean()) / df_fmri['CAQ_log'].std()
        print("  Using CAQ_log as creativity score")
        
    elif 'CBQ' in df_fmri.columns:
        df_fmri['crea'] = (df_fmri['CBQ'] - df_fmri['CBQ'].mean()) / df_fmri['CBQ'].std()
        print("  Using CBQ as creativity score")
    
    # === 7. 整理列顺序 ===
    # 重要列放前面
    priority_cols = ['sub_id', 'name', 'gender', 'age', 'crea', 'IQ', 
                     'CAQ_raw', 'CAQ_log', 'CBQ', 'O']  # O=开放性，与创造力相关
    
    ordered_cols = [c for c in priority_cols if c in df_fmri.columns]
    other_cols = [c for c in df_fmri.columns if c not in ordered_cols]
    df_fmri = df_fmri[ordered_cols + other_cols]
    
    # === 8. 保存结果 ===
    output_file = os.path.join(args.output_dir, 'behavior_data.csv')
    df_fmri.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {output_file}")
    
    # 同时保存所有被试的数据（包括无fMRI的）
    output_file_all = os.path.join(args.output_dir, 'behavior_data_all.csv')
    merged_df.to_csv(output_file_all, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_file_all}")
    
    # === 9. 打印汇总 ===
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total subjects with fMRI: {len(df_fmri)}")
    print(f"\nVariables available:")
    for col in df_fmri.columns:
        n_valid = df_fmri[col].notna().sum()
        print(f"  {col}: {n_valid} valid values")
    
    print(f"\nDescriptive statistics for key variables:")
    desc_cols = ['age', 'crea', 'IQ', 'CAQ_log', 'CBQ', 'O']
    desc_cols = [c for c in desc_cols if c in df_fmri.columns]
    print(df_fmri[desc_cols].describe().round(2))
    
    # === 10. 检查与论文方法的匹配 ===
    print("\n" + "="*60)
    print("Notes for Analysis")
    print("="*60)
    print("""
创造力测量:
  - CAQ (Creative Achievement Questionnaire): 创造成就问卷
  - CBQ (Creative Behavior Questionnaire): 创造行为问卷
  - crea: CAQ和CBQ的标准化综合分数
  
与原论文的对应:
  - 原论文使用AUT (Alternate Uses Task)发散思维任务
  - 你的数据使用问卷法测量创造力
  - 两者测量的是创造力的不同方面，但都是有效的创造力指标
  
推荐分析:
  1. 使用 'crea' 作为主要创造力指标进行分析
  2. 也可以分别用CAQ和CBQ进行分析，检验一致性
  3. 使用 'O' (开放性) 作为人格-创造力的补充分析
  4. 使用 'IQ' 作为控制变量或进行对比分析（智力 vs 创造力）
""")


if __name__ == '__main__':
    main()
