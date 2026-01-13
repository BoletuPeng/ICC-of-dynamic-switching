# ICC of Dynamic Switching: Replication and Reliability Study

本仓库包含对 Chen et al. (2025) *Communications Biology* 论文的复现及其重测信度（ICC）验证分析。原研究发现默认模式网络(DMN)与执行控制网络(ECN)之间的动态切换频率可以预测创造力。

---

## 目录

- [研究背景](#研究背景)
- [数据概述](#数据概述)
- [预处理流程](#预处理流程)
- [代码架构](#代码架构)
- [分析流程](#分析流程)
  - [组级别分析](#1-组级别分析group-level)
  - [个体化分析](#2-个体化分析individualized)
- [结果汇总](#结果汇总)
- [目录结构](#目录结构)
- [使用指南](#使用指南)
- [致谢与引用](#致谢与引用)

---

## 研究背景

Chen et al. (2025) 提出了一种基于动态功能连接的创造力预测方法，核心思想是：

1. 使用**乘法时间导数(MTD)**方法计算DMN与ECN之间的时间分辨功能连接
2. 对每个时间窗口应用**Louvain社区检测算法**，识别网络的分离/整合状态
3. 使用**K-means聚类**将时间点分为两种状态
4. 计算**状态切换频率**作为动态指标

本项目旨在：
- 复现原研究的分析管线
- 使用多session设计验证该指标的**重测信度(ICC)**
- 探索不同分析参数和脑区定义对结果的影响

---

## 数据概述

### 被试信息

| 指标 | 数值 |
|------|------|
| 影像被试数 | 40 |
| 行为数据样本 | 112 |
| Sessions数 | 6 sessions/被试 |
| 年龄范围 | 18-24岁 (M = 18.7) |
| 性别比例 | 女性 77.7%, 男性 22.3% |
| 利手 | 右利手为主 |

### 扫描参数

- **时间点**: 242 TRs/session
- **有效时间窗**: 223个（跳过前5 TR + 滑动窗口修剪）
- **ROIs**: 106个脑区（DMN 58 + ECN 48，基于Schaefer 300 parcels × 17 networks）

### 行为测量

| 变量 | 量表 | 分数范围 |
|------|------|----------|
| 大五人格 | NEO-FFI | N(26-33), E(34-50), O(35-51), A(30-42), C(41-56) |
| 创造成就 | CAQ | 原始分3-19 |
| 创造行为 | CBQ | 总分14-24 |
| 智力 | 瑞文推理 | 23-28 |

---

## 预处理流程

### 工具

- **DeepPrep 25.0** (Docker容器)

### 预处理步骤

1. **结构像处理**: FreeSurfer重建
2. **功能像处理**:
   - 头动校正 (MCFLIRT)
   - 配准到T1w空间
   - 投影到fsaverage6表面空间
   - 配准到MNI152NLin6Asym体积空间

### 后处理步骤

| 参数 | 设置 |
|------|------|
| 带通滤波 | **无** |
| 空间平滑 | **无** |
| 信号回归 | aCompCor 6成分 + 24P (6运动参数 + 导数 + 二次项) |

### 运动排除

基于运动参数分析，22个session被标记为高运动并可选排除：
- sub-003 ses-05, sub-007 ses-03/04, sub-008 ses-03, sub-009 ses-05 等

---

## 代码架构

### 从MATLAB到Python的翻译

原始OSF代码为MATLAB实现，本项目将其翻译为Python并进行了性能优化：

| OSF原始文件 | Python实现 | 功能 |
|-------------|------------|------|
| `coupling.m` | `coupling_optimized.py` | MTD时间分辨连接 |
| `modularity_louvain_und_sign.m` | `modularity_louvain_und_sign_optimized.py` | Louvain社区检测 |
| - | `agreement_weighted_optimized.py` | 加权协议矩阵 |
| - | `consensus_und_optimized.py` | 共识聚类 |
| - | `participation_coef_sign_optimized.py` | 参与系数 |
| - | `module_degree_zscore_optimized.py` | 模块内度数z-score |

### 性能优化

- **Numba JIT编译**: 核心循环使用`@njit`装饰器加速
- **并行处理**: 使用`joblib`进行被试级别的并行计算
- **向量化计算**: 利用NumPy进行矩阵运算

### 主要脚本

```
分析脚本脚本/
├── extract_roi_timeseries_v2.py              # ROI时间序列提取（组atlas）
├── extract_roi_timeseries_v2_individualized.py # ROI时间序列提取（个体化）
├── run_dynamic_analysis_by_session_parallel_v2.4.py          # 主分析（组atlas）
├── run_dynamic_analysis_by_session_parallel_v2.4_individualized.py  # 主分析（个体化）
├── prepare_behavior_data.py                  # 行为数据整合
└── [优化后的BCT模块...]
```

---

## 分析流程

### 1. 组级别分析（Group-level）

使用Schaefer 2018组级别atlas进行的分析。

#### 空间配置

| 空间 | 描述 | ROI数量 |
|------|------|---------|
| fsaverage6 | 表面空间 | 106 (58 DMN + 48 ECN) |
| MNI | 体积空间 (2mm) | 106 (58 DMN + 48 ECN) |

#### 分析参数

```
Louvain迭代次数: 500
共识聚类阈值(tau): 0.1
共识聚类重复(reps): 500
K-means n_init: 100
滑动窗口大小: 14 TRs
跳过帧数: 5
```

#### 分析变体

每个空间下运行4种变体：
1. 无Motion排除 + 无tSNR过滤
2. 有Motion排除 + 无tSNR过滤
3. 无Motion排除 + 有tSNR过滤 (±2.0SD)
4. 有Motion排除 + 有tSNR过滤 (±2.0SD)

#### 组级别结果汇总

##### fsaverage6空间

| 变体 | Switch Count (M±SD) | ICC(2,1) | Switch-Creativity r | p |
|------|---------------------|----------|---------------------|---|
| noMotion_notSNR | 43.7 ± 4.2 | 0.023 | -0.184 | 0.262 |
| yesMotion_notSNR | 43.7 ± 4.2 | -0.017 | -0.143 | 0.384 |
| noMotion_yestSNR | 44.3 ± 5.2 | 0.070 | 0.031 | 0.850 |
| yesMotion_yestSNR | 44.2 ± 5.1 | -0.010 | 0.079 | 0.635 |

##### MNI空间

| 变体 | Switch Count (M±SD) | ICC(2,1) | Switch-Creativity r | p |
|------|---------------------|----------|---------------------|---|
| noMotion_notSNR | 46.4 ± 3.3 | -0.039 | 0.155 | 0.347 |
| yesMotion_notSNR | 46.3 ± 3.5 | -0.033 | 0.211 | 0.198 |
| noMotion_yestSNR | 46.4 ± 3.6 | 0.006 | 0.024 | 0.883 |
| yesMotion_yestSNR | 46.3 ± 3.9 | 0.041 | 0.056 | 0.734 |

---

### 2. 个体化分析（Individualized）

使用MSHBM 2022进行个体化脑区分割，在fsaverage6空间进行。

#### 网络定义

| Atlas | DMN ROIs | ECN ROIs | Total |
|-------|----------|----------|-------|
| Schaefer 2018 | 58 | 48 | 106 |
| Kong 2022 | 59 | 54 | 113 |

#### 个体化分析结果汇总

##### Schaefer定义

| 变体 | Switch Count (M±SD) | ICC(2,1) | Switch-Creativity r | p |
|------|---------------------|----------|---------------------|---|
| noMotion_notSNR | 44.3 ± 3.1 | -0.081 | -0.000 | 0.999 |
| yesMotion_notSNR | 44.4 ± 3.2 | -0.087 | 0.064 | 0.701 |
| noMotion_yestSNR | 44.3 ± 3.7 | -0.023 | 0.040 | 0.808 |
| yesMotion_yestSNR | 44.4 ± 3.9 | -0.045 | 0.036 | 0.828 |

##### Kong定义

| 变体 | Switch Count (M±SD) | ICC(2,1) | Switch-Creativity r | p |
|------|---------------------|----------|---------------------|---|
| noMotion_notSNR | 43.3 ± 4.3 | -0.007 | 0.074 | 0.654 |
| yesMotion_notSNR | 43.2 ± 4.4 | -0.038 | 0.128 | 0.437 |
| noMotion_yestSNR | 43.9 ± 4.7 | -0.000 | 0.059 | 0.722 |
| yesMotion_yestSNR | 44.2 ± 4.9 | -0.010 | 0.007 | 0.968 |

---

## 结果汇总

### ICC可靠性评估

所有16个分析管线的ICC结果：

```
ICC范围: -0.087 ~ 0.070
所有ICC评级: Poor (< 0.5)
```

根据Cicchetti (1994)的ICC解释标准：
- < 0.40: Poor
- 0.40-0.59: Fair
- 0.60-0.74: Good
- 0.75-1.00: Excellent

**关键发现**: 动态切换频率指标在跨session的重测信度极低，ICC值接近0甚至为负，表明该指标可能主要反映测量噪声或session间的随机波动，而非稳定的个体特质。

### 与创造力的相关分析

```
所有相关系数: -0.184 ~ 0.211
所有p值: > 0.05 (未达显著)
```

**关键发现**: 在本样本中，未能复现原研究报告的动态切换频率与创造力之间的显著相关。

### 结果解读

1. **低ICC可能的原因**:
   - 动态指标本身的不稳定性
   - 静息态fMRI的内在变异性
   - 样本量限制（虽然有40人×6sessions=240观测）
   - 方法学选择（窗口大小、聚类参数等）

2. **相关不显著的可能原因**:
   - 样本特征差异（年龄、文化背景等）
   - 行为测量与原研究的差异
   - 预处理流程差异

---

## 目录结构

```
ICC-of-dynamic-switching/
│
├── README.md                          # 本文件
│
├── Atlas/                             # 脑区模板文件
│   ├── Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.nii.gz
│   ├── lh.Schaefer2018_300Parcels_17Networks_order.annot
│   ├── rh.Schaefer2018_300Parcels_17Networks_order.annot
│   ├── lh.Schaefer2018_300Parcels_Kong2022_17Networks_order.annot
│   ├── rh.Schaefer2018_300Parcels_Kong2022_17Networks_order.annot
│   └── Kong2022_to_Schaefer2018_mapping.csv
│
├── OSF原始代码/                        # 原始MATLAB参考代码
│   ├── README.txt                     # 原作者说明
│   ├── rsfc_DynamicIndex.m            # 静息态分析主脚本
│   ├── tsfc_DynamicIndex.m            # 任务态分析主脚本
│   ├── dystats_task.m                 # 任务态统计
│   ├── Statistic Analysis R Code.R    # R统计脚本
│   ├── dytool/                        # 依赖函数库
│   │   ├── coupling.m
│   │   ├── modularity_louvain_und_sign.m
│   │   ├── kmeans.m
│   │   └── gretna_fishertrans.m
│   └── data/                          # 示例数据
│
├── 分析脚本脚本/                        # Python分析脚本
│   ├── extract_roi_timeseries_v2.py
│   ├── extract_roi_timeseries_v2_individualized.py
│   ├── run_dynamic_analysis_by_session_parallel_v2.4.py
│   ├── run_dynamic_analysis_by_session_parallel_v2.4_individualized.py
│   ├── prepare_behavior_data.py
│   ├── coupling_optimized.py
│   ├── modularity_louvain_und_sign_optimized.py
│   ├── modularity_und_optimized.py
│   ├── consensus_und_optimized.py
│   ├── agreement_weighted_optimized.py
│   ├── participation_coef_sign_optimized.py
│   ├── module_degree_zscore_optimized.py
│   ├── tSNR_optimized.py
│   ├── step4_optimized.py
│   ├── step5_optimized.py
│   └── final_step_optimized.py
│
└── 分析日志或数据结构说明文档/          # 日志与说明
    ├── v2.4主分析日志.txt              # 组级别分析完整日志
    ├── v2.4个体化主分析日志.txt        # 个体化分析完整日志
    ├── v2提取日志.txt                  # ROI提取日志
    ├── v2个体化提取日志.txt
    ├── 三次筛选.txt                    # DeepPrep输出结构说明
    ├── 行为数据.txt                    # 行为数据说明
    ├── 预处理指令.txt                  # DeepPrep命令
    └── 预处理指令_后处理.txt           # 后处理配置
```

---

## 使用指南

### 环境要求

```bash
Python >= 3.8
numpy
scipy
nibabel
numba
joblib
pandas
scikit-learn
tqdm
```

### 运行组级别分析

```bash
python run_dynamic_analysis_by_session_parallel_v2.4.py \
    --roi_dir "/path/to/roi_timeseries" \
    --behavior_file "/path/to/behavior_data.csv" \
    --methods_dir "/path/to/分析脚本脚本" \
    --output_dir "/path/to/output" \
    --level ROI \
    --Louv 500 \
    --LouvTau 0.1 \
    --LouvReps 500 \
    --Kmns 100 \
    --tSNR --tSNR_threshold 2.0 \
    --Motion --motion_file "/path/to/excluded_sessions.txt" \
    --n_jobs 23
```

### 运行个体化分析

```bash
python run_dynamic_analysis_by_session_parallel_v2.4_individualized.py \
    --roi_dir "/path/to/individualized_timeseries" \
    --behavior_file "/path/to/behavior_data.csv" \
    --methods_dir "/path/to/分析脚本脚本" \
    --output_dir "/path/to/output" \
    --level ROI \
    --Louv 500 \
    --LouvTau 0.1 \
    --LouvReps 500 \
    --Kmns 100 \
    --tSNR --tSNR_threshold 2.0 \
    --Motion --motion_file "/path/to/excluded_sessions.txt" \
    --n_jobs 23
```

### 输出文件

每个分析管线输出：
- `dynamic_results_by_session.csv`: 按session的详细结果
- `dynamic_results_subject_mean.csv`: 被试均值
- `statistics.json`: 统计分析结果
- `Trady_results_mtd.mat`: MATLAB兼容格式

---

## 致谢与引用

### 原始研究

```
Chen, Q., et al. (2025). Dynamic switching between brain networks
predicts creative ability across cultures. Communications Biology.
```

### 方法学参考

```
Shine, J. M., et al. (2015). The dynamics of functional brain networks:
Integrated network states during cognitive task performance. Neuron.

Schaefer, A., et al. (2018). Local-global parcellation of the human
cerebral cortex from intrinsic functional connectivity MRI.
Cerebral Cortex.

Kong, R., et al. (2022). Individual-specific areal-level parcellations
improve functional connectivity prediction of behavior.
Cerebral Cortex.
```

### 工具

- [DeepPrep](https://github.com/pBFSLab/DeepPrep): fMRI预处理
- [Brain Connectivity Toolbox](https://sites.google.com/site/babornik/): 网络分析算法
- [Numba](https://numba.pydata.org/): Python JIT编译

---

## 许可证

本项目仅供学术研究使用。

---

*最后更新: 2026年1月*
