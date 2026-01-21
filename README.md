# 基于扩散模型的二维材料结构生成与优化

这是一个面向工程实践的 **二维材料原子结构生成与优化框架**。  
项目结合了 **条件扩散模型（DDPM）** 和 **图神经网络（GNN）**，在给定材料性能目标的条件下，生成满足几何与物理约束的二维原子结构，并对候选结构进行打分和排序。

主要面向以下任务：
- 二维材料原子结构生成
- 基于性能指标（ΔG、热稳定性、可合成性）的引导优化
- 候选结构筛选与排序

---

## 一、整体流程概览

整体流程分为三步：

1. **数据加载与预处理**
   - 从 `JVASP-*.json` 文件中读取晶体结构与性能标签
   - 构建原子级全连接图（cutoff 截断）
   - 对不同原子数结构进行 padding + mask

2. **模型训练**
   - 训练一个 **性质预测模型（PropertyPredictor）**
   - 训练一个 **结构扩散去噪模型（DiffusionDenoiser）**

3. **结构生成与引导优化**
   - 使用扩散模型生成新结构
   - 通过已训练的性质预测器进行梯度引导（Guidance）
   - 强制二维几何约束
   - 对生成结果进行排序和可视化

---

## 二、项目目录结构

```text
project/
├── dataset/
│   └── material_dataset.py      # 材料数据集定义 & collate
├── models/
│   ├── diffusion_model.py       # 扩散模型 & 性质预测模型
│   ├── structure_generator.py   # 扩散采样与调度
│   └── optimization.py          # 引导目标函数与排序
├── utils/
│   ├── geo_utils.py              # 距离、mask、二维约束
│   └── vis.py                    # 可视化工具
├── train.py                      # 训练脚本
└── test.py                       # 生成 / 推理 / 排序脚本
```

## 三、模型结构与模块说明
### 1 条件扩散模型（DDPM）

本项目采用基于 DDPM 的连续坐标扩散模型，对二维材料的原子分数坐标进行逐步去噪生成。

- 正向过程：逐步向原子坐标加入高斯噪声
- 反向过程：学习去噪网络，预测噪声或残差坐标
- 条件信息：原子类型 z、晶格 lattice、原子 mask

### 2 材料结构生成器（Structure Generator）

扩散采样模块封装于 `structure_generator.py` 中，负责：

- 设定扩散步数 T
- 执行反向去噪采样
- 在采样过程中施加二维几何约束
- 支持引导梯度（Guidance）

该模块输出满足几何约束的候选二维材料原子结构。

### 3 性能引导优化模块

为了提升生成材料在 HER 催化任务中的性能，本项目引入基于梯度的引导优化模块。

优化目标包含：
- ΔG_H 接近 0（HER 活性）
- 热力学稳定性最大化
- 可合成性评分最大化
- 几何约束（最小原子间距、厚度、平滑性）

总体目标函数为加权和形式：

L = w₁|ΔG_H - ΔG_target|
  + w₂ ReLU(thermo_min - thermo)
  + w₃ ReLU(synth_min - synth)
  + w₄ L_min_dist
  + w₅ L_thickness
  + w₆ L_smooth

---

## 四、结果可视化
- `results/loss_curve.png`：训练过程中损失函数变化
- `results/her_performance.png`：生成材料的 ΔG_H 分布
- `results/stability_curve.png`：稳定性与合成性评分分布
- `results/generated_structures.png`：部分生成二维材料结构可视化
所有结果均由 `test.py` 自动生成并保存至 results/ 目录。

---

## 五、复现方式
### 1. 环境配置
```bash
conda create -n material_2d python=3.9
pip install -r requirements.txt
```
### 2. 训练
```bash
python train.py --data_dir ./data/JVASP
```
### 3. 生成与评估
```bash
python test.py --ckpt checkpoints/model.pt
```
---

## 六. 实验参数设置

### 1 数据与输入格式
- 数据格式：`.json`
- 每个结构包含：
  - `atomic_numbers`
  - `frac_coords`
  - `lattice`（可选）
  - `properties`（deltaG_H, thermo_stability, synth_score）
- 最大原子数：`max_atoms = 64`

**来源文件：**
- `dataset/material_dataset.py`

### 2 扩散模型参数
- 扩散步数：`T = 200`
- 扩散模型权重文件：`kpt_diffusion.pt`
- 是否施加 2D 约束：是（z 方向压缩）

**来源文件：**
- `test.py`
- `models/structure_generator.py`
- `utils/geo_utils.py (enforce_2d_constraints)`

### 3 性质预测模型参数
- 输出性质维度：3  
  - ΔG_H
  - thermo_stability
  - synth_score
- 性质预测模型权重文件：`kpt_property.pt`

**来源文件：**
- `test.py`
- `models/diffusion_model.py`

### 4 引导优化目标
优化目标函数由以下项加权组成：

L =  
w₁ · |ΔG_H − ΔG_target|  
+ w₂ · ReLU(thermo_min − thermo)  
+ w₃ · ReLU(synth_min − synth)  
+ w₄ · L_min_dist  
+ w₅ · L_thickness  
+ w₆ · L_smooth  

对应权重：
- w_deltaG = 1.0
- w_thermo = 0.8
- w_synth = 0.8
- w_min_dist = 0.8
- w_thickness = 0.3
- w_smooth = 0.1

目标约束：
- ΔG_H_target = 0.0
- thermo ≥ 0.5
- synth ≥ 0.8
- 最小原子间距 ≥ 1.6 Å
- 2D 厚度 ≤ 0.06

**来源文件：**
- `test.py`
- `models/optimization.py`
- `utils/geo_utils.py`

---

## 七. 实验评价指标

### 1 预测性能指标
- 预测 ΔG_H 分布（直方图）
- 预测热稳定性 vs 合成性散点图

**来源文件：**
- `utils/vis.py`
- `test.py (evaluate_and_save)`

### 2 结构合理性指标
- 最小原子间距约束
- 2D 厚度约束
- 原子坐标平滑性

**来源文件：**
- `utils/geo_utils.py`

### 3 综合评分与排序
- 使用性质预测结果计算综合 score
- 按 score 从高到低排序
- 输出 Top 候选结构

**输出文件：**
- `top_candidates.json`

**来源文件：**
- `models/optimization.py (rank_candidates)`
- `test.py`

---

## 八. 实验输出结果

### 1 生成结构结果
- 多个生成结构统一写入：
  - `top_candidates.json`
- 每个条目包含：
  - rank
  - score
  - 预测性质
  - 原子编号
  - 分数坐标
  - 晶格参数

**来源文件：**
- `test.py`


### 2 可视化结果
- `her_performance.png`（ΔG_H 分布）
- `stability_curve.png`（稳定性 vs 合成性）
- `generated_structures.png`（2D 超胞投影）

**来源文件：**
- `utils/vis.py`
- `test.py`

---

## 九. 结构文件格式说明

- 当前生成结果为 `.json` 格式
- `.json` 可直接转换为 `.cif` 用于材料模拟与可视化
- CIF 转换仅涉及：
  - 晶格参数
  - 分数坐标
  - 原子类型

**说明来源：**
- 项目整体数据流设计

---

## 十、创新点与方法说明
### 1 方法创新

- 提出一种 **基于条件扩散模型（DDPM）的二维材料结构生成框架**，在原子坐标空间直接进行结构建模。
- 将 **HER 催化活性（ΔG_H）**、**热稳定性** 与 **可合成性** 作为条件信号，引导结构生成过程。
- 通过训练独立的 **性质预测模型（Property Predictor）**，在扩散采样阶段进行梯度引导优化（Classifier-free Guidance 风格）。
### 2 网络结构概览

整体网络由三部分组成：

2.1. **性质预测网络（Property Predictor）**
   - 输入：原子类型 + 原子坐标 + 邻接图
   - 输出：ΔG_H、热稳定性、可合成性

2.2. **扩散去噪网络（Diffusion Denoiser）**
   - 逐步去噪原子坐标
   - 条件输入为时间步 t 与结构嵌入

2.3. **结构生成与优化模块**
   - 在采样阶段引入目标函数进行引导
   - 强制二维几何约束

