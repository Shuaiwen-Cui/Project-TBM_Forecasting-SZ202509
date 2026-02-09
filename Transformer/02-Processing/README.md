# 实验框架使用说明

## 概述

本实验框架基于 `plan.txt` 的实验设计，用于对比研究4种深度学习模型（ARIMA、LSTM、1D-CNN、Transformer）在盾构掘进关键参数预测任务上的性能。

## 实验设计（基于plan.txt）

### I. AI模型（精简版）
- ARIMA
- LSTM
- 1D-CNN
- Transformer

### II. 序列长度（输入历史数据长度）
- **6步**：5s × 6 = 30s = 0.5min（极短期，捕捉瞬时模式）
- **60步**：5s × 60 = 300s = 5min（短期，捕捉短期趋势）
- **120步**：5s × 120 = 600s = 10min（中期，捕捉中期模式）
- **360步**：5s × 360 = 1800s = 30min（长期，捕捉长期趋势）

### III. 预测长度（预测未来步数）
- **1步**：5s = 0.083min（极短期，单步预测）
- **6步**：5s × 6 = 30s = 0.5min（短期预测）
- **120步**：5s × 120 = 600s = 10min（中期预测）
- **360步**：5s × 360 = 1800s = 30min（长期预测）

### IV. 评估指标（基于plan.txt）

#### 模型大小
- 参数量（model_size_params）
- 模型文件大小（model_size_mb）

#### 精度类指标
- MSE（均方误差）
- MAE（平均绝对误差）
- RMSE（均方根误差）
- MAPE（平均绝对百分比误差）
- R²（决定系数）

#### 性能类指标
- 推理时间（inference_time_ms，单位：毫秒）
- 内存占用（memory_usage_mb，单位：MB）
- 训练时间（training_time_s，单位：秒）

### 总实验数
4模型 × 4序列长度 × 4预测长度 = **64个实验组合**

## 目录结构

```
02-Processing/
├── config.py                 # 统一配置文件（所有参数）
├── data_loader.py           # 数据加载和预处理
├── utils.py                 # 工具函数（评估指标、时间测量等）
├── experiment_runner.py     # 主实验运行器
├── run_experiments.py       # 主运行脚本
├── analyze_results.py      # 结果分析脚本
├── models/                  # 模型实现目录
│   ├── __init__.py
│   ├── base_model.py       # 模型基类（统一接口）
│   ├── arima_model.py      # ARIMA实现（需要实现）
│   ├── lstm_model.py       # LSTM实现（需要实现）
│   ├── cnn1d_model.py      # 1D-CNN实现（需要实现）
│   └── transformer_model.py # Transformer实现（需要实现）
├── results/                 # 结果保存目录
│   ├── experiment_summary.csv  # 汇总表（所有实验结果）
│   └── {model}_{seq_len}_{pred_len}_results.pkl  # 详细结果
├── plan.txt                 # 实验设计文档
└── README.md               # 本文件
```

## 快速开始

### 1. 检查环境

```bash
# 进入实验目录
cd Transformer/02-Processing

# 安装依赖（需要根据实际情况创建requirements.txt）
pip install numpy pandas scikit-learn torch psutil
```

### 2. 检查数据文件

确保数据文件存在：
```bash
python -c "from pathlib import Path; from config import DATA_FILE; print(f'数据文件: {DATA_FILE}'); print(f'存在: {DATA_FILE.exists()}')"
```

### 3. 实现模型（重要）

在运行实验前，需要实现各个模型。每个模型需要继承 `BaseModel` 并实现以下方法：
- `train()`: 训练模型
- `predict()`: 预测
- `save()`: 保存模型
- `load()`: 加载模型

模型文件应放在 `models/` 目录下：
- `models/arima_model.py`
- `models/lstm_model.py`
- `models/cnn1d_model.py`
- `models/transformer_model.py`

### 4. 运行实验

```bash
# 仅运行未完成的实验（默认，跳过已有结果）
python run_experiments.py
# 或
python run_experiments.py --resume

# 全部执行并覆盖已有结果（便于论文分析）
python run_experiments.py --overwrite
# 或
python run_experiments.py -f
```

这会：
- 自动运行所有64个实验
- **默认**：仅运行未完成的（跳过已有），支持断点续跑
- **加 `--overwrite` / `-f`**：全部执行并覆盖已有结果，便于论文分析
- 每个实验独立加载数据，用完即释放（节省内存）
- 记录所有关键数据（基于plan.txt的评估指标）

### 5. 分析结果

```bash
python analyze_results.py
```

这会生成：
- `results/analysis_summary.txt` - 文本格式的详细分析
- `results/model_statistics.csv` - CSV格式的统计表（便于导入Excel和论文写作）

## 结果文件说明

### 汇总表 (`experiment_summary.csv`)

包含所有实验的关键指标，便于快速对比和分析。可以直接用Excel打开。

列包括：
- `model`, `seq_len`, `pred_len`: 实验配置
- `MSE`, `MAE`, `RMSE`, `MAPE`, `R2`: 精度类指标
- `model_size_params`, `model_size_mb`: 模型大小
- `inference_time_ms`, `batch_inference_time_ms`, `memory_usage_mb`, `training_time_s`: 性能类指标

### 详细结果 (`{model}_{seq_len}_{pred_len}_results.pkl`)

包含：
- `model_name`: 模型名称
- `feature_names`: 特征名称列表
- `metrics`: 所有评估指标（基于plan.txt）
- `costs`: 性能指标和模型大小（基于plan.txt）
- `config`: 实验配置参数
- `y_true`, `y_pred`: 归一化后的真实值和预测值
- `y_true_inv`, `y_pred_inv`: 反归一化后的真实值和预测值（用于绘图）
- `scaler`: 归一化器
- `feature_mapping`: 特征映射信息（便于绘图脚本使用）

## 内存优化

框架采用以下策略减少内存占用：
1. **分次加载**: 每个实验独立加载数据，用完即释放
2. **分批处理**: 大数据集分批处理序列生成
3. **及时清理**: 每10个实验主动清理内存
4. **float32**: 统一使用float32数据类型

## 覆盖与断点续跑

- **默认**（`python run_experiments.py` 或 `--resume`）：仅运行未完成的实验，跳过已有结果，支持断点续跑。
- **覆盖模式**（`python run_experiments.py --overwrite` 或 `-f`）：全部执行并覆盖已有 pkl 与汇总表，便于论文分析。
- 状态保存在 `results/experiment_status.json`。

## 注意事项

1. **运行时间**: 64个实验可能需要数小时到数天，取决于模型复杂度和硬件配置
2. **GPU支持**: 如果有GPU，PyTorch模型（LSTM、1D-CNN、Transformer）会自动使用GPU加速
3. **数据路径**: 确保 `01-Preprocessing/data_preprocessed.csv` 存在
4. **模型实现**: 在运行实验前，需要先实现各个模型

## 论文写作支持

### 生成论文图片

运行绘图脚本会自动生成关键图片：

```bash
cd ../03-Comparison
python generate_figures.py
```

### 数据分析

运行分析脚本可以生成详细的分析报告，包括：
- 按模型统计的平均值和标准差
- 最佳模型（按R²值）
- 最快模型（按推理时间）
- 序列长度和预测长度对性能的影响
- 精度-效率权衡分析

所有分析结果都基于plan.txt的评估指标，便于论文写作。

## 故障排除

1. **模型未实现**: 如果模型未实现，实验会跳过并记录警告
2. **内存不足**: 可以减少序列长度或预测长度的列表
3. **实验失败**: 查看终端错误信息，单个实验失败不影响其他实验
4. **结果丢失**: 所有结果都保存在 `results/` 目录，可以随时查看
