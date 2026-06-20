# -*- coding: utf-8 -*-
"""
统一配置：数据路径、序列/预测长度、划分比例、Transformer 超参数。
"""
from pathlib import Path

# 项目根目录（02-Processing 的上级）
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# 数据文件（与 01-Preprocessing 输出一致）
DATA_FILE = PROJECT_ROOT / "01-Preprocessing" / "data_preprocessed.csv"

# 结果目录
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 前几列为元数据（不参与建模）
METADATA_COLS = 5

# 实验组合：输入序列长度 × 预测长度 = 20 组
SEQ_LENGTHS = [6, 30, 60, 120, 360]
PRED_LENGTHS = [1, 6, 120, 360]

# 数据划分比例（训练 / 验证 / 测试）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Transformer 模型超参数
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "batch_size": 64,
    "epochs": 20,
    "lr": 1e-3,
}

# 推理时每批样本数，避免 seq_len 很大时整批测试集导致 OOM
INFERENCE_BATCH_SIZE = 32

# 状态与汇总文件名
STATUS_FILE = RESULTS_DIR / "experiment_status.json"
SUMMARY_CSV = RESULTS_DIR / "experiment_summary.csv"
MODEL_NAME = "Transformer"
