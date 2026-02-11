# -*- coding: utf-8 -*-
"""03-Analyses 路径与绘图常量，与 02-Processing 实验配置一致。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

RESULTS_DIR = PROJECT_ROOT / "02-Processing" / "results"
DATA_PREPROCESSED = PROJECT_ROOT / "01-Preprocessing" / "data_preprocessed.csv"
CSV_FILE = RESULTS_DIR / "experiment_summary.csv"
PKL_DIR = RESULTS_DIR
OUTPUT_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

METADATA_COLS = 5
SEQ_LENGTHS = [6, 30, 60, 120, 360]
PRED_LENGTHS = [1, 6, 120, 360]
MODEL_NAME = "Transformer"

# 绘图（学术规范：中文宋体、英文与数字 Times New Roman）
CHINESE_FONT = "SimSun"
ENGLISH_FONT = "Times New Roman"
FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_PANEL = 9
COL_WIDTH_INCH = 3.5
COL_WIDTH_DOUBLE = 7.0
DPI = 300
FIG_H_SINGLE = 2.4
FIG_H_PANEL = 2.2
CMAP_HEATMAP = "RdYlGn"
CMAP_CORR = "RdBu_r"
# 曲线与柱状配色（美观得体）
GREY_TRUE = "#2d2d2d"
COLOR_PRED = "#1f77b4"
COLOR_BAR = "#4A90D9"
COLOR_BAR_EDGE = "#2E5A8A"
COLOR_SECOND = "#2ca02c"

# 代表性配置（用于精度-效率等）
REP_CONFIGS = [(60, 1), (60, 120)]
SAMPLE_RANGE = (0, 500)
# 预测曲线：图7～10 固定一个物理量（贯入度），每图固定 pred_len，五子图对应 seq_len=6,30,60,120,360
# 图7: pred_len=1；图8: pred_len=6；图9: pred_len=120；图10: pred_len=360
PREDICTION_CURVE_PRED_LENS = [1, 6, 120, 360]  # 图7/8/9/10 对应的预测长度
# 图7～10 使用的物理量（特征索引, 特征名称），统一用贯入度
PREDICTION_FEATURE_SINGLE = (0, "贯入度")
