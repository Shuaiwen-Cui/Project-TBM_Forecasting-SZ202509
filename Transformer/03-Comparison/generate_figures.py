#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图片生成脚本 - 生成论文中规划的16张图
对应 论文.md 中图1～图16，输出到 03-Comparison/figures/fig01_*.png ... fig16_*.png
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / '02-Processing' / 'results'
DATA_PREPROCESSED = BASE_DIR / '01-Preprocessing' / 'data_preprocessed.csv'
OUTPUT_DIR = BASE_DIR / '03-Comparison' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / 'experiment_summary.csv'
PKL_DIR = RESULTS_DIR
METADATA_COLS = 5  # 与 02-Processing/config 一致

# 字体：中文宋体，其他一律 Times New Roman（符合投稿要求）
CHINESE_FONT = 'SimSun'
ENGLISH_FONT = 'Times New Roman'
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_PANEL = 7    # 子图标签 (a)(b)(c) 用 Times New Roman Bold，字号略小

# 双栏论文：单栏宽约 89 mm = 3.5 inch，全栏 183 mm ≈ 7.2 inch
COL_WIDTH_INCH = 3.5
DPI = 600
# 单图高度约 2.0–2.5 inch；多行一列时每行约 2.0 inch
FIG_H_SINGLE = 2.2
FIG_H_PANEL = 2.0

# 配色再淡一点（更浅的 pastel）
COLORS = {
    'ARIMA': '#7EB8DA',      # 淡蓝
    'LSTM': '#F0C674',       # 淡橙
    '1D-CNN': '#7ED4B3',     # 淡绿
    'Transformer': '#E8C4E0' # 淡紫
}
# 图1 竖排示意图用更浅色
COLORS_LIGHT = {
    'ARIMA': '#B8D4E8',
    'LSTM': '#F5E0B0',
    '1D-CNN': '#B8E8D4',
    'Transformer': '#F0DCE8'
}
# 热力图与散点图用相同顺序
CMAP_heatmap = 'RdYlGn'  # 或 'viridis'；保持 R² 红绿语义
CMAP_corr = 'RdBu_r'
GREY_TRUE = '#333333'  # 真实值曲线用深灰

MODEL_NAMES = ['ARIMA', 'LSTM', '1D-CNN', 'Transformer']
MODEL_NAMES_CN = {'ARIMA': 'ARIMA', 'LSTM': 'LSTM', '1D-CNN': '1D-CNN', 'Transformer': 'Transformer'}

SEQ_LENGTHS = [6, 60, 120, 360]
PRED_LENGTHS = [1, 6, 120, 360]
SEQ_LABELS = ['6步\n(0.5分钟)', '60步\n(5分钟)', '120步\n(10分钟)', '360步\n(30分钟)']
PRED_LABELS = ['1步\n(5秒)', '6步\n(0.5分钟)', '120步\n(10分钟)', '360步\n(30分钟)']

PKL_CONFIG = {
    'seq_len': 60,
    'pred_len': 1,
    'feature_indices': {'贯入度': 0, '推进压力': 1, '刀盘转速': 20},
    'sample_range': (0, 500)
}
# 图13～16 采用两种代表性配置，避免单一配置偏颇：单步(60,1) + 多步(60,120)
REP_CONFIGS = [(60, 1), (60, 120)]   # (seq_len, pred_len)
REP_CONFIG_LABELS = ['(a) 单步预测\nseq=60, pred=1', '(b) 多步预测\nseq=60, pred=120']

def setup_fonts():
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.serif'] = [ENGLISH_FONT, 'Times', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 2.5
    plt.rcParams['ytick.major.size'] = 2.5
    plt.rcParams['font.size'] = FONT_SIZE_TICK
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    # 纯白底、无网格或极淡网格，Nature/Science 风格
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
    sns.set_style("white", {'axes.edgecolor': '.15', 'grid.color': '.9'})

def style_axis(ax, panel_label=None, tick_fontsize=None):
    """统一坐标轴样式；刻度数字用 Times New Roman；子图标签 (a)(b)(c) 用 Times New Roman Bold。"""
    size = tick_fontsize if tick_fontsize is not None else FONT_SIZE_TICK
    ax.tick_params(axis='both', which='major', direction='out', labelsize=size)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(ENGLISH_FONT)
    if panel_label is not None:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=FONT_SIZE_PANEL,
                fontweight='bold', va='top', ha='left', fontfamily=ENGLISH_FONT)

def load_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"已加载实验汇总: {len(df)} 条")
    return df

def load_pkl_data(model_name, seq_len, pred_len):
    pkl_file = PKL_DIR / f'{model_name}_{seq_len}_{pred_len}_results.pkl'
    if not pkl_file.exists():
        return None
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def _get_series_from_pkl(data, key, feat_idx, start, end):
    """从 pkl 的 y_true_inv 或 y_pred_inv 取出单特征序列。形状为 (N*T, F) 或 (N, F)。"""
    y = data.get(key)
    if y is None:
        return None
    if y.ndim == 3:
        y = y[:, 0, :]  # 取第一个预测步
    return y[start:end, feat_idx].flatten()

# -----------------------------------------------------------------------------
# 图1：四种预测模型架构对比示意图（第3.3节）— 竖着画四个，配色浅
# -----------------------------------------------------------------------------
def fig01_architecture():
    titles = ['ARIMA', 'LSTM', '1D-CNN', 'Transformer']
    descs = ['自回归+滑动平均\n线性/平稳序列', '门控循环单元\n长时依赖', '一维卷积\n局部特征', '自注意力\n全局依赖']
    colors = [COLORS_LIGHT['ARIMA'], COLORS_LIGHT['LSTM'], COLORS_LIGHT['1D-CNN'], COLORS_LIGHT['Transformer']]
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_PANEL * 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)
    ax.axis('off')
    box_h, gap = 0.78, 0.06
    for i, (t, d) in enumerate(zip(titles, descs)):
        y_bottom = 3 - i * (box_h + gap)
        box = FancyBboxPatch((0.08, y_bottom), 0.84, box_h, boxstyle="round,pad=0.02",
                             facecolor=colors[i], edgecolor='.35', linewidth=0.6, alpha=0.9)
        ax.add_patch(box)
        ax.text(0.5, y_bottom + box_h * 0.68, t, ha='center', va='center', fontsize=FONT_SIZE_TITLE,
                fontweight='bold', fontfamily=ENGLISH_FONT)
        ax.text(0.5, y_bottom + box_h * 0.35, d, ha='center', va='center', fontsize=FONT_SIZE_TICK - 1,
                fontfamily=CHINESE_FONT)
    ax.set_title('四种预测模型架构对比', fontsize=FONT_SIZE_TITLE, fontfamily=CHINESE_FONT)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig01_architecture.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig01_architecture.png")

# -----------------------------------------------------------------------------
# 图2：数据集统计信息与参数分布（第3.2.1节）
# -----------------------------------------------------------------------------
def fig02_dataset():
    if not DATA_PREPROCESSED.exists():
        print("警告: data_preprocessed.csv 不存在，跳过图2")
        return
    df = pd.read_csv(DATA_PREPROCESSED, encoding='utf-8-sig', low_memory=False)
    if len(df) > 2:
        df = df.iloc[2:].reset_index(drop=True)
    numeric_cols = df.columns[METADATA_COLS:]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df_num = df[numeric_cols].ffill().bfill()
    n_samples, n_features = df_num.shape

    # 仅绘制参数相关性热力图（单子图形式）
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_PANEL * 1.2))
    # 参数相关性热力图 — 横轴刻度错开避免重叠（间隔显示或旋转）
    sub = df_num.iloc[:, :min(15, n_features)]
    corr = sub.corr()
    nf = len(sub.columns)
    im = ax.imshow(corr, cmap=CMAP_corr, aspect='auto', vmin=-1, vmax=1)
    # 横轴每个刻度都显示
    xticks = list(range(nf))
    ax.set_xticks(xticks)
    ax.set_yticks(range(nf))
    ax.set_xticklabels([f'F{xt+1}' for xt in xticks], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK - 1)
    ax.set_yticklabels([f'F{i+1}' for i in range(nf)], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK - 1)
    ax.set_xlabel('参数（Parameter）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily(ENGLISH_FONT)
    # style_axis(ax, '(b)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig02_dataset.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig02_dataset.png")

# -----------------------------------------------------------------------------
# 图3：四种模型预测精度对比（R²）（第5.1.1节）
# -----------------------------------------------------------------------------
def fig03_accuracy_r2(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    model_r2 = df.groupby('model')['R2'].mean().reindex(MODEL_NAMES)
    x = np.arange(4)
    bars = ax.bar(x, model_r2.values, color=[COLORS[m] for m in MODEL_NAMES], edgecolor='.3', linewidth=0.5, width=0.65)
    for bar, val in zip(bars, model_r2.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 if val >= 0 else val - 0.06,
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES_CN[m] for m in MODEL_NAMES], fontfamily=CHINESE_FONT)
    ax.set_ylim(0, 1)
    ax.set_ylabel('R²（平均值，mean）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.axhline(0, color='.4', linestyle='--', linewidth=0.6)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig03_accuracy_r2.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig03_accuracy_r2.png")

# -----------------------------------------------------------------------------
# 图4：四种模型预测精度对比（MSE、MAE、RMSE）（第5.1.1节）— 多行一列
# -----------------------------------------------------------------------------
def fig04_accuracy_mse_mae_rmse(df):
    fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH_INCH, FIG_H_PANEL * 3))
    metrics = [('MSE', '均方误差', 'MSE'), ('MAE', '平均绝对误差', 'MAE'), ('RMSE', '均方根误差', 'RMSE')]
    for idx, (col, cn_name, abbr) in enumerate(metrics):
        ax = axes[idx]
        model_metric = df.groupby('model')[col].mean().reindex(MODEL_NAMES)
        x = np.arange(4)
        vals = model_metric.values
        bars = ax.bar(x, vals, color=[COLORS[m] for m in MODEL_NAMES], edgecolor='.3', linewidth=0.5, width=0.65)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + val*0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=FONT_SIZE_TICK - 1, fontfamily=ENGLISH_FONT)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES_CN[m] for m in MODEL_NAMES], fontfamily=CHINESE_FONT)
        ax.set_ylabel(f'{cn_name}（{abbr}）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 0.01)
        style_axis(ax, f'({chr(97+idx)})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig04_accuracy_mse_mae_rmse.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig04_accuracy_mse_mae_rmse.png")

# -----------------------------------------------------------------------------
# 图5：四种模型推理时间对比（第5.1.2节）
# -----------------------------------------------------------------------------
def fig05_inference_time(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    model_time = df.groupby('model')['inference_time_ms'].mean().reindex(MODEL_NAMES)
    vals = model_time.values
    x = np.arange(4)
    bars = ax.bar(x, vals, color=[COLORS[m] for m in MODEL_NAMES], edgecolor='.3', linewidth=0.5, width=0.65)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES_CN[m] for m in MODEL_NAMES], fontfamily=CHINESE_FONT)
    ax.set_ylabel('推理时间（Inference time, ms）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_ylim(0, max(vals) * 1.15)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig05_inference_time.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig05_inference_time.png")

# -----------------------------------------------------------------------------
# 图6：四种模型内存占用对比（第5.1.2节）
# -----------------------------------------------------------------------------
def fig06_memory(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    model_mem = df.groupby('model')['memory_usage_mb'].mean().reindex(MODEL_NAMES)
    vals = np.maximum(model_mem.values, 0.01)
    x = np.arange(4)
    bars = ax.bar(x, vals, color=[COLORS[m] for m in MODEL_NAMES], edgecolor='.3', linewidth=0.5, width=0.65)
    for bar, v in zip(bars, model_mem.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02, f'{v:.1f}',
                ha='center', va='bottom', fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES_CN[m] for m in MODEL_NAMES], fontfamily=CHINESE_FONT)
    ax.set_ylabel('内存占用（Memory, MB）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_ylim(0, max(vals) * 1.15)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig06_memory.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig06_memory.png")

# -----------------------------------------------------------------------------
# 图7：四种模型训练时间对比（第5.1.2节）
# -----------------------------------------------------------------------------
def fig07_training_time(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    model_time = df.groupby('model')['training_time_s'].mean().reindex(MODEL_NAMES)
    vals = model_time.values
    x = np.arange(4)
    bars = ax.bar(x, vals, color=[COLORS[m] for m in MODEL_NAMES], edgecolor='.3', linewidth=0.5, width=0.65)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(vals)*0.02, f'{val:.0f}',
                ha='center', va='bottom', fontsize=FONT_SIZE_TICK, fontfamily=ENGLISH_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES_CN[m] for m in MODEL_NAMES], fontfamily=CHINESE_FONT)
    ax.set_ylabel('训练时间（Training time, s）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_ylim(0, max(vals) * 1.15)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig07_training_time.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig07_training_time.png")

# -----------------------------------------------------------------------------
# 图8：序列长度对预测性能的影响（第5.2.1节）
# -----------------------------------------------------------------------------
def fig08_seq_len_impact(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    for model in MODEL_NAMES:
        md = df[df['model'] == model]
        seq_r2 = md.groupby('seq_len')['R2'].mean().reindex(SEQ_LENGTHS)
        ax.plot(SEQ_LENGTHS, seq_r2.values, marker='o', linewidth=1.2, markersize=4,
                label=MODEL_NAMES_CN[model], color=COLORS[model])
    ax.set_xlabel('序列长度（Sequence length, 步）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_ylabel('R²（决定系数）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_xticks(SEQ_LENGTHS)
    ax.set_xticklabels(['6', '60', '120', '360'], fontfamily=ENGLISH_FONT)
    ax.legend(fontsize=FONT_SIZE_LEGEND, prop={'family': CHINESE_FONT})
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig08_seq_len_impact.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig08_seq_len_impact.png")

# -----------------------------------------------------------------------------
# 图9：预测长度对预测性能的影响（第5.2.2节）
# -----------------------------------------------------------------------------
def fig09_pred_len_impact(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    for model in MODEL_NAMES:
        md = df[df['model'] == model]
        pred_r2 = md.groupby('pred_len')['R2'].mean().reindex(PRED_LENGTHS)
        ax.plot(PRED_LENGTHS, pred_r2.values, marker='s', linewidth=1.2, markersize=4,
                label=MODEL_NAMES_CN[model], color=COLORS[model])
    ax.set_xlabel('预测长度（Prediction length, 步）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_ylabel('R²（决定系数）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_xticks(PRED_LENGTHS)
    ax.set_xticklabels(['1', '6', '120', '360'], fontfamily=ENGLISH_FONT)
    ax.legend(fontsize=FONT_SIZE_LEGEND, prop={'family': CHINESE_FONT})
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig09_pred_len_impact.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig09_pred_len_impact.png")

# -----------------------------------------------------------------------------
# 图10：时间尺度组合热力图（第5.2.3节）— 多行一列
# -----------------------------------------------------------------------------
def fig10_heatmap(df):
    fig, axes = plt.subplots(4, 1, figsize=(COL_WIDTH_INCH, FIG_H_PANEL * 4))
    for idx, model in enumerate(MODEL_NAMES):
        ax = axes[idx]
        md = df[df['model'] == model]
        pivot = md.pivot_table(values='R2', index='pred_len', columns='seq_len', aggfunc='mean')
        pivot = pivot.reindex(PRED_LENGTHS).reindex(columns=SEQ_LENGTHS)
        im = ax.imshow(pivot.values, cmap=CMAP_heatmap, aspect='auto', vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(len(SEQ_LENGTHS)))
        ax.set_xticklabels(['6', '60', '120', '360'], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK - 1)
        ax.set_yticks(range(len(PRED_LENGTHS)))
        ax.set_yticklabels(['1', '6', '120', '360'], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK - 1)
        for i in range(len(PRED_LENGTHS)):
            for j in range(len(SEQ_LENGTHS)):
                v = pivot.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FONT_SIZE_TICK - 2, fontfamily=ENGLISH_FONT)
        ax.set_xlabel('序列长度（步）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
        ax.set_ylabel('预测长度（步）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
        style_axis(ax, f'({chr(97+idx)}) {MODEL_NAMES_CN[model]}')
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)
        for t in cbar.ax.get_yticklabels():
            t.set_fontfamily(ENGLISH_FONT)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig10_heatmap.png")

# -----------------------------------------------------------------------------
# 图11：精度-效率权衡散点图（第5.4节）
# -----------------------------------------------------------------------------
def fig11_tradeoff_scatter(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE))
    markers = ['o', 's', '^', 'D']
    for idx, model in enumerate(MODEL_NAMES):
        md = df[df['model'] == model]
        ax.scatter(md['inference_time_ms'], md['R2'], s=28, alpha=0.85,
                  label=MODEL_NAMES_CN[model], color=COLORS[model], marker=markers[idx],
                  edgecolors='.3', linewidths=0.4)
    ax.set_xlabel('推理时间（Inference time, ms）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.set_ylabel('R²（决定系数）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
    ax.legend(fontsize=FONT_SIZE_LEGEND, prop={'family': CHINESE_FONT})
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_tradeoff_scatter.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig11_tradeoff_scatter.png")

# -----------------------------------------------------------------------------
# 图12：模型综合性能雷达图（第5.4节）— 多行一列 4 子图
# -----------------------------------------------------------------------------
def fig12_radar(df):
    from math import pi
    metrics = ['R2', 'inference_time_ms', 'memory_usage_mb', 'training_time_s']
    metric_names = ['R²', '推理速度', '内存', '训练时间']
    def norm_r2(x, lo, hi):
        return (x - lo) / (hi - lo) if hi != lo else 0.5
    def norm_inv(x, lo, hi):
        return 1 - (x - lo) / (hi - lo) if hi != lo else 0.5

    ranges = {m: (df[m].min(), df[m].max()) for m in metrics}
    fig, axes = plt.subplots(4, 1, figsize=(COL_WIDTH_INCH, FIG_H_PANEL * 4), subplot_kw=dict(projection='polar'))
    fs_small = max(6, FONT_SIZE_TICK - 2)  # 图12 字体稍小
    for idx, model in enumerate(MODEL_NAMES):
        ax = axes[idx]
        md = df[df['model'] == model]
        vals = []
        for m in metrics:
            v = md[m].mean()
            if m == 'R2':
                vals.append(norm_r2(v, ranges[m][0], ranges[m][1]))
            else:
                vals.append(norm_inv(v, ranges[m][0], ranges[m][1]))
        vals += vals[:1]
        angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))] + [2 * pi]
        ax.plot(angles, vals, 'o-', linewidth=1.2, color=COLORS[model], markersize=3)
        ax.fill(angles, vals, alpha=0.2, color=COLORS[model])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=fs_small, fontfamily=CHINESE_FONT)
        ax.set_ylim(0, 1)
        ax.set_yticks([1.0])
        ax.set_yticklabels(['1.0'], fontsize=fs_small, fontfamily=CHINESE_FONT)
        ax.set_title(f'({chr(97+idx)}) {MODEL_NAMES_CN[model]}', fontsize=FONT_SIZE_PANEL - 1, fontfamily=CHINESE_FONT, pad=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_radar.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig12_radar.png")

# -----------------------------------------------------------------------------
# 图13/14/15：关键参数预测效果对比（贯入度、推进压力、刀盘转速）（第5.5节）
# 两列子图：左 单步(60,1)，右 多步(60,120)，避免单一配置偏颇
# -----------------------------------------------------------------------------
def _fig_prediction_one_param_dual(feat_name, feat_idx, unit, filename):
    start, end = PKL_CONFIG['sample_range']
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH_INCH * 1.1, FIG_H_SINGLE * 0.9))
    for col, ((seq_len, pred_len), title) in enumerate(zip(REP_CONFIGS, REP_CONFIG_LABELS)):
        ax = axes[col]
        first = None
        for m in MODEL_NAMES:
            d = load_pkl_data(m, seq_len, pred_len)
            if d is not None:
                first = d
                break
        if first is None:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
            continue
        y_true = _get_series_from_pkl(first, 'y_true_inv', feat_idx, start, end)
        if y_true is None or len(y_true) == 0:
            ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
            continue
        x = np.arange(len(y_true))
        ax.plot(x, y_true, color=GREY_TRUE, linewidth=1.0, label='真实值', alpha=0.95)
        for model in MODEL_NAMES:
            d = load_pkl_data(model, seq_len, pred_len)
            if d is None:
                continue
            y_pred = _get_series_from_pkl(d, 'y_pred_inv', feat_idx, start, end)
            if y_pred is not None and len(y_pred) == len(y_true):
                ax.plot(x, y_pred, '-', linewidth=0.8, label=MODEL_NAMES_CN[model], color=COLORS[model], alpha=0.9)
        ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
        ax.set_xlabel('样本索引', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_ylabel(f'{feat_name} {unit}', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.legend(fontsize=FONT_SIZE_LEGEND - 1, prop={'family': CHINESE_FONT}, loc='upper right', frameon=True)
        style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"已生成: {filename}")

def fig13_pred_penetration():
    _fig_prediction_one_param_dual('贯入度', 0, '(mm/min)', 'fig13_pred_penetration.png')

def fig14_pred_pressure():
    _fig_prediction_one_param_dual('推进压力（上）', 1, '(MPa)', 'fig14_pred_pressure.png')

def fig15_pred_cutterhead():
    _fig_prediction_one_param_dual('刀盘转速', 20, '(r/min)', 'fig15_pred_cutterhead.png')

# -----------------------------------------------------------------------------
# 图16：多参数预测误差分布箱线图（第5.5节）
# 两列：左 单步(60,1)，右 多步(60,120)，与图13～15一致，避免单一配置偏颇
# -----------------------------------------------------------------------------
def fig16_error_boxplot():
    n_features = 30
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH_INCH * 1.1, FIG_H_SINGLE * 0.9))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.5, edgecolor='.3', label='箱体：25%～75%分位'),
        Line2D([0], [0], color='.3', linewidth=2, label='横线：中位数'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markeredgecolor='green', markersize=5, linestyle='None', label='三角：均值'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='.5', markersize=4, linestyle='None', label='圆圈：异常值'),
    ]
    for col, ((seq_len, pred_len), title) in enumerate(zip(REP_CONFIGS, REP_CONFIG_LABELS)):
        ax = axes[col]
        data_by_model = {m: [] for m in MODEL_NAMES}
        for model in MODEL_NAMES:
            d = load_pkl_data(model, seq_len, pred_len)
            if d is None:
                continue
            y_true = d['y_true_inv']
            y_pred = d['y_pred_inv']
            if y_true.ndim == 3:
                y_true = y_true.reshape(-1, y_true.shape[-1])
                y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            for f in range(min(n_features, y_true.shape[1])):
                mse = np.mean((y_true[:, f] - y_pred[:, f]) ** 2)
                data_by_model[model].append(mse)
        if not any(data_by_model[m] for m in MODEL_NAMES):
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
            continue
        box_data = [data_by_model[m] for m in MODEL_NAMES if data_by_model[m]]
        labels = [MODEL_NAMES_CN[m] for m in MODEL_NAMES if data_by_model[m]]
        colors_list = [COLORS[m] for m in MODEL_NAMES if data_by_model[m]]
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, widths=0.6)
        for patch, c in zip(bp['boxes'], colors_list):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            patch.set_edgecolor('.3')
        ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
        ax.set_ylabel('均方误差（MSE）', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_xticklabels(labels, fontfamily=CHINESE_FONT)
        style_axis(ax)
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZE_LEGEND - 2, prop={'family': CHINESE_FONT})
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig16_error_boxplot.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig16_error_boxplot.png")

# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 60)
    print("论文图片生成脚本 - 生成图1～图16")
    print("=" * 60)
    setup_fonts()
    df = load_data()

    fig01_architecture()
    fig02_dataset()
    fig03_accuracy_r2(df)
    fig04_accuracy_mse_mae_rmse(df)
    fig05_inference_time(df)
    fig06_memory(df)
    fig07_training_time(df)
    fig08_seq_len_impact(df)
    fig09_pred_len_impact(df)
    fig10_heatmap(df)
    fig11_tradeoff_scatter(df)
    fig12_radar(df)
    fig13_pred_penetration()
    fig14_pred_pressure()
    fig15_pred_cutterhead()
    fig16_error_boxplot()

    print("\n" + "=" * 60)
    print(f"所有图片已保存至: {OUTPUT_DIR}")
    print("图1～图16 对应论文 论文.md 中图表清单")
    print("=" * 60)

if __name__ == '__main__':
    main()
