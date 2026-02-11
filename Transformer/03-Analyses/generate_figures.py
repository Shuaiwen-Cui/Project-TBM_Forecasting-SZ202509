#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图表生成脚本（03-Analyses）
- 图1：仅数据集参数相关性（无架构子图、无(b)标号）
- 多子图统一为多行一列；中文标注宋体，字母数字 Times
- 图2/3：展示全部16种(seq_len,pred_len)配置
- 图4：四模型三维曲面（seq_len × pred_len → R²）
- 图6：上下罗列；图7：多行一列、多种配置
- 图8：不再绘制
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from math import pi
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 路径
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / '02-Processing' / 'results'
DATA_PREPROCESSED = BASE_DIR / '01-Preprocessing' / 'data_preprocessed.csv'
PKL_DIR = RESULTS_DIR
OUTPUT_DIR = Path(__file__).resolve().parent / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / 'experiment_summary.csv'
METADATA_COLS = 5

CHINESE_FONT = 'SimSun'
ENGLISH_FONT = 'Times New Roman'
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_PANEL = 7
FONT_SIZE_TICK_SMALL = 6  # 图1等刻度较密时使用

COL_WIDTH_INCH = 3.5
DPI = 600
FIG_H_SINGLE = 2.2
FIG_H_PANEL = 2.0

COLORS = {
    'ARIMA': '#7EB8DA',
    'LSTM': '#F0C674',
    '1D-CNN': '#7ED4B3',
    'Transformer': '#E8C4E0'
}
CMAP_heatmap = 'RdYlGn'
CMAP_corr = 'RdBu_r'
GREY_TRUE = '#333333'

MODEL_NAMES = ['ARIMA', 'LSTM', '1D-CNN', 'Transformer']
MODEL_NAMES_CN = {'ARIMA': 'ARIMA', 'LSTM': 'LSTM', '1D-CNN': '1D-CNN', 'Transformer': 'Transformer'}
SEQ_LENGTHS = [6, 60, 120, 360]
PRED_LENGTHS = [1, 6, 120, 360]

# 16种配置的固定顺序（与 CSV 按 model 分组后一致）
CONFIG_ORDER = [(s, p) for s in SEQ_LENGTHS for p in PRED_LENGTHS]
CONFIG_LABELS = [f'{s}-{p}' for s in SEQ_LENGTHS for p in PRED_LENGTHS]

REP_CONFIGS = [(60, 1), (60, 120)]
REP_CONFIG_LABELS = ['(60,1) 单步', '(60,120) 多步']
SAMPLE_RANGE = (0, 500)


def setup_fonts():
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.serif'] = [ENGLISH_FONT, 'Times', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['font.size'] = FONT_SIZE_TICK
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    sns.set_style("white", {'axes.edgecolor': '.15', 'grid.color': '.9'})


def style_axis(ax, panel_label=None):
    ax.tick_params(axis='both', which='major', direction='out', labelsize=FONT_SIZE_TICK)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(ENGLISH_FONT)
    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=FONT_SIZE_PANEL,
               fontweight='bold', va='top', ha='left', fontfamily=ENGLISH_FONT)


def load_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {CSV_FILE}")
    return pd.read_csv(CSV_FILE)


def load_pkl_data(model_name, seq_len, pred_len):
    pkl_file = PKL_DIR / f'{model_name}_{seq_len}_{pred_len}_results.pkl'
    if not pkl_file.exists():
        return None
    try:
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _get_series_from_pkl(data, key, feat_idx, start, end):
    y = data.get(key)
    if y is None:
        return None
    if y.ndim == 3:
        y = y[:, 0, :]
    return y[start:end, feat_idx].flatten()


def _get_model_config_values(df, model, col):
    """按 CONFIG_ORDER 顺序返回某模型某列的 16 个值。"""
    md = df[df['model'] == model]
    if len(md) != 16:
        # 若行数不对，按 (seq_len, pred_len) 排序后取
        md = md.sort_values(['seq_len', 'pred_len'])
    lookup = md.set_index(['seq_len', 'pred_len'])[col]
    return [lookup.loc[(s, p)] for s, p in CONFIG_ORDER]


# -----------------------------------------------------------------------------
# 图1：仅数据集参数分布（无架构子图、无(b)标号）
# -----------------------------------------------------------------------------
def fig01_dataset_only():
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH, FIG_H_SINGLE * 1.2))
    if DATA_PREPROCESSED.exists():
        df = pd.read_csv(DATA_PREPROCESSED, encoding='utf-8-sig', low_memory=False)
        if len(df) > 2:
            df = df.iloc[2:].reset_index(drop=True)
        numeric_cols = df.columns[METADATA_COLS:]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df_num = df[numeric_cols].ffill().bfill()
        sub = df_num.iloc[:, :min(15, len(df_num.columns))]
        corr = sub.corr()
        nf = len(sub.columns)
        im = ax.imshow(corr, cmap=CMAP_corr, aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(nf))
        ax.set_yticks(range(nf))
        ax.set_xticklabels([f'F{i+1}' for i in range(nf)], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK_SMALL)
        ax.set_yticklabels([f'F{i+1}' for i in range(nf)], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK_SMALL)
        ax.set_xlabel('参数（Parameter）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        for t in cbar.ax.get_yticklabels():
            t.set_fontfamily(ENGLISH_FONT)
    else:
        ax.text(0.5, 0.5, 'data_preprocessed.csv 不存在', ha='center', va='center', transform=ax.transAxes, fontfamily=CHINESE_FONT)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig01_dataset.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig01_dataset.png")


# -----------------------------------------------------------------------------
# 图2：整体预测精度 — 4 个 3D 子图，图例共用一个放在所有子图外，子图间距加大
# -----------------------------------------------------------------------------
def fig02_accuracy(df):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D
    metrics = [
        ('R2', '决定系数 (R²)', 0, 1),
        ('MSE', '均方误差 (MSE)', None, None),
        ('MAE', '平均绝对误差 (MAE)', None, None),
        ('RMSE', '均方根误差 (RMSE)', None, None),
    ]
    fig = plt.figure(figsize=(COL_WIDTH_INCH * 1.2, FIG_H_PANEL * 4.2))
    for row, (col, zlabel_cn_en, zlo, zhi) in enumerate(metrics):
        ax = fig.add_subplot(4, 1, row + 1, projection='3d')
        for model in MODEL_NAMES:
            xs = [s for s, p in CONFIG_ORDER]
            ys = [p for s, p in CONFIG_ORDER]
            zs = _get_model_config_values(df, model, col)
            ax.scatter(xs, ys, zs, s=20, alpha=0.85, color=COLORS[model], edgecolors='.3', linewidths=0.3)
        ax.set_xlabel('序列长度 (步)', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_ylabel('预测长度 (步)', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_zlabel(zlabel_cn_en, fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        if zlo is not None and zhi is not None:
            ax.set_zlim(zlo, zhi)
        for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            t.set_fontfamily(ENGLISH_FONT)
            t.set_fontsize(FONT_SIZE_TICK - 1)
        ax.text2D(0.02, 0.98, f'({chr(97+row)})', transform=ax.transAxes, fontsize=FONT_SIZE_PANEL, fontweight='bold', fontfamily=ENGLISH_FONT)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[m], markersize=8, label=MODEL_NAMES_CN[m]) for m in MODEL_NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=FONT_SIZE_LEGEND - 1, prop={'family': CHINESE_FONT})
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.96, hspace=0.55)
    plt.savefig(OUTPUT_DIR / 'fig02_accuracy.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig02_accuracy.png")


# -----------------------------------------------------------------------------
# 图3：计算性能 — 3 个 3D 子图，图例共用一个放在所有子图外，子图间距加大
# -----------------------------------------------------------------------------
def fig03_compute(df):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D
    configs = [
        ('inference_time_ms', '推理时间 (ms)'),
        ('memory_usage_mb', '内存 (MB)'),
        ('training_time_s', '训练时间 (s)'),
    ]
    fig = plt.figure(figsize=(COL_WIDTH_INCH * 1.2, FIG_H_PANEL * 3.2))
    for row, (col, zlabel_cn_en) in enumerate(configs):
        ax = fig.add_subplot(3, 1, row + 1, projection='3d')
        for model in MODEL_NAMES:
            xs = [s for s, p in CONFIG_ORDER]
            ys = [p for s, p in CONFIG_ORDER]
            zs = [max(v, 1e-6) for v in _get_model_config_values(df, model, col)]
            ax.scatter(xs, ys, zs, s=20, alpha=0.85, color=COLORS[model], edgecolors='.3', linewidths=0.3)
        ax.set_xlabel('序列长度 (步)', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_ylabel('预测长度 (步)', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_zlabel(zlabel_cn_en, fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            t.set_fontfamily(ENGLISH_FONT)
            t.set_fontsize(FONT_SIZE_TICK - 1)
        ax.text2D(0.02, 0.98, f'({chr(97+row)})', transform=ax.transAxes, fontsize=FONT_SIZE_PANEL, fontweight='bold', fontfamily=ENGLISH_FONT)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[m], markersize=8, label=MODEL_NAMES_CN[m]) for m in MODEL_NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=FONT_SIZE_LEGEND - 1, prop={'family': CHINESE_FONT})
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.14, top=0.96, hspace=0.6)
    plt.savefig(OUTPUT_DIR / 'fig03_compute.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig03_compute.png")


# -----------------------------------------------------------------------------
# 图4：时间尺度影响 — 四模型各一张三维曲面
# 坐标：x=序列长度, y=预测长度, z（纵轴/竖直）= 决定系数 R²；减少刻度重叠
# -----------------------------------------------------------------------------
def fig04_3d_surface(df):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(COL_WIDTH_INCH * 1.15, FIG_H_PANEL * 4.8))
    fig.suptitle('纵轴 z = 决定系数 R²（竖直方向）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, y=0.98)
    X = np.array(SEQ_LENGTHS)
    Y = np.array(PRED_LENGTHS)
    XX, YY = np.meshgrid(X, Y)
    vmin, vmax = -0.5, 1.0
    elev, azim = 22, 48
    z_scale = 0.45
    for idx, model in enumerate(MODEL_NAMES):
        ax = fig.add_subplot(4, 1, idx + 1, projection='3d')
        md = df[df['model'] == model]
        pivot = md.pivot_table(values='R2', index='pred_len', columns='seq_len', aggfunc='mean')
        pivot = pivot.reindex(PRED_LENGTHS).reindex(columns=SEQ_LENGTHS)
        ZZ = np.array(pivot.values, dtype=float)
        ax.plot_surface(XX, YY, ZZ, cmap=CMAP_heatmap, alpha=0.85, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x: 序列长度(步)', fontsize=FONT_SIZE_LABEL - 2, fontfamily=CHINESE_FONT)
        ax.set_ylabel('y: 预测长度(步)', fontsize=FONT_SIZE_LABEL - 2, fontfamily=CHINESE_FONT)
        ax.set_zlabel('z: R²（纵轴）', fontsize=FONT_SIZE_LABEL - 2, fontfamily=CHINESE_FONT)
        ax.set_title(MODEL_NAMES_CN[model], fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
        ax.set_xticks(SEQ_LENGTHS)
        ax.set_yticks(PRED_LENGTHS)
        ax.tick_params(axis='x', labelsize=FONT_SIZE_TICK - 2, pad=2)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK - 2, pad=2)
        ax.tick_params(axis='z', labelsize=FONT_SIZE_TICK - 2, pad=2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, z_scale))
        for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            t.set_fontfamily(ENGLISH_FONT)
    plt.subplots_adjust(left=0.10, right=0.92, bottom=0.05, top=0.92, hspace=0.65)
    plt.savefig(OUTPUT_DIR / 'fig04_3d_surface.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig04_3d_surface.png")


# -----------------------------------------------------------------------------
# 图5：时间尺度组合热力图 — 多行一列，四模型
# -----------------------------------------------------------------------------
def fig05_heatmap(df):
    fig, axes = plt.subplots(4, 1, figsize=(COL_WIDTH_INCH * 1.1, FIG_H_PANEL * 4))
    vmin, vmax = -0.5, 1.0
    for idx, model in enumerate(MODEL_NAMES):
        ax = axes[idx]
        md = df[df['model'] == model]
        pivot = md.pivot_table(values='R2', index='pred_len', columns='seq_len', aggfunc='mean')
        pivot = pivot.reindex(PRED_LENGTHS).reindex(columns=SEQ_LENGTHS)
        im = ax.imshow(pivot.values, cmap=CMAP_heatmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SEQ_LENGTHS)))
        ax.set_xticklabels([str(s) for s in SEQ_LENGTHS], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK - 1)
        ax.set_yticks(range(len(PRED_LENGTHS)))
        ax.set_yticklabels([str(p) for p in PRED_LENGTHS], fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_TICK - 1)
        for i in range(len(PRED_LENGTHS)):
            for j in range(len(SEQ_LENGTHS)):
                v = pivot.values[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FONT_SIZE_TICK - 2, fontfamily=ENGLISH_FONT)
        ax.set_xlabel('序列长度 (步)', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_ylabel('预测长度 (步)', fontsize=FONT_SIZE_LABEL - 1, fontfamily=CHINESE_FONT)
        ax.set_title(f'({chr(97+idx)}) {MODEL_NAMES_CN[model]}', fontsize=FONT_SIZE_PANEL, fontfamily=CHINESE_FONT)
        plt.colorbar(im, ax=ax, shrink=0.7, label='R²')
        style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig05_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig05_heatmap.png")


# -----------------------------------------------------------------------------
# 图6：精度-效率 — 4×4 网格，每格一种(seq_len, pred_len)配置，格内 4 根柱=四模型 R²
# 行=预测长度 pred，列=序列长度 seq；图例 2×2 共用一个
# -----------------------------------------------------------------------------
def fig06_tradeoff(df):
    from matplotlib.lines import Line2D
    fp_cn = font_manager.FontProperties(family=CHINESE_FONT, size=FONT_SIZE_LEGEND)
    fp_en = font_manager.FontProperties(family=ENGLISH_FONT, size=FONT_SIZE_TICK)

    fig, axes = plt.subplots(4, 4, figsize=(COL_WIDTH_INCH * 2.0, FIG_H_PANEL * 2.2))
    for row, pred_len in enumerate(PRED_LENGTHS):
        for col, seq_len in enumerate(SEQ_LENGTHS):
            ax = axes[row, col]
            vals = []
            for model in MODEL_NAMES:
                r = df[(df['model'] == model) & (df['seq_len'] == seq_len) & (df['pred_len'] == pred_len)]
                vals.append(r['R2'].values[0] if len(r) > 0 else 0)
            x = np.arange(4)
            bars = ax.bar(x, vals, color=[COLORS[m] for m in MODEL_NAMES], edgecolor='.35', linewidth=0.5, width=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_NAMES_CN[m] for m in MODEL_NAMES], fontsize=FONT_SIZE_TICK - 2, fontfamily=CHINESE_FONT, rotation=15)
            ax.set_ylabel('R²', fontsize=FONT_SIZE_LABEL - 1, fontproperties=fp_cn)
            ax.set_ylim(0, 1)
            ax.set_title(f'seq={seq_len}, pred={pred_len}', fontsize=FONT_SIZE_PANEL - 1, fontfamily=CHINESE_FONT)
            for label in ax.get_yticklabels():
                label.set_fontproperties(fp_en)
    for ax in axes.flat:
        style_axis(ax)
    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[m], markersize=10, label=MODEL_NAMES_CN[m]) for m in MODEL_NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), prop=fp_cn, fontsize=FONT_SIZE_LEGEND - 1)
    plt.suptitle('各配置下四模型 R²（行=预测长度 pred，列=序列长度 seq）', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, y=1.02)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.90, hspace=0.45, wspace=0.35)
    plt.savefig(OUTPUT_DIR / 'fig06_tradeoff.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig06_tradeoff.png")


# -----------------------------------------------------------------------------
# 图7：仅贯入度，4×4 子图 — 每格一种(seq_len, pred_len)配置，格内真实值+四模型预测曲线
# 行=序列长度 seq，列=预测长度 pred；展示所有 4 模型在各自 16 种配置下的贯入度预测
# -----------------------------------------------------------------------------
def fig07_prediction_curves():
    start, end = SAMPLE_RANGE[0], SAMPLE_RANGE[1]
    feat_name, feat_idx, unit = '贯入度', 0, 'mm/min'
    fig, axes = plt.subplots(4, 4, figsize=(COL_WIDTH_INCH * 2.2, FIG_H_PANEL * 2.2))
    for row, seq_len in enumerate(SEQ_LENGTHS):
        for col, pred_len in enumerate(PRED_LENGTHS):
            ax = axes[row, col]
            first = None
            for m in MODEL_NAMES:
                d = load_pkl_data(m, seq_len, pred_len)
                if d is not None:
                    first = d
                    break
            if first is None:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes, fontfamily=CHINESE_FONT)
                ax.set_title(f'seq={seq_len}, pred={pred_len}', fontsize=FONT_SIZE_PANEL - 1, fontfamily=CHINESE_FONT)
                style_axis(ax)
                continue
            y_true = _get_series_from_pkl(first, 'y_true_inv', feat_idx, start, end)
            if y_true is None or len(y_true) == 0:
                ax.set_title(f'seq={seq_len}, pred={pred_len}', fontsize=FONT_SIZE_PANEL - 1, fontfamily=CHINESE_FONT)
                style_axis(ax)
                continue
            step = max(1, len(y_true) // 350)
            x_plot = np.arange(len(y_true))[::step]
            y_true_plot = y_true[::step]
            ax.plot(x_plot, y_true_plot, color=GREY_TRUE, linewidth=0.8, label='真实值', alpha=0.95)
            for model in MODEL_NAMES:
                d = load_pkl_data(model, seq_len, pred_len)
                if d is None:
                    continue
                y_pred = _get_series_from_pkl(d, 'y_pred_inv', feat_idx, start, end)
                if y_pred is not None and len(y_pred) == len(y_true):
                    ax.plot(x_plot, y_pred[::step], '-', linewidth=0.5, label=MODEL_NAMES_CN[model], color=COLORS[model], alpha=0.9)
            ax.set_title(f'seq={seq_len}, pred={pred_len}', fontsize=FONT_SIZE_PANEL - 1, fontfamily=CHINESE_FONT)
            ax.set_xlabel('样本', fontsize=FONT_SIZE_LABEL - 2, fontfamily=CHINESE_FONT)
            ax.set_ylabel('贯入度', fontsize=FONT_SIZE_LABEL - 2, fontfamily=CHINESE_FONT)
            if row == 0 and col == 0:
                ax.legend(fontsize=FONT_SIZE_LEGEND - 2, prop={'family': CHINESE_FONT}, loc='upper right')
            style_axis(ax)
    plt.suptitle('贯入度：4×4 配置（行=序列长度 seq，列=预测长度 pred），每格真实+四模型', fontsize=FONT_SIZE_LABEL, fontfamily=CHINESE_FONT, y=1.02)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.88, hspace=0.5, wspace=0.35)
    plt.savefig(OUTPUT_DIR / 'fig07_prediction_curves.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print("已生成: fig07_prediction_curves.png")


# =============================================================================
# 主函数（图8 不再绘制）
# =============================================================================
def main():
    print("=" * 60)
    print("论文图表生成脚本 (03-Analyses)")
    print("=" * 60)
    setup_fonts()
    df = load_data()
    print(f"已加载实验汇总: {len(df)} 条")

    fig01_dataset_only()
    fig02_accuracy(df)
    fig03_compute(df)
    fig04_3d_surface(df)
    fig05_heatmap(df)
    fig06_tradeoff(df)
    fig07_prediction_curves()
    # 图8 不再绘制

    print("\n" + "=" * 60)
    print(f"所有图片已保存至: {OUTPUT_DIR}")
    print("图1～图7（图8 已取消）")
    print("=" * 60)


if __name__ == '__main__':
    main()
