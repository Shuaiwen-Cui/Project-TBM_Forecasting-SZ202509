# -*- coding: utf-8 -*-
"""
Transformer-TS 论文图表生成：读取 02-Processing 结果，输出 fig01（图1）、fig03～fig11（图3～图11）至 figures/；图2 为 Transformer 架构图 transformer.png，不由此脚本生成。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
# 中文显示：优先宋体，否则微软雅黑/黑体（确保汉字能渲染）
_FONT_CANDIDATES = ["SimSun", "宋体", "Microsoft YaHei", "SimHei", "KaiTi", "FangSong", "DejaVu Sans"]
_available = [f.name for f in fm.fontManager.ttflist]
_CHINESE_FONT_RESOLVED = next((x for x in _FONT_CANDIDATES if any(x in a for a in _available)), "DejaVu Sans")
matplotlib.rcParams["font.sans-serif"] = [_CHINESE_FONT_RESOLVED] + [x for x in _FONT_CANDIDATES if x != _CHINESE_FONT_RESOLVED]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 中文用 FontProperties 指定备选族，避免系统未装 SimSun 时方框
FONT_CN = FontProperties(family=["SimSun", "宋体", "Microsoft YaHei", "SimHei", "sans-serif"])

from config import (
    DATA_PREPROCESSED,
    DPI,
    FIG_H_PANEL,
    FIG_H_SINGLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_PANEL,
    FONT_SIZE_TICK,
    CHINESE_FONT,
    ENGLISH_FONT,
    METADATA_COLS,
    OUTPUT_DIR,
    PRED_LENGTHS,
    REP_CONFIGS,
    SAMPLE_RANGE,
    SEQ_LENGTHS,
    COL_WIDTH_INCH,
    COL_WIDTH_DOUBLE,
    CMAP_CORR,
    CMAP_HEATMAP,
    GREY_TRUE,
    COLOR_BAR,
    COLOR_BAR_EDGE,
    COLOR_PRED,
    PREDICTION_CURVE_PRED_LENS,
    PREDICTION_FEATURES_BY_FIG,
)
from load_utils import load_summary_csv, load_pkl, get_series_from_pkl, get_feature_names_from_preprocessed


def setup_fonts():
    plt.rcParams["font.sans-serif"] = [_CHINESE_FONT_RESOLVED, "SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["font.serif"] = [ENGLISH_FONT, "Times", "DejaVu Serif"]
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = False
    sns.set_style("white", {"axes.edgecolor": ".15"})


def style_axis(ax, panel_label=None):
    ax.tick_params(axis="both", which="major", direction="out", labelsize=FONT_SIZE_TICK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(ENGLISH_FONT)
    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=FONT_SIZE_PANEL,
                fontweight="bold", va="top", ha="left", fontfamily=ENGLISH_FONT)


# -----------------------------------------------------------------------------
# 图1：数据集特征相关性（中文宋体+英文/数字 Times）
# -----------------------------------------------------------------------------
def fig01_dataset():
    n_features_total = 0
    if DATA_PREPROCESSED.exists():
        df = pd.read_csv(DATA_PREPROCESSED, low_memory=False)
        if len(df) > 1:
            first = df.iloc[0]
            if not pd.to_numeric(first, errors="coerce").notna().all():
                df = df.iloc[1:].reset_index(drop=True)
        numeric_cols = df.columns[METADATA_COLS:].tolist()
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df_num = df[numeric_cols].ffill().bfill()
        n_show = len(df_num.columns)
        n_features_total = len(numeric_cols)
    else:
        n_show = 0
    # 全维度热力图，图幅随特征数放大以便标注清晰
    fig_size = (max(COL_WIDTH_INCH * 1.2, n_show * 0.22), max(FIG_H_SINGLE * 1.3, n_show * 0.22))
    fig, ax = plt.subplots(figsize=fig_size)
    if DATA_PREPROCESSED.exists() and n_show > 0:
        sub = df_num.iloc[:, :n_show]
        corr = sub.corr()
        # 横纵轴只标特征序号 1, 2, ..., 不写 F
        labels = [str(i + 1) for i in range(n_show)]
        im = ax.imshow(corr, cmap=CMAP_CORR, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(n_show))
        ax.set_yticks(range(n_show))
        fp_tick = FontProperties(family=[ENGLISH_FONT, "Times", "DejaVu Sans"], size=FONT_SIZE_TICK - 2)
        fp_label = FontProperties(family=["SimSun", "宋体", "Microsoft YaHei", "SimHei", "sans-serif"], size=FONT_SIZE_LABEL + 2)
        ax.set_xticklabels(labels, fontproperties=fp_tick, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontproperties=fp_tick)
        ax.set_xlabel("参数", fontproperties=fp_label)
        ax.set_ylabel("参数", fontproperties=fp_label)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        for t in cbar.ax.get_yticklabels():
            t.set_fontfamily(ENGLISH_FONT)
    else:
        ax.text(0.5, 0.5, "data_preprocessed.csv 不存在", ha="center", va="center", transform=ax.transAxes, fontproperties=FONT_CN)
    ax.tick_params(axis="both", which="major", direction="out", labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig01_dataset.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    if n_features_total > 0:
        print(f"已生成: fig01_dataset.png（图1，特征总数: {n_features_total}）")
    else:
        print("已生成: fig01_dataset.png（图1）")


# -----------------------------------------------------------------------------
# 图2：精度指标（R²、MSE、MAE、RMSE）多行一列
# -----------------------------------------------------------------------------
def fig02_accuracy(df):
    config_labels = [f"{int(s)}-{int(p)}" for s, p in zip(df["seq_len"], df["pred_len"])]
    x = np.arange(len(config_labels))
    fig, axes = plt.subplots(4, 1, figsize=(COL_WIDTH_INCH * 1.15, FIG_H_PANEL * 4))
    for ax, (col, title) in zip(axes, [
        ("R2", "(a) 决定系数 R²"),
        ("MSE", "(b) 均方误差 MSE"),
        ("MAE", "(c) 平均绝对误差 MAE"),
        ("RMSE", "(d) 均方根误差 RMSE"),
    ]):
        vals = df[col].values
        ax.bar(x, vals, color=COLOR_BAR, edgecolor=COLOR_BAR_EDGE, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1, fontfamily=ENGLISH_FONT)
        ax.set_ylabel(col if col == "R2" else col, fontsize=FONT_SIZE_LABEL - 1, fontfamily=ENGLISH_FONT)
        ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
        if col == "R2":
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig03_accuracy.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("已生成: fig03_accuracy.png（图3）")


# -----------------------------------------------------------------------------
# 图3：计算性能（训练时间、推理时间、内存）
# -----------------------------------------------------------------------------
def fig03_compute(df):
    config_labels = [f"{int(s)}-{int(p)}" for s, p in zip(df["seq_len"], df["pred_len"])]
    x = np.arange(len(config_labels))
    fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH_INCH * 1.15, FIG_H_PANEL * 2.8))
    for ax, col, title in [
        (axes[0], "training_time_s", "(a) 训练时间 (s)"),
        (axes[1], "batch_inference_time_ms", "(b) 整测试集推理时间 (ms)"),
        (axes[2], "memory_usage_mb", "(c) 内存占用 (MB)"),
    ]:
        vals = np.maximum(df[col].values.astype(float), 1e-6)
        ax.bar(x, vals, color=COLOR_BAR, edgecolor=COLOR_BAR_EDGE, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha="right", fontsize=FONT_SIZE_TICK - 1, fontfamily=ENGLISH_FONT)
        ax.set_ylabel(title.split()[-1].strip("()"), fontsize=FONT_SIZE_LABEL - 1, fontfamily=ENGLISH_FONT)
        ax.set_title(title, fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
        style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig04_compute.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("已生成: fig04_compute.png（图4）")


# -----------------------------------------------------------------------------
# 图4：三维曲面 seq_len × pred_len → R²（缺数据用 NaN 填充后显示）
# -----------------------------------------------------------------------------
def fig04_3d_surface(df):
    from mpl_toolkits.mplot3d import Axes3D
    X = np.array(SEQ_LENGTHS)
    Y = np.array(PRED_LENGTHS)
    pivot = df.pivot_table(values="R2", index="pred_len", columns="seq_len", aggfunc="mean")
    pivot = pivot.reindex(PRED_LENGTHS).reindex(columns=SEQ_LENGTHS)
    ZZ = np.array(pivot.values, dtype=float)
    ZZ = np.nan_to_num(ZZ, nan=-0.3)
    XX, YY = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(COL_WIDTH_INCH * 1.6, FIG_H_PANEL * 1.9))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(XX, YY, ZZ, cmap=CMAP_HEATMAP, alpha=0.85, vmin=-0.5, vmax=1.0)
    ax.set_xlabel("序列长度 (步)", fontsize=FONT_SIZE_LABEL - 1, fontproperties=FONT_CN)
    ax.set_ylabel("预测长度 (步)", fontsize=FONT_SIZE_LABEL - 1, fontproperties=FONT_CN)
    ax.set_zlabel("R²", fontsize=FONT_SIZE_LABEL - 1, fontfamily=ENGLISH_FONT)
    ax.set_title("Transformer：R² 随序列长度与预测长度变化", fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
    for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        t.set_fontfamily(ENGLISH_FONT)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig05_3d_surface.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("已生成: fig05_3d_surface.png（图5）")


# -----------------------------------------------------------------------------
# 图5：R² 热力图 行=pred_len 列=seq_len
# -----------------------------------------------------------------------------
def fig05_heatmap(df):
    pivot = df.pivot_table(values="R2", index="pred_len", columns="seq_len", aggfunc="mean")
    pivot = pivot.reindex(PRED_LENGTHS).reindex(columns=SEQ_LENGTHS)
    vals = np.array(pivot.values, dtype=float)
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH * 1.15, FIG_H_PANEL * 1.25))
    vals_plot = np.nan_to_num(vals, nan=-0.5)
    im = ax.imshow(vals_plot, cmap=CMAP_HEATMAP, aspect="auto", vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(SEQ_LENGTHS)))
    ax.set_xticklabels([str(s) for s in SEQ_LENGTHS], fontfamily=ENGLISH_FONT)
    ax.set_yticks(range(len(PRED_LENGTHS)))
    ax.set_yticklabels([str(p) for p in PRED_LENGTHS], fontfamily=ENGLISH_FONT)
    for i in range(len(PRED_LENGTHS)):
        for j in range(len(SEQ_LENGTHS)):
            v = pivot.values[i, j]
            s = f"{v:.2f}" if np.isfinite(v) else "—"
            ax.text(j, i, s, ha="center", va="center", fontsize=FONT_SIZE_TICK - 1, fontfamily=ENGLISH_FONT)
    ax.set_xlabel("序列长度 (步)", fontsize=FONT_SIZE_LABEL, fontproperties=FONT_CN)
    ax.set_ylabel("预测长度 (步)", fontsize=FONT_SIZE_LABEL, fontproperties=FONT_CN)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="R²")
    cbar.ax.set_ylabel("R²", fontfamily=ENGLISH_FONT, fontsize=FONT_SIZE_LABEL)
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily(ENGLISH_FONT)
    style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig06_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("已生成: fig06_heatmap.png（图6）")


# -----------------------------------------------------------------------------
# 图6：精度-效率权衡 R² vs 训练时间（颜色=预测长度，形状=序列长度，图例 4+5 项）
# -----------------------------------------------------------------------------
# 预测长度 1, 6, 120, 360 对应颜色；序列长度 6, 30, 60, 120, 360 对应标记形状
_PRED_LEN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_SEQ_LEN_MARKERS = ["o", "s", "^", "D", "v"]

def fig06_tradeoff(df):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_INCH * 1.3, FIG_H_PANEL * 1.35))
    x = df["training_time_s"].values.astype(float)
    y = df["R2"].values.astype(float)
    seq_lens = df["seq_len"].values.astype(int)
    pred_lens = df["pred_len"].values.astype(int)
    pred_len_to_color = {p: _PRED_LEN_COLORS[i] for i, p in enumerate(PRED_LENGTHS)}
    seq_len_to_marker = {s: _SEQ_LEN_MARKERS[i] for i, s in enumerate(SEQ_LENGTHS)}
    for i in range(len(df)):
        c = pred_len_to_color[pred_lens[i]]
        m = seq_len_to_marker[seq_lens[i]]
        ax.scatter(x[i], y[i], c=c, edgecolors="white", linewidths=0.5, s=72, marker=m, alpha=0.9, zorder=3)
    # 图例：预测长度（颜色） + 序列长度（形状），放在图框下方、图外
    pred_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_PRED_LEN_COLORS[i], markeredgecolor="white", markersize=7, label=f"预测长度 {p}") for i, p in enumerate(PRED_LENGTHS)]
    seq_handles = [plt.Line2D([0], [0], marker=_SEQ_LEN_MARKERS[i], color="w", markerfacecolor="gray", markeredgecolor="white", markersize=6, label=f"序列长度 {s}") for i, s in enumerate(SEQ_LENGTHS)]
    ax.set_xlabel("训练时间 (s)", fontsize=FONT_SIZE_LABEL, fontproperties=FONT_CN)
    ax.set_ylabel("R²", fontsize=FONT_SIZE_LABEL, fontfamily=ENGLISH_FONT)
    ax.set_title("精度-效率权衡", fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
    style_axis(ax)
    plt.tight_layout()
    # 为图例预留底部空间，图例置于图框下方（图外），横向宽度与主图对齐
    fig.subplots_adjust(bottom=0.26)
    # bbox_to_anchor=(0, y, 1, h) 表示图例占满主图横向范围 [0,1]，mode="expand" 使图例框横向拉满
    leg = ax.legend(handles=pred_handles + seq_handles, loc="upper center", bbox_to_anchor=(0, -0.32, 1, 0.12), ncol=2, mode="expand", fontsize=FONT_SIZE_TICK - 1, prop=FONT_CN, framealpha=0.92, columnspacing=1.2, handletextpad=1)
    plt.savefig(OUTPUT_DIR / "fig07_tradeoff.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("已生成: fig07_tradeoff.png（图7）")


# -----------------------------------------------------------------------------
# 图7～10：固定一个物理量（贯入度），固定预测长度，五子图 = 不同输入长度 (6, 30, 60, 120, 360)
# 图7: pred_len=1；图8: pred_len=6；图9: pred_len=120；图10: pred_len=360
# -----------------------------------------------------------------------------
def _draw_prediction_curves_fixed_pred_len(feat_idx, feat_name, pred_len, figpath, suptitle_prefix):
    """固定 pred_len，依次展示 seq_len=6,30,60,120,360 五个子图的真实值 vs 预测值。"""
    start, end = SAMPLE_RANGE[0], SAMPLE_RANGE[1]
    n = len(SEQ_LENGTHS)
    fig, axes = plt.subplots(n, 1, figsize=(COL_WIDTH_INCH * 1.15, FIG_H_PANEL * n * 0.95))
    if n == 1:
        axes = [axes]
    for ax, seq_len in zip(axes, SEQ_LENGTHS):
        d = load_pkl(seq_len, pred_len)
        if d is None:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes, fontproperties=FONT_CN)
            ax.set_title(f"输入长度 = {seq_len}", fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
            style_axis(ax)
            continue
        # 纵轴标题统一用当前图对应的特征名 feat_name（与 PREDICTION_FEATURES_BY_FIG 一致）
        ylabel = feat_name
        y_true = get_series_from_pkl(d, "y_true_inv", feat_idx, start, end)
        y_pred = get_series_from_pkl(d, "y_pred_inv", feat_idx, start, end)
        if y_true is None or y_pred is None or len(y_true) == 0:
            ax.set_title(f"输入长度 = {seq_len}", fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
            style_axis(ax)
            continue
        step = max(1, len(y_true) // 400)
        x_plot = np.arange(len(y_true))[::step]
        ax.plot(x_plot, y_true[::step], color=GREY_TRUE, linewidth=1, label="真实值", alpha=0.95)
        ax.plot(x_plot, y_pred[::step], "-", color=COLOR_PRED, linewidth=0.8, label="预测值", alpha=0.9)
        ax.set_xlabel("样本序号", fontsize=FONT_SIZE_LABEL - 1, fontproperties=FONT_CN)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL - 1, fontproperties=FONT_CN)
        ax.set_title(f"输入长度 = {seq_len}", fontsize=FONT_SIZE_PANEL, fontproperties=FONT_CN)
        ax.legend(fontsize=FONT_SIZE_LEGEND - 1, prop=FONT_CN, loc="lower right")
        style_axis(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / figpath, dpi=DPI, bbox_inches="tight")
    plt.close()


def fig07_prediction_curves():
    """图8：贯入度，预测长度=1"""
    feat_idx, feat_name = PREDICTION_FEATURES_BY_FIG[0]
    pred_len = PREDICTION_CURVE_PRED_LENS[0]
    _draw_prediction_curves_fixed_pred_len(
        feat_idx, feat_name, pred_len,
        "fig08_penetration.png",
        f"{feat_name}：真实值与 Transformer 预测值",
    )
    print("已生成: fig08_penetration.png（图8）")


def fig08_prediction_curves():
    """图9：推进油缸总推力，预测长度=6"""
    feat_idx, feat_name = PREDICTION_FEATURES_BY_FIG[1]
    pred_len = PREDICTION_CURVE_PRED_LENS[1]
    _draw_prediction_curves_fixed_pred_len(
        feat_idx, feat_name, pred_len,
        "fig09_thrust.png",
        f"{feat_name}：真实值与 Transformer 预测值",
    )
    print("已生成: fig09_thrust.png（图9）")


def fig09_prediction_curves():
    """图10：刀盘扭矩，预测长度=120"""
    feat_idx, feat_name = PREDICTION_FEATURES_BY_FIG[2]
    pred_len = PREDICTION_CURVE_PRED_LENS[2]
    _draw_prediction_curves_fixed_pred_len(
        feat_idx, feat_name, pred_len,
        "fig10_torque.png",
        f"{feat_name}：真实值与 Transformer 预测值",
    )
    print("已生成: fig10_torque.png（图10）")


def fig10_prediction_curves():
    """图11：刀盘转速，预测长度=360"""
    feat_idx, feat_name = PREDICTION_FEATURES_BY_FIG[3]
    pred_len = PREDICTION_CURVE_PRED_LENS[3]
    _draw_prediction_curves_fixed_pred_len(
        feat_idx, feat_name, pred_len,
        "fig11_penetration_pred360.png",
        f"{feat_name}：真实值与 Transformer 预测值",
    )
    print("已生成: fig11_penetration_pred360.png（图11）")


def main():
    setup_fonts()
    print("已加载实验汇总...")
    df = load_summary_csv()
    print(f"  {len(df)} 条 (Transformer)")
    fig01_dataset()
    fig02_accuracy(df)
    fig03_compute(df)
    fig04_3d_surface(df)
    fig05_heatmap(df)
    fig06_tradeoff(df)
    fig07_prediction_curves()
    fig08_prediction_curves()
    fig09_prediction_curves()
    fig10_prediction_curves()
    print(f"\n所有图片已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
