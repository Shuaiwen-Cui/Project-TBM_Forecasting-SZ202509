# -*- coding: utf-8 -*-
"""一键运行：生成所有图、表，并写入分析摘要（供论文引用）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, TABLES_DIR
from generate_figures import main as run_figures
from generate_tables import run as run_tables
from load_utils import load_summary_csv


def main():
    print("=" * 60)
    print("Transformer-TS 03-Analyses 一键分析")
    print("=" * 60)
    df = load_summary_csv()
    n = len(df)
    print(f"实验组数: {n}（Transformer seq_len × pred_len）")
    print()
    run_figures()
    print()
    run_tables()
    best = df.loc[df["R2"].idxmax()]
    worst = df.loc[df["R2"].idxmin()]
    summary_lines = [
        "Transformer-TS 分析摘要（供论文写作引用）",
        "=" * 50,
        "",
        f"共 {n} 组实验（seq_len × pred_len）。",
        "",
        f"最佳 R² = {best['R2']:.4f}，对应 seq_len={int(best['seq_len'])}, pred_len={int(best['pred_len'])}；"
        f"MSE = {best['MSE']:.6f}，训练时间 = {best['training_time_s']:.1f} s。",
        "",
        f"最差 R² = {worst['R2']:.4f}，对应 seq_len={int(worst['seq_len'])}, pred_len={int(worst['pred_len'])}。",
        "",
        "图表输出：",
        f"  图: {OUTPUT_DIR}",
        f"  表: {TABLES_DIR}",
        "",
        "图 1 数据集相关性；图 2 精度（R²/MSE/MAE/RMSE）；图 3 计算性能；图 4 R² 三维曲面；图 5 R² 热力图；图 6 精度-效率权衡；图 7 预测曲线。",
    ]
    summary_path = Path(__file__).resolve().parent / "分析摘要.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n已写入: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
