# -*- coding: utf-8 -*-
"""从 experiment_summary 生成 tables/ 下的 CSV、LaTeX 表与最佳/最差配置摘要。"""
import pandas as pd
from pathlib import Path

from config import TABLES_DIR
from load_utils import load_summary_csv


def _to_latex_table(df):
    """生成适合论文的 LaTeX 表格（保留 seq_len, pred_len, R2, MSE, MAE, RMSE, training_time_s）。"""
    cols = ["seq_len", "pred_len", "R2", "MSE", "MAE", "RMSE", "training_time_s"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    sub["seq_len"] = sub["seq_len"].astype(int)
    sub["pred_len"] = sub["pred_len"].astype(int)
    sub["R2"] = sub["R2"].round(4)
    for c in ["MSE", "MAE", "RMSE"]:
        if c in sub.columns:
            sub[c] = sub[c].round(6)
    if "training_time_s" in sub.columns:
        sub["training_time_s"] = sub["training_time_s"].round(1)
    header = " & ".join(["序列长度", "预测长度", "$R^2$", "MSE", "MAE", "RMSE", "训练时间(s)"][: len(cols)]) + " \\\\"
    lines = ["\\begin{table}[htbp]", "\\centering", "\\caption{Transformer 多时间尺度实验指标汇总}", "\\label{tab:transformer-metrics}"]
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\hline")
    lines.append(header)
    lines.append("\\hline")
    for _, row in sub.iterrows():
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def run():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summary_csv()
    # 完整指标表 CSV
    out_csv = TABLES_DIR / "table_metrics.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已生成: {out_csv}")

    # LaTeX 表
    try:
        latex_str = _to_latex_table(df)
        out_tex = TABLES_DIR / "table_metrics.tex"
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"已生成: {out_tex}")
    except Exception as e:
        print(f"LaTeX 表跳过: {e}")

    # 按 R² 排序，输出最佳/最差各 5 组及一句结论
    df_sorted = df.sort_values("R2", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    top_k = min(5, n)
    lines = [
        "Transformer-TS 实验配置：按 R² 排序",
        "=" * 55,
        "",
        "【R² 最佳 5 组】",
    ]
    for i in range(top_k):
        r = df_sorted.iloc[i]
        lines.append(f"  {i+1}. seq_len={int(r['seq_len'])}, pred_len={int(r['pred_len'])}  R²={r['R2']:.4f}  MSE={r['MSE']:.6f}  MAE={r['MAE']:.4f}  训练时间={r['training_time_s']:.1f}s")
    lines.append("")
    lines.append("【R² 最差 5 组】")
    for i in range(max(0, n - 5), n):
        r = df_sorted.iloc[i]
        lines.append(f"  seq_len={int(r['seq_len'])}, pred_len={int(r['pred_len'])}  R²={r['R2']:.4f}  MSE={r['MSE']:.6f}  MAE={r['MAE']:.4f}  训练时间={r['training_time_s']:.1f}s")
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]
    lines.extend([
        "",
        "【一句结论】",
        f"最佳: seq_len={int(best['seq_len'])}, pred_len={int(best['pred_len'])}  R²={best['R2']:.4f}  MSE={best['MSE']:.6f}",
        f"最差: seq_len={int(worst['seq_len'])}, pred_len={int(worst['pred_len'])}  R²={worst['R2']:.4f}  MSE={worst['MSE']:.6f}",
    ])
    out_txt = TABLES_DIR / "table_best_worst.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成: {out_txt}")


if __name__ == "__main__":
    run()
