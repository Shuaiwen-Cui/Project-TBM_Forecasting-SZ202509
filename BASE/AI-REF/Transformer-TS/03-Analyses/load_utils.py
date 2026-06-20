# -*- coding: utf-8 -*-
"""读取 02-Processing 的 CSV 与 PKL，统一接口。支持从 PKL 补全汇总表。"""
import pickle
from pathlib import Path

import pandas as pd

from config import CSV_FILE, DATA_PREPROCESSED, METADATA_COLS, MODEL_NAME, PKL_DIR, PRED_LENGTHS, SEQ_LENGTHS


def _compute_metrics(y_true, y_pred):
    """从 y_true/y_pred 计算 MSE/MAE/RMSE/MAPE/R2，与 02-Processing/utils 一致。"""
    import numpy as np
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = (1.0 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def build_summary_from_pkl():
    """从所有 Transformer_*_results.pkl 构建与 experiment_summary.csv 同结构的 DataFrame。"""
    rows = []
    for seq_len in SEQ_LENGTHS:
        for pred_len in PRED_LENGTHS:
            d = load_pkl(seq_len, pred_len)
            if d is None:
                continue
            m = d.get("metrics")
            if m is None and d.get("y_true") is not None and d.get("y_pred") is not None:
                m = _compute_metrics(d["y_true"], d["y_pred"])
            m = m or {}
            c = d.get("costs") or {}
            rows.append({
                "model": MODEL_NAME,
                "seq_len": seq_len,
                "pred_len": pred_len,
                "MSE": m.get("MSE", float("nan")),
                "MAE": m.get("MAE", float("nan")),
                "RMSE": m.get("RMSE", float("nan")),
                "MAPE": m.get("MAPE", float("nan")),
                "R2": m.get("R2", float("nan")),
                "model_size_params": c.get("model_size_params", 0),
                "model_size_mb": c.get("model_size_mb", 0),
                "inference_time_ms": c.get("inference_time_ms", 0),
                "batch_inference_time_ms": c.get("batch_inference_time_ms", 0),
                "memory_usage_mb": c.get("memory_usage_mb", 0),
                "training_time_s": c.get("training_time_s", 0),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["seq_len", "pred_len"]).reset_index(drop=True)


def load_summary_csv():
    """读取 experiment_summary.csv（仅 Transformer）；若行数不足 20 则用 PKL 补全后返回。"""
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        df = df[df["model"] == MODEL_NAME].copy()
        if len(df) >= len(SEQ_LENGTHS) * len(PRED_LENGTHS):
            return df.sort_values(["seq_len", "pred_len"]).reset_index(drop=True)
    df_pkl = build_summary_from_pkl()
    if len(df_pkl) > 0:
        return df_pkl
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        df = df[df["model"] == MODEL_NAME].copy()
        return df.sort_values(["seq_len", "pred_len"]).reset_index(drop=True)
    raise FileNotFoundError(f"无可用结果: {CSV_FILE} 且无 PKL 可读")


def load_pkl(seq_len, pred_len):
    """读取 Transformer_{seq_len}_{pred_len}_results.pkl，返回字典。"""
    pkl_path = PKL_DIR / f"{MODEL_NAME}_{seq_len}_{pred_len}_results.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_metrics_row(df, seq_len, pred_len):
    """从 DataFrame 中取指定 (seq_len, pred_len) 的一行。"""
    r = df[(df["seq_len"] == seq_len) & (df["pred_len"] == pred_len)]
    return r.iloc[0] if len(r) > 0 else None


def get_series_from_pkl(data, key, feat_idx, start, end):
    """从 pkl 的 y_true_inv 或 y_pred_inv 取单特征序列。形状可能 (N, pred_len, F) 或 (N, F)。"""
    y = data.get(key)
    if y is None:
        return None
    import numpy as np
    y = np.asarray(y)
    if y.ndim == 3:
        y = y[:, 0, :]
    return y[start:end, feat_idx].flatten()


def get_feature_names_from_preprocessed(max_len=20):
    """从 data_preprocessed.csv 读取特征列名（跳过前 METADATA_COLS 列），用于图表标签。"""
    if not DATA_PREPROCESSED.exists():
        return None
    df = pd.read_csv(DATA_PREPROCESSED, nrows=0)
    cols = df.columns[METADATA_COLS:].tolist()
    if max_len and max_len > 0:
        cols = [c[:max_len] if len(c) > max_len else c for c in cols]
    return cols
