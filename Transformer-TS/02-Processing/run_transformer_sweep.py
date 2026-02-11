# -*- coding: utf-8 -*-
"""
仅训练 Transformer 模型：在 5×4=20 种 (seq_len, pred_len) 组合下进行时序预测。
- 数据：01-Preprocessing/data_preprocessed.csv
- 支持接续训练：默认只训练未完成的组合；--overwrite / -f 覆盖已有结果。
- 输出进度提示与结果到 results/。
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 保证从 02-Processing 目录可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_FILE,
    INFERENCE_BATCH_SIZE,
    MODEL_NAME,
    PRED_LENGTHS,
    RESULTS_DIR,
    SEQ_LENGTHS,
    STATUS_FILE,
    SUMMARY_CSV,
    TRANSFORMER_CONFIG,
)
from data_loader import load_dataset
from models import TransformerForecaster
from utils import compute_metrics, count_parameters


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one(seq_len, pred_len, overwrite=False):
    """训练单组 (seq_len, pred_len)，返回结果字典与是否跳过。"""
    task_id = f"{MODEL_NAME}_{seq_len}_{pred_len}"
    pkl_path = RESULTS_DIR / f"{task_id}_results.pkl"

    if not overwrite and pkl_path.exists():
        return None, True  # 跳过

    # 加载数据
    ds = load_dataset(seq_len, pred_len)
    (X_train, y_train), (_X_val, _y_val), (X_test, y_test) = (
        ds["train"],
        ds["val"],
        ds["test"],
    )
    scaler = ds["scaler"]
    feature_names = ds["feature_names"]
    n_features = X_train.shape[2]

    device = get_device()
    cfg = TRANSFORMER_CONFIG
    model = TransformerForecaster(
        n_features=n_features,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = torch.nn.MSELoss()

    # 训练
    batch_size = cfg["batch_size"]
    n_train = X_train.shape[0]
    t0 = time.perf_counter()
    for _ in range(cfg["epochs"]):
        perm = np.random.permutation(n_train)
        model.train()
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
    training_time_s = time.perf_counter() - t0

    # 测试集预测（分批推理，避免 seq_len 大时 OOM）
    model.eval()
    n_test = X_test.shape[0]
    batch_infer = min(INFERENCE_BATCH_SIZE, n_test)
    if seq_len >= 360:
        batch_infer = min(batch_infer, 8)
    elif seq_len >= 120:
        batch_infer = min(batch_infer, 16)
    pred_list = []
    with torch.no_grad():
        for start in range(0, n_test, batch_infer):
            end = min(start + batch_infer, n_test)
            xb = torch.from_numpy(X_test[start:end]).to(device)
            pred_list.append(model(xb).cpu().numpy())
    y_pred = np.concatenate(pred_list, axis=0)
    y_true = y_test
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, n_features)).reshape(y_true.shape)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, n_features)).reshape(y_pred.shape)

    metrics = compute_metrics(y_true, y_pred)
    n_params = count_parameters(model)
    model_size_mb = n_params * 4 / (1024 * 1024)  # float32

    # 推理时间：单样本与分批整测试集
    with torch.no_grad():
        x1 = torch.from_numpy(X_test[:1]).to(device)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time_ms = (time.perf_counter() - t0) / 100 * 1000
        t0 = time.perf_counter()
        for start in range(0, n_test, batch_infer):
            end = min(start + batch_infer, n_test)
            _ = model(torch.from_numpy(X_test[start:end]).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_inference_time_ms = (time.perf_counter() - t0) * 1000

    try:
        import psutil
        proc = psutil.Process()
        memory_usage_mb = proc.memory_info().rss / (1024 * 1024)
    except (ImportError, AttributeError):
        memory_usage_mb = 0.0

    result = {
        "model_name": MODEL_NAME,
        "feature_names": feature_names,
        "metrics": metrics,
        "costs": {
            "model_size_params": n_params,
            "model_size_mb": model_size_mb,
            "inference_time_ms": inference_time_ms,
            "batch_inference_time_ms": batch_inference_time_ms,
            "memory_usage_mb": memory_usage_mb,
            "training_time_s": training_time_s,
        },
        "config": {"seq_len": seq_len, "pred_len": pred_len, **cfg},
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_inv": y_true_inv,
        "y_pred_inv": y_pred_inv,
        "scaler": scaler,
        "feature_mapping": {i: name for i, name in enumerate(feature_names)},
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)

    # 汇总行
    row = {
        "model": MODEL_NAME,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "MSE": metrics["MSE"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "model_size_params": n_params,
        "model_size_mb": model_size_mb,
        "inference_time_ms": inference_time_ms,
        "batch_inference_time_ms": batch_inference_time_ms,
        "memory_usage_mb": memory_usage_mb,
        "training_time_s": training_time_s,
    }
    return row, False


def update_summary_csv(rows):
    """将 Transformer 相关行写入或合并到 experiment_summary.csv。"""
    cols = [
        "model", "seq_len", "pred_len", "MSE", "MAE", "RMSE", "MAPE", "R2",
        "model_size_params", "model_size_mb", "inference_time_ms",
        "batch_inference_time_ms", "memory_usage_mb", "training_time_s",
    ]
    new_df = pd.DataFrame(rows, columns=cols)
    if SUMMARY_CSV.exists():
        old = pd.read_csv(SUMMARY_CSV)
        old = old[old["model"] != MODEL_NAME]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(SUMMARY_CSV, index=False)


def update_status(tasks_done):
    """更新 experiment_status.json 中 Transformer 任务状态。"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            status = json.load(f)
    else:
        status = {"tasks": {}, "last_update": "", "total": 0, "completed": 0, "failed": 0, "pending": 0}
    for task_id, row in tasks_done:
        status["tasks"][task_id] = {
            "model": MODEL_NAME,
            "seq_len": row["seq_len"],
            "pred_len": row["pred_len"],
            "status": "completed",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
    status["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Transformer 时序预测：20 种 seq_len × pred_len 组合")
    parser.add_argument("--overwrite", "-f", action="store_true", help="覆盖已有结果，全部重新训练")
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"错误：数据文件不存在 {DATA_FILE}")
        sys.exit(1)

    tasks = [(s, p) for s in SEQ_LENGTHS for p in PRED_LENGTHS]
    total = len(tasks)
    print(f"数据文件: {DATA_FILE}")
    print(f"组合数: {total} (seq_len={SEQ_LENGTHS}, pred_len={PRED_LENGTHS})")
    print(f"模式: {'覆盖已有，全部训练' if args.overwrite else '接续训练（跳过已完成）'}")
    print("-" * 60)

    done_rows = []
    skipped = 0
    for i, (seq_len, pred_len) in enumerate(tasks, 1):
        print(f"[{i:2d}/{total}] seq_len={seq_len:3d} pred_len={pred_len:3d} ", end="", flush=True)
        try:
            row, skipped_this = train_one(seq_len, pred_len, overwrite=args.overwrite)
            if skipped_this:
                print("(已存在，跳过)")
                skipped += 1
                continue
            done_rows.append((f"{MODEL_NAME}_{seq_len}_{pred_len}", row))
            print(f"完成 R2={row['R2']:.4f} MSE={row['MSE']:.6f} 训练={row['training_time_s']:.1f}s")
        except Exception as e:
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()

    if done_rows:
        update_summary_csv([r[1] for r in done_rows])
        update_status(done_rows)
        print("-" * 60)
        print(f"本次完成: {len(done_rows)} 组，跳过: {skipped} 组。汇总已写入 {SUMMARY_CSV}")
    else:
        print("-" * 60)
        print(f"无新完成任务（跳过 {skipped} 组）。")


if __name__ == "__main__":
    main()
