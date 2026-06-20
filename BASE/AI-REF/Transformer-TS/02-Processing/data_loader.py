# -*- coding: utf-8 -*-
"""
从 data_preprocessed.csv 加载数据，构建 (seq_len, pred_len) 的时序样本。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from config import DATA_FILE, METADATA_COLS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def load_raw_data():
    """读取 CSV，跳过单位行，只保留数值列。"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    # 若第一行是单位行（非数值），去掉
    if df.shape[0] > 0:
        first_row = df.iloc[0]
        try:
            first_row_numeric = pd.to_numeric(first_row, errors="coerce").notna().all()
        except Exception:
            first_row_numeric = False
        if not first_row_numeric:
            df = df.iloc[1:].reset_index(drop=True)
    # 从第 METADATA_COLS 列起为特征
    feature_cols = df.columns[METADATA_COLS:].tolist()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    data = df[feature_cols].ffill().bfill().values.astype(np.float32)
    return data, feature_cols


def build_sequences(data, seq_len, pred_len):
    """
    data: (T, F)，构建 X (N, seq_len, F), y (N, pred_len, F)。
    """
    T, F = data.shape
    need = seq_len + pred_len
    if T < need:
        return None, None
    X_list, y_list = [], []
    for i in range(T - need + 1):
        X_list.append(data[i : i + seq_len])
        y_list.append(data[i + seq_len : i + need])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def get_splits(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """按比例划分训练/验证/测试，不打乱顺序（时序）。"""
    n = len(X)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    X_train, X_val, X_test = X[:t1], X[t1:t2], X[t2:]
    y_train, y_val, y_test = y[:t1], y[t1:t2], y[t2:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_dataset(seq_len, pred_len):
    """
    加载数据并生成指定 seq_len / pred_len 的数据集。
    返回:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) 均为 numpy
        scaler: 在 X_train 上拟合的 StandardScaler（按特征维度）
        feature_names: 特征列名列表
    """
    data, feature_names = load_raw_data()
    X, y = build_sequences(data, seq_len, pred_len)
    if X is None:
        raise ValueError(f"数据长度不足: seq_len={seq_len}, pred_len={pred_len}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X, y)

    # 用训练集拟合 scaler：按特征维度归一化（reshape 成 (N*L, F)）
    n_train, L, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))

    def scale_arr(arr):
        orig = arr.shape
        flat = arr.reshape(-1, F)
        out = scaler.transform(flat)
        return out.reshape(orig).astype(np.float32)

    X_train = scale_arr(X_train)
    X_val = scale_arr(X_val)
    X_test = scale_arr(X_test)
    y_train = scale_arr(y_train)
    y_val = scale_arr(y_val)
    y_test = scale_arr(y_test)

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "scaler": scaler,
        "feature_names": feature_names,
    }
