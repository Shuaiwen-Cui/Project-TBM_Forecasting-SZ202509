#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# === Configure matplotlib to support Chinese ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# Config
CSV_PATH = "data_preprocessed.csv"
OUT_DIR = "plots"

def normalize_header(h: str) -> str:
    """Remove leading [number] prefix and trim spaces."""
    return re.sub(r"^\[\d+\]\s*", "", str(h)).strip()

def guess_time_column(headers: list[str], max_check: int = 5) -> int | None:
    """Guess a time-like column index within first max_check columns."""
    keywords = ["时间", "timestamp", "time", "date", "datetime", "日期"]
    for i, h in enumerate(headers[:max_check]):
        hn = normalize_header(h).lower()
        if any(kw in hn for kw in keywords):
            return i
    return None

def safe_filename(s: str) -> str:
    """Make a safe filename from column name."""
    s = normalize_header(s)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    return s.strip()[:80]

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}")
        return

    # Load header
    head_df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", nrows=0)
    headers = list(head_df.columns)
    n_cols = len(headers)

    # Auto-detect x-axis column (in first 5 cols)
    x_idx = guess_time_column(headers)

    # Read all columns (we'll use index or detected time as X)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    # Normalize headers (remove [index] prefix)
    df.rename(columns={c: normalize_header(c) for c in df.columns}, inplace=True)
    norm_headers = [normalize_header(h) for h in headers]

    # Prepare x-axis
    if x_idx is not None:
        x_name = normalize_header(headers[x_idx])
        # Try to parse time-like column; if parsing fails it will remain as original
        x = pd.to_datetime(df[x_name], errors="ignore", infer_datetime_format=True)
        x_label = f"[{x_idx}] {x_name}"
    else:
        x = df.index
        x_label = "Index"

    os.makedirs(OUT_DIR, exist_ok=True)

    # Plot from column 6 (index = 5) to the last
    for idx in range(5, n_cols):
        col_name = norm_headers[idx]
        if col_name not in df.columns:
            print(f"WARNING: Column [{idx}] {col_name} not found.")
            continue

        y = pd.to_numeric(df[col_name], errors="coerce")

        plt.figure()
        plt.plot(x, y, label=f"[{idx}] {col_name}")
        plt.xlabel(x_label)
        plt.ylabel(col_name)
        plt.title(f"列 [{idx}] {col_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fn = os.path.join(OUT_DIR, f"{idx}_{safe_filename(col_name)}.png")
        plt.savefig(fn, dpi=150)
        plt.close()  # IMPORTANT: close figure to free memory
        print(f"图像已保存: {fn}")

    # No plt.show() to avoid keeping many figures open

if __name__ == "__main__":
    main()
