import re
import os
import pandas as pd

# Paths
CSV_PATH = "data/suzhoudata.csv"
OUT_TXT  = "check_result.txt"
OUT_CSV  = "data_preprocessed.csv"

# Target headers to verify & extract (order defines the output order AFTER the first 5 columns)
TARGET_HEADERS = [
    "贯入度",
    "推进区间的压力（上）",
    "推进区间的压力（右）",
    "推进区间的压力（下）",
    "推进区间的压力（左）",
    # "土舱土压（右上）",
    "土舱土压（右下）",
    # "土舱土压（左上）",
    "土舱土压（左下）",
    "No.16推进千斤顶速度",
    "No.4推进千斤顶速度",
    "No.8推进千斤顶速度",
    "No.12推进千斤顶速度",
    "推进油缸总推力",
    "No.16推进千斤顶行程",
    "No.4推进千斤顶行程",
    "No.8推进千斤顶行程",
    "No.12推进千斤顶行程",
    "千斤顶行程差 上下",
    "推进平均速度",
    "刀盘转速",
    "刀盘扭矩",
    "No.1刀盘电机扭矩",
    "No.2刀盘电机扭矩",
    "No.3刀盘电机扭矩",
    "No.4刀盘电机扭矩",
    "No.5刀盘电机扭矩",
    "No.6刀盘电机扭矩",
    "No.7刀盘电机扭矩",
    "No.8刀盘电机扭矩",
    "No.9刀盘电机扭矩",
    "No.10刀盘电机扭矩",
]

def normalize(header: str) -> str:
    """Remove leading [index] like [137] and trim whitespace."""
    return re.sub(r"^\[\d+\]\s*", "", header).strip()

def main():
    if not os.path.exists(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        return

    # Read header only to build matching maps
    df_head = pd.read_csv(CSV_PATH, encoding="utf-8-sig", nrows=0)
    raw_headers = list(df_head.columns)

    # Build maps: normalized -> original, original -> position
    norm_map = {normalize(h): h for h in raw_headers}
    pos_map  = {h: i for i, h in enumerate(raw_headers)}

    # ------------- Part A: check & collect matches -------------
    results = []
    matches = []  # (target, original_header, normalized_name, position)

    for target in TARGET_HEADERS:
        status = "Not Found"
        matched_orig = ""
        pos = ""
        # exact normalized match
        if target in norm_map:
            matched_orig = norm_map[target]
            pos = pos_map[matched_orig]
            status = "Found"
        else:
            # fallback: compact spaces and do substring search on normalized names
            tgt_compact = re.sub(r"\s+", "", target)
            for norm, orig in norm_map.items():
                if tgt_compact in re.sub(r"\s+", "", norm):
                    matched_orig = orig
                    pos = pos_map[matched_orig]
                    status = "Found"
                    break

        results.append((target, status, matched_orig, pos))
        if status == "Found":
            matches.append((target, matched_orig, normalize(matched_orig), pos))

    # Pretty, fixed-width output
    w1, w2, w3, w4 = 22, 10, 36, 10
    header_line = f"{'Target'.ljust(w1)}{'Status'.ljust(w2)}{'Matched Header'.ljust(w3)}{'Position'.ljust(w4)}"
    print("=== Checking Results ===")
    print(header_line)
    print("-" * (w1 + w2 + w3 + w4))
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=== Header Check Result ===\n")
        f.write(header_line + "\n")
        f.write("-" * (w1 + w2 + w3 + w4) + "\n")
        for target, status, name, pos in results:
            line = f"{target.ljust(w1)}{status.ljust(w2)}{str(name).ljust(w3)}{str(pos).ljust(w4)}"
            print(line)
            f.write(line + "\n")

    # ------------- Part B: build output column order -------------
    if not matches:
        print("No target headers matched. Preprocessing skipped.")
        return

    # (1) Keep the FIRST 5 columns from the original file (in original order)
    first5_orig = raw_headers[:5]
    first5_new  = [normalize(h) for h in first5_orig]

    # (2) Then append the matched target columns following TARGET_HEADERS order
    ordered_orig = []
    ordered_new  = []
    seen = set()

    # Start with first 5
    for orig, new in zip(first5_orig, first5_new):
        if orig not in seen:
            ordered_orig.append(orig)
            ordered_new.append(new)
            seen.add(orig)

    # Then append requested variables (in the order of TARGET_HEADERS)
    for target in TARGET_HEADERS:
        for t, orig, new_name, _ in matches:
            if t == target and orig not in seen:
                ordered_orig.append(orig)
                ordered_new.append(new_name)
                seen.add(orig)
                break

    # ------------- Part C: read, rename, reorder, save -------------
    # Read only needed columns; low_memory=False reduces mixed-type warnings.
    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig",
        usecols=ordered_orig,
        low_memory=False
    )

    # Rename to normalized names (remove [index] prefix)
    rename_map = {orig: new for orig, new in zip(ordered_orig, ordered_new)}
    df.rename(columns=rename_map, inplace=True)

    # Force final column order
    df = df[ordered_new]

    # Save to current directory
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Preprocessed file saved: {OUT_CSV}")
    print(f"Columns kept ({len(df.columns)}): {list(df.columns)}")

if __name__ == "__main__":
    main()
