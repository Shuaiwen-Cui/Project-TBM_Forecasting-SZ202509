import os
import re
from pathlib import Path

# Directory containing the CSV files (use the script's directory)
base_dir = Path(__file__).parent

# Regex to match files like: TL250717_85.Csv, TL250724_179.Csv, ...
pattern = re.compile(r"^TL\d{6}_(\d+)\.Csv$")

def numeric_suffix(fname: str) -> int:
    """Extract the numeric suffix after '_' and before '.Csv'."""
    m = pattern.match(fname)
    return int(m.group(1)) if m else -1

# Collect matching files and sort by numeric suffix
candidates = [p for p in base_dir.iterdir() if p.is_file() and pattern.match(p.name)]
if not candidates:
    print("No matching files found (pattern: TLxxxxxx_<number>.Csv).")
    raise SystemExit(1)

files_sorted = sorted(candidates, key=lambda p: numeric_suffix(p.name))

# OPTIONAL: restrict to a suffix range (uncomment if needed)
# files_sorted = [p for p in files_sorted if 85 <= numeric_suffix(p.name) <= 215]

print("Files to merge (in order):")
for p in files_sorted:
    print(f"  - {p.name}")

output_path = base_dir / "suzhoudata.csv"

# Merge logic: keep the first two lines from the first file only
first_two_lines_ref = None
total_written = 0

with open(output_path, "w", encoding="utf-8-sig", newline="") as fout:
    for idx, path in enumerate(files_sorted):
        with open(path, "r", encoding="utf-8-sig", newline="") as fin:
            lines = fin.readlines()

        if idx == 0:
            # Write the entire first file (including its first two lines)
            fout.writelines(lines)
            total_written += len(lines)
            # Save the first two lines for optional validation against later files
            first_two_lines_ref = lines[:2] if len(lines) >= 2 else lines[:]
            print(f"[OK] {path.name}: wrote full file ({len(lines)} lines).")
        else:
            if len(lines) < 2:
                print(f"[WARN] {path.name}: less than 2 lines, skipped.")
                continue

            # Optional sanity check: verify the first two lines match the reference
            if first_two_lines_ref is not None and lines[:2] != first_two_lines_ref:
                print(f"[WARN] {path.name}: first two lines differ from the first file.")

            # Skip the first two lines and append the rest
            fout.writelines(lines[2:])
            total_written += max(0, len(lines) - 2)
            print(f"[OK] {path.name}: appended {max(0, len(lines) - 2)} lines (skipped first 2).")

print(f"Done. Merged into {output_path.name}. Total lines written: {total_written}")
