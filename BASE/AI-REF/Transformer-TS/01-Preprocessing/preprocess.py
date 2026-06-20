import os
import pandas as pd

# ---------- Config ----------
REL_PATH = "data/suzhoudata.csv"       # relative path to your CSV
OUTPUT_FILE = "headers_output.txt"     # output txt file
# ----------------------------

def resolve_path(rel_path: str) -> str:
    """Resolve file path relative to this script file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, rel_path)

def main():
    csv_path = resolve_path(REL_PATH)
    out_path = resolve_path(OUTPUT_FILE)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # Let pandas read only the header (no data)
    df = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
    cols = list(df.columns)

    # Write one column name per line
    with open(out_path, "w", encoding="utf-8") as fout:
        for c in cols:
            fout.write(f"{c}\n")

    print(f"Column headers saved to {out_path}")

if __name__ == "__main__":
    main()
