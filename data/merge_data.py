"""
Merge all quarterly CSV files into two master files:
  - all_origination_2023_2025.csv
  - all_performance_2023_2025.csv
"""

import os
import pandas as pd

data_root = os.path.dirname(os.path.abspath(__file__))

# Collect files
orig_files = []
perf_files = []
for dirpath, _, filenames in os.walk(data_root):
    for fn in sorted(filenames):
        if not fn.endswith(".csv"):
            continue
        full = os.path.join(dirpath, fn)
        if "_time_" in fn:
            perf_files.append(full)
        else:
            orig_files.append(full)

# ---------------------------------------------------------------------------
# 1. Merge origination files (fits in memory)
# ---------------------------------------------------------------------------
print("=== Merging origination files ===")
orig_parts = []
for f in orig_files:
    qtr = os.path.basename(f).replace("historical_data_", "").replace(".csv", "")
    print(f"  Reading {qtr} ...")
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig", low_memory=False)
    orig_parts.append(df)

df_orig_all = pd.concat(orig_parts, ignore_index=True)
orig_out = os.path.join(data_root, "all_origination_2023_2025.csv")
df_orig_all.to_csv(orig_out, index=False, encoding="utf-8-sig")
print(f"  -> all_origination_2023_2025.csv  "
      f"({len(df_orig_all):,} rows, {os.path.getsize(orig_out)/1024**2:.1f} MB)\n")

# Free memory
del df_orig_all, orig_parts

# ---------------------------------------------------------------------------
# 2. Merge performance files (stream chunked to avoid OOM)
# ---------------------------------------------------------------------------
print("=== Merging performance files ===")
perf_out = os.path.join(data_root, "all_performance_2023_2025.csv")
first_file = True
total_rows = 0

for f in perf_files:
    qtr = os.path.basename(f).replace("historical_data_time_", "").replace(".csv", "")
    print(f"  Appending {qtr} ...")

    for chunk in pd.read_csv(f, dtype=str, encoding="utf-8-sig",
                             chunksize=200_000, low_memory=False):
        chunk.to_csv(perf_out, mode="w" if first_file else "a",
                     header=first_file, index=False, encoding="utf-8-sig")
        first_file = False
        total_rows += len(chunk)

print(f"  -> all_performance_2023_2025.csv  "
      f"({total_rows:,} rows, {os.path.getsize(perf_out)/1024**2:.1f} MB)\n")

print("Done. Two merged files created:")
print(f"  {orig_out}")
print(f"  {perf_out}")
