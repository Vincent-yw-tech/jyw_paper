"""从合并数据中删除 2023 年贷款，保留 2024-2025 年数据。"""

import os
import pandas as pd

data_root = r"D:\研究生\毕业论文\data"

# ---------------------------------------------------------------------------
# 1. 过滤 origination
# ---------------------------------------------------------------------------
print("=== Step 1: Filtering origination ===")
df_orig = pd.read_csv(os.path.join(data_root, "all_origination_2023_2025.csv"),
                      dtype=str, encoding="utf-8-sig", low_memory=False)
before = len(df_orig)
df_orig = df_orig[~df_orig["First_Payment_Date"].str.startswith("2023")]
after = len(df_orig)
print(f"  2023 年删除 {before - after:,} 笔, 剩余 {after:,} 笔")

keep_ids = set(df_orig["Loan_Sequence_Number"].tolist())

df_orig.to_csv(os.path.join(data_root, "all_origination_2023_2025.csv"),
               index=False, encoding="utf-8-sig")
print("  -> all_origination_2023_2025.csv 已更新\n")

del df_orig

# ---------------------------------------------------------------------------
# 2. 过滤 performance (chunked)
# ---------------------------------------------------------------------------
print("=== Step 2: Filtering performance ===")
perf_path = os.path.join(data_root, "all_performance_2023_2025.csv")
tmp_path = os.path.join(data_root, "_perf_temp.csv")
first_chunk = True
total_kept = 0
total_all = 0

for chunk in pd.read_csv(perf_path, dtype=str, encoding="utf-8-sig",
                         chunksize=500_000, low_memory=False):
    total_all += len(chunk)
    filtered = chunk[chunk["Loan_Sequence_Number"].isin(keep_ids)]
    filtered.to_csv(tmp_path, mode="w" if first_chunk else "a",
                    header=first_chunk, index=False, encoding="utf-8-sig")
    first_chunk = False
    total_kept += len(filtered)
    if total_all % 5_000_000 == 0:
        print(f"  处理 {total_all/1e6:.0f}M 行...")

print(f"  总 {total_all:,} 行 → 保留 {total_kept:,} 行")

# Replace original
os.replace(tmp_path, perf_path)
print("  -> all_performance_2023_2025.csv 已更新\n")
print("Done.")
