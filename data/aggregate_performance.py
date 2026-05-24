"""
将 performance 面板数据按 Loan_Sequence_Number 压缩为每笔贷款一行：
  - 数值型变量 → max
  - 日期型变量 → max (最近日期)
然后与 origination 一对一合并，输出 merged_loan_level.csv
"""

import os
import numpy as np
import pandas as pd

data_root = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 列分类
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "Current_Actual_UPB",
    "Current_Loan_Delinquency_Status",   # 需转为数值，max 反映最严重逾期
    "Loan_Age",
    "Remaining_Months_to_Legal_Maturity",
    "Current_Interest_Rate",
    "Current_Non_Interest_Bearing_UPB",
    "MI_Recoveries",
    "Net_Sale_Proceeds",
    "Non_MI_Recoveries",
    "Total_Expenses",
    "Legal_Costs",
    "Maintenance_and_Preservation_Costs",
    "Taxes_and_Insurance",
    "Miscellaneous_Expenses",
    "Actual_Loss_Calculation",
    "Cumulative_Modification_Cost",
    "Estimated_Loan_To_Value_ELTV",
    "Zero_Balance_Removal_UPB",
    "Delinquent_Accrued_Interest",
    "Current_Month_Modification_Cost",
    "Interest_Bearing_UPB",
]

DATE_COLS = [
    "Monthly_Reporting_Period",
    "Defect_Settlement_Date",
    "Zero_Balance_Effective_Date",
    "Due_Date_of_Last_Paid_Installment_DDLPI",
]

# 分类列也取 max（如有标记则保留）
CAT_COLS = [
    "Modification_Flag",
    "Zero_Balance_Code",
    "Step_Modification_Flag",
    "Payment_Deferral",
    "Delinquency_Due_to_Disaster",
    "Borrower_Assistance_Status_Code",
]

# 去掉 CAT_COLS 中实际不存在的列（拼写差异）
# Borrower_Assistance_Status_Code vs Borrower_Assistant_Status_Code
# 用脚本中实际拼写

# ---------------------------------------------------------------------------
# 1. 分块读取 performance，按 Loan_Sequence_Number 聚合
# ---------------------------------------------------------------------------
print("=== Step 1: Aggregating performance data (chunked) ===")
perf_path = os.path.join(data_root, "all_performance_2023_2025.csv")
chunk_aggs = []

# 修正：匹配实际 CSV 列名
ACTUAL_NUMERIC_COLS = []
ACTUAL_DATE_COLS = []
ACTUAL_CAT_COLS = []
ACTUAL_OTHER = ["Loan_Sequence_Number"]

# 先读一行获取实际列名
header = pd.read_csv(perf_path, nrows=0, encoding="utf-8-sig")
all_perf_cols = list(header.columns)

for c in NUMERIC_COLS:
    if c in all_perf_cols:
        ACTUAL_NUMERIC_COLS.append(c)
for c in DATE_COLS:
    if c in all_perf_cols:
        ACTUAL_DATE_COLS.append(c)
for c in CAT_COLS:
    if c in all_perf_cols:
        ACTUAL_CAT_COLS.append(c)

print(f"  Numeric columns: {len(ACTUAL_NUMERIC_COLS)}")
print(f"  Date columns:    {len(ACTUAL_DATE_COLS)}")
print(f"  Cat columns:     {len(ACTUAL_CAT_COLS)}")

agg_dict = {}
for c in ACTUAL_NUMERIC_COLS:
    agg_dict[c] = "max"
for c in ACTUAL_DATE_COLS:
    agg_dict[c] = "max"
for c in ACTUAL_CAT_COLS:
    agg_dict[c] = "max"

chunk_idx = 0
for chunk in pd.read_csv(perf_path, dtype=str, encoding="utf-8-sig",
                         chunksize=300_000, low_memory=False,
                         keep_default_na=False, na_filter=False):
    # 构建类型化 DataFrame
    typed_data = {"Loan_Sequence_Number": chunk["Loan_Sequence_Number"]}
    for c in ACTUAL_NUMERIC_COLS:
        # 空字符串 → NaN，其他 → float
        s = chunk[c].replace("", np.nan)
        typed_data[c] = pd.to_numeric(s, errors="coerce")
    for c in ACTUAL_DATE_COLS:
        # 保持字符串，空值保留为 ""
        typed_data[c] = chunk[c]
    for c in ACTUAL_CAT_COLS:
        typed_data[c] = chunk[c]

    typed_chunk = pd.DataFrame(typed_data)

    # 按 Loan_Sequence_Number 聚合
    grp = typed_chunk.groupby("Loan_Sequence_Number", as_index=False, sort=False).agg(agg_dict)
    chunk_aggs.append(grp)
    chunk_idx += 1
    if chunk_idx % 20 == 0:
        print(f"  Processed {chunk_idx} chunks ({chunk_idx * 300_000:,} rows)...")

print(f"  Processed {chunk_idx} chunks total. Combining...")

# 合并所有分块聚合结果，再聚合一次（因为同一贷款可能跨 chunk）
df_perf_agg = pd.concat(chunk_aggs, ignore_index=True)
df_perf_agg = df_perf_agg.groupby("Loan_Sequence_Number", as_index=False, sort=False).agg(agg_dict)

print(f"  Aggregated performance: {len(df_perf_agg):,} unique loans\n")

# ---------------------------------------------------------------------------
# 2. 与 origination 合并（一对一）
# ---------------------------------------------------------------------------
print("=== Step 2: Merging with origination data ===")
df_orig = pd.read_csv(
    os.path.join(data_root, "all_origination_2023_2025.csv"),
    dtype=str, encoding="utf-8-sig", low_memory=False
)
print(f"  Origination: {len(df_orig):,} loans")

df_merged = df_orig.merge(df_perf_agg, on="Loan_Sequence_Number", how="left")

# ---------------------------------------------------------------------------
# 3. 输出
# ---------------------------------------------------------------------------
# 构造 Default_Flag
print("=== Step 3: Creating Default_Flag ===")
max_delinq = pd.to_numeric(df_merged["Current_Loan_Delinquency_Status"], errors="coerce").fillna(0)
df_merged["Default_Flag"] = (
    (max_delinq >= 3) | (df_merged["Zero_Balance_Code"].isin(["03", "09"]))
).astype(int)

default_rate = df_merged["Default_Flag"].mean() * 100
print(f"  Default rate: {default_rate:.2f}%  "
      f"({df_merged['Default_Flag'].sum():,} / {len(df_merged):,})")

# 输出
out_path = os.path.join(data_root, "merged_loan_level.csv")
df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  -> merged_loan_level.csv  "
      f"({len(df_merged):,} rows, {len(df_merged.columns)} cols, "
      f"{os.path.getsize(out_path)/1024**2:.1f} MB)")

print("\nDone.")
print("\n--- Quick Summary ---")
print(f"Loans with performance data: {df_merged['Loan_Age'].notna().sum():,}")
print(f"Loans missing performance data: {df_merged['Loan_Age'].isna().sum():,}")
