"""
高效的 performance 聚合脚本。
使用 Python 原生 csv 逐行流式处理，一次遍历完成 max 聚合，
避免 pandas 读取大文件的 OOM 和缓慢问题。
"""

import csv
import os
import numpy as np
import pandas as pd
from collections import defaultdict

data_root = r"D:\研究生\毕业论文\data"
perf_path = os.path.join(data_root, "all_performance_2023_2025.csv")
orig_path = os.path.join(data_root, "all_origination_2023_2025.csv")
out_path = os.path.join(data_root, "merged_loan_level.csv")

# ---------------------------------------------------------------------------
# Delinquency 映射 – 将字符串状态转为数值
# ---------------------------------------------------------------------------
DELINQ_MAP = {
    "": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "RA": 4, "FC": 4, "RE": 4, "XX": 0, "999": 0,
}

def parse_delinq(val):
    return DELINQ_MAP.get(val.strip(), 0)

# ---------------------------------------------------------------------------
# 列分类（按 init_data 函数中使用的键名）
# ---------------------------------------------------------------------------
NUMERIC_KEYS = [
    "Current_Actual_UPB", "Current_Loan_Delinquency_Status",
    "Loan_Age", "Remaining_Months_to_Legal_Maturity",
    "Current_Interest_Rate", "Current_Non_Interest_Bearing_UPB",
    "MI_Recoveries", "Net_Sale_Proceeds", "Non_MI_Recoveries",
    "Total_Expenses", "Legal_Costs", "Maintenance_and_Preservation_Costs",
    "Taxes_and_Insurance", "Miscellaneous_Expenses",
    "Actual_Loss_Calculation", "Cumulative_Modification_Cost",
    "Estimated_Loan_To_Value_ELTV", "Zero_Balance_Removal_UPB",
    "Delinquent_Accrued_Interest", "Current_Month_Modification_Cost",
    "Interest_Bearing_UPB",
]

DATE_KEYS = [
    "Monthly_Reporting_Period", "Defect_Settlement_Date",
    "Zero_Balance_Effective_Date", "Due_Date_of_Last_Paid_Installment_DDLPI",
]

CAT_KEYS = [
    "Modification_Flag", "Zero_Balance_Code",
    "Step_Modification_Flag", "Payment_Deferral",
    "Delinquency_Due_to_Disaster", "Borrower_Assistance_Status_Code",
]

def empty_state():
    """返回一个贷款的初始累积状态。"""
    state = {}
    for k in NUMERIC_KEYS:
        state[k] = 0.0
    for k in DATE_KEYS:
        state[k] = ""
    for k in CAT_KEYS:
        state[k] = ""
    return state

# ---------------------------------------------------------------------------
# 1. 一次遍历 performance CSV
# ---------------------------------------------------------------------------
print("=== Step 1: Streaming aggregation of performance data ===")
accum = defaultdict(empty_state)

with open(perf_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    header = set(reader.fieldnames)

    # 只处理实际存在的列
    num_keys = [k for k in NUMERIC_KEYS if k in header]
    date_keys = [k for k in DATE_KEYS if k in header]
    cat_keys = [k for k in CAT_KEYS if k in header]

    # Delinquency 特殊处理
    has_delinq = "Current_Loan_Delinquency_Status" in header

    row_count = 0
    for row in reader:
        loan_id = row["Loan_Sequence_Number"]
        cur = accum[loan_id]

        # 数值型：比较并保留最大值
        for k in num_keys:
            if k == "Current_Loan_Delinquency_Status" and has_delinq:
                v = parse_delinq(row[k])
            else:
                raw = row[k].strip()
                if raw == "" or raw == "999":
                    v = 0.0
                else:
                    try:
                        v = float(raw)
                    except ValueError:
                        v = 0.0
            if v > cur[k]:
                cur[k] = v

        # 日期型：保留最大（最近）
        for k in date_keys:
            v = row[k].strip()
            if v > cur[k]:
                cur[k] = v

        # 分类型：保留最大
        for k in cat_keys:
            v = row[k].strip()
            if v > cur[k]:
                cur[k] = v

        row_count += 1
        if row_count % 2_000_000 == 0:
            print(f"  已处理 {row_count / 1e6:.0f}M 行, "
                  f"累计 {len(accum):,} 个唯一贷款")

print(f"  完成: {row_count:,} 行 → {len(accum):,} 个唯一贷款")

# ---------------------------------------------------------------------------
# 2. 转为 DataFrame，与 origination 合并
# ---------------------------------------------------------------------------
print("\n=== Step 2: Building DataFrame & merging ===")
df_perf = pd.DataFrame.from_dict(accum, orient="index")
df_perf.index.name = "Loan_Sequence_Number"
df_perf = df_perf.reset_index()

# 读 origination
df_orig = pd.read_csv(orig_path, dtype=str, encoding="utf-8-sig", low_memory=False)

# 合并
df_merged = df_orig.merge(df_perf, on="Loan_Sequence_Number", how="left")
print(f"  Origination: {len(df_orig):,}  |  Performance: {len(df_perf):,}  |  Merged: {len(df_merged):,}")

# ---------------------------------------------------------------------------
# 3. Default_Flag
# ---------------------------------------------------------------------------
print("\n=== Step 3: Creating Default_Flag ===")
max_delinq = df_merged["Current_Loan_Delinquency_Status"].astype(float)
df_merged["Default_Flag"] = (
    (max_delinq >= 3) | (df_merged["Zero_Balance_Code"].isin(["03", "09"]))
).astype(int)

default_rate = df_merged["Default_Flag"].mean() * 100
print(f"  Default rate: {default_rate:.2f}%  "
      f"({df_merged['Default_Flag'].sum():,} / {len(df_merged):,})")

# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------
df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n  -> merged_loan_level.csv  "
      f"({len(df_merged):,} rows, {len(df_merged.columns)} cols, "
      f"{os.path.getsize(out_path)/1024**2:.1f} MB)")
print("Done.")
