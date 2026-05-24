"""
按照 Mushava & Murray (2024) 论文方法论构建不平衡数据集（FM24，2022-2025）。
内存优化版：先扫 performance → 确定有效贷款 ID → 按需读取 origination。
"""

import os
import gc
import numpy as np
import pandas as pd

data_root = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 配置
# ===========================================================================
OUTCOME_WINDOW = 24
QUARTERS = [
    "2022Q1", "2022Q2", "2022Q3", "2022Q4",
    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    "2024Q1", "2024Q2", "2024Q3", "2024Q4",
    "2025Q1", "2025Q2", "2025Q3",
]

ORIG_COLS = [
    "Credit_Score","First_Payment_Date","First_Time_Homebuyer_Flag","Maturity_Date",
    "Metropolitan_Statistical_Area","Mortgage_Insurance_Percentage","Number_of_Units",
    "Occupancy_Status","Original_Combined_Loan_To_Value_CLTV","Original_Debt_To_Income_DTI_Ratio",
    "Original_UPB","Original_Loan_To_Value_LTV","Original_Interest_Rate","Channel",
    "Prepayment_Penalty_Mortgage_Flag","Amortization_Type","Property_State","Property_Type",
    "Postal_Code","Loan_Sequence_Number","Loan_Purpose","Original_Loan_Term",
    "Number_of_Borrowers","Seller_Name","Servicer_Name","Super_Conforming_Flag",
    "Pre_Relief_Refinance_Loan_Sequence_Number","Special_Eligibility_Program",
    "Relief_Refinance_Indicator","Property_Valuation_Method","Interest_Only_Indicator",
    "MI_Cancellation_Indicator",
]

PERF_COLS = [
    "Loan_Sequence_Number","Monthly_Reporting_Period","Current_Actual_UPB",
    "Current_Loan_Delinquency_Status","Loan_Age","Remaining_Months_to_Legal_Maturity",
    "Defect_Settlement_Date","Modification_Flag","Zero_Balance_Code",
    "Zero_Balance_Effective_Date","Current_Interest_Rate","Current_Non_Interest_Bearing_UPB",
    "Due_Date_of_Last_Paid_Installment_DDLPI","MI_Recoveries","Net_Sale_Proceeds",
    "Non_MI_Recoveries","Total_Expenses","Legal_Costs","Maintenance_and_Preservation_Costs",
    "Taxes_and_Insurance","Miscellaneous_Expenses","Actual_Loss_Calculation",
    "Cumulative_Modification_Cost","Step_Modification_Flag","Payment_Deferral",
    "Estimated_Loan_To_Value_ELTV","Zero_Balance_Removal_UPB","Delinquent_Accrued_Interest",
    "Delinquency_Due_to_Disaster","Borrower_Assistance_Status_Code",
    "Current_Month_Modification_Cost","Interest_Bearing_UPB",
]

ALL_STATES = [
    "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","GU","HI","IA",
    "ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT",
    "NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","PR","RI",
    "SC","SD","TN","TX","UT","VA","VI","VT","WA","WI","WV","WY",
]


def _find_file(qtr, file_type):
    """file_type: 'orig' or 'perf'. 优先 CSV，否则 TXT."""
    d = os.path.join(data_root, f"historical_data_{qtr}")
    if file_type == "orig":
        csv_f = os.path.join(d, f"historical_data_{qtr}.csv")
        txt_f = os.path.join(d, f"historical_data_{qtr}.txt")
        cols = ORIG_COLS
    else:
        csv_f = os.path.join(d, f"historical_data_time_{qtr}.csv")
        txt_f = os.path.join(d, f"historical_data_time_{qtr}.txt")
        cols = PERF_COLS

    if os.path.exists(csv_f):
        return "csv", csv_f, cols
    elif os.path.exists(txt_f):
        return "txt", txt_f, cols
    return None, None, None


# ===========================================================================
# Step 1: 扫描 Performance → 计算每笔贷款的 max_loan_age + 违约状态
# ===========================================================================
print("=" * 60)
print(f"Step 1: Scanning performance data ({OUTCOME_WINDOW}m window)")
print("=" * 60)

agg_full, agg_win = [], []

for qtr in QUARTERS:
    ftype, fpath, cols = _find_file(qtr, "perf")
    if fpath is None:
        print(f"  SKIP {qtr}: not found")
        continue

    print(f"  {qtr} ({ftype}) ...", flush=True)
    n = 0
    reader_kwargs = dict(chunksize=300_000, low_memory=False, dtype=str, encoding="utf-8")
    if ftype == "txt":
        reader_kwargs.update(dict(sep="|", header=None, names=cols))
    else:
        reader_kwargs["encoding"] = "utf-8-sig"

    for chunk in pd.read_csv(fpath, **reader_kwargs):
        chunk["Loan_Age_num"] = pd.to_numeric(chunk["Loan_Age"], errors="coerce").fillna(0)
        chunk["Delinq_num"] = pd.to_numeric(chunk["Current_Loan_Delinquency_Status"], errors="coerce").fillna(0)

        # 全量 max_loan_age
        g = chunk.groupby("Loan_Sequence_Number").agg(max_loan_age=("Loan_Age_num", "max")).reset_index()
        agg_full.append(g)

        # 窗口内违约
        m = chunk["Loan_Age_num"] <= OUTCOME_WINDOW
        cw = chunk[m]
        if len(cw) > 0:
            cw = cw.copy()
            cw["has_zb"] = cw["Zero_Balance_Code"].isin(["03", "09"]).astype(int)
            gw = cw.groupby("Loan_Sequence_Number").agg(
                max_delinq_win=("Delinq_num", "max"),
                has_zb_default=("has_zb", "max"),
            ).reset_index()
            agg_win.append(gw)

        n += 1
        if n % 20 == 0:
            print(f"    {n} chunks ({n * 300_000:,} rows)", flush=True)

print(f"\n  Combining: {len(agg_full)} full + {len(agg_win)} window chunks ...", flush=True)

df_f = pd.concat(agg_full, ignore_index=True, copy=False)
del agg_full; gc.collect()
df_f = df_f.groupby("Loan_Sequence_Number", as_index=False).agg(max_loan_age=("max_loan_age", "max"))

if agg_win:
    df_w = pd.concat(agg_win, ignore_index=True, copy=False)
    del agg_win; gc.collect()
    df_w = df_w.groupby("Loan_Sequence_Number", as_index=False).agg(
        max_delinq_win=("max_delinq_win", "max"),
        has_zb_default=("has_zb_default", "max"),
    )
    df_perf = df_f.merge(df_w, on="Loan_Sequence_Number", how="left")
    df_perf["max_delinq_win"] = df_perf["max_delinq_win"].fillna(0)
    df_perf["has_zb_default"] = df_perf["has_zb_default"].fillna(0).astype(int)
else:
    df_perf = df_f
    df_perf["max_delinq_win"] = 0.0
    df_perf["has_zb_default"] = 0

del df_f, df_w; gc.collect()

df_perf["Default_Flag"] = (
    (df_perf["max_delinq_win"] >= 3) | (df_perf["has_zb_default"] == 1)
).astype(int)
df_perf["complete_obs"] = (df_perf["Default_Flag"] == 1) | (df_perf["max_loan_age"] >= OUTCOME_WINDOW)

n_total = len(df_perf)
n_defaults = df_perf["Default_Flag"].sum()
n_complete = df_perf["complete_obs"].sum()
print(f"  Performance loans: {n_total:,}")
print(f"  Defaults ({OUTCOME_WINDOW}m): {n_defaults:,}")
print(f"  Complete obs: {n_complete:,}")
print(f"  Insufficient obs: {n_total - n_complete:,}")

# 有效贷款 ID 集合（完整观测）
valid_ids = set(df_perf.loc[df_perf["complete_obs"], "Loan_Sequence_Number"].values)
print(f"  Valid loan IDs (complete obs): {len(valid_ids):,}")

# 保存 per-loan 违约状态（dict，快速查询）
df_complete = df_perf[df_perf["complete_obs"]]
default_dict = dict(zip(df_complete["Loan_Sequence_Number"].values,
                        df_complete["Default_Flag"].values))
del df_perf, df_complete; gc.collect()

# ===========================================================================
# Step 2: 按需读取 Origination → 过滤 + 特征工程 → 边读边写
# ===========================================================================
print("\n" + "=" * 60)
print("Step 2: Filtering origination + feature engineering")
print("=" * 60)

out_path = os.path.join(data_root, "imbalanced_dataset_FM24_2022_2025.csv")
first_write = True
total_written = 0
total_defaults = 0

for qtr in QUARTERS:
    ftype, fpath, cols = _find_file(qtr, "orig")
    if fpath is None:
        continue

    reader_kwargs = dict(dtype=str, low_memory=False)
    if ftype == "txt":
        reader_kwargs.update(dict(sep="|", header=None, names=cols, encoding="utf-8", chunksize=200_000))
    else:
        reader_kwargs.update(dict(encoding="utf-8-sig", chunksize=200_000))

    chunks_out = []
    for chunk in pd.read_csv(fpath, **reader_kwargs):
        # 过滤：只保留 performance 中有完整观测的贷款
        mask_id = chunk["Loan_Sequence_Number"].isin(valid_ids)
        chunk_f = chunk[mask_id]
        if len(chunk_f) == 0:
            continue

        # 过滤：Credit_Score 300-753
        cs = pd.to_numeric(chunk_f["Credit_Score"], errors="coerce")
        mask_cs = (cs >= 300) & (cs < 754)
        chunk_f = chunk_f[mask_cs]
        if len(chunk_f) == 0:
            continue

        # 特征工程
        out = pd.DataFrame()
        out["Loanref"] = chunk_f["Loan_Sequence_Number"]

        out["Credit_Score"] = pd.to_numeric(chunk_f["Credit_Score"], errors="coerce")
        mi = pd.to_numeric(chunk_f["Mortgage_Insurance_Percentage"], errors="coerce")
        out["Mortgage_Insurance"] = mi / 100.0
        out["Number_of_units"] = pd.to_numeric(chunk_f["Number_of_Units"], errors="coerce")
        out["CLoan_to_value"] = pd.to_numeric(chunk_f["Original_Combined_Loan_To_Value_CLTV"], errors="coerce")
        out["Debt_to_income"] = pd.to_numeric(chunk_f["Original_Debt_To_Income_DTI_Ratio"], errors="coerce")
        out["OLoan_to_value"] = pd.to_numeric(chunk_f["Original_Loan_To_Value_LTV"], errors="coerce")

        nb = chunk_f["Number_of_Borrowers"].fillna("09").str.strip()
        out["Single_borrower"] = (nb == "01").astype(int)

        # One-hot
        lp = chunk_f["Loan_Purpose"].fillna("9").str.strip().str.upper()
        out["is_Loan_purpose_purc"] = (lp == "P").astype(int)
        out["is_Loan_purpose_cash"] = (lp == "C").astype(int)
        out["is_Loan_purpose_noca"] = (lp == "N").astype(int)

        fthb = chunk_f["First_Time_Homebuyer_Flag"].fillna("9").str.strip().str.upper()
        out["is_First_time_homeowner"] = (fthb == "Y").astype(int)
        out["is_First_time_homeowner_No"] = (fthb == "N").astype(int)
        out["is_First_time_homeowner_miss"] = (~fthb.isin(["Y", "N"])).astype(int)

        occ = chunk_f["Occupancy_Status"].fillna("9").str.strip().str.upper()
        out["is_Occupancy_status_prim"] = (occ == "P").astype(int)
        out["is_Occupancy_status_inve"] = (occ == "I").astype(int)
        out["is_Occupancy_status_seco"] = (occ == "S").astype(int)

        ch = chunk_f["Channel"].fillna("9").str.strip().str.upper()
        out["is_Origination_channel_reta"] = (ch == "R").astype(int)
        out["is_Origination_channel_brok"] = (ch == "B").astype(int)
        out["is_Origination_channel_corr"] = (ch == "C").astype(int)

        pt = chunk_f["Property_Type"].fillna("9").str.strip().str.upper()
        out["is_Property_type_cond"] = (pt == "CO").astype(int)
        out["is_Property_type_coop"] = (pt == "CP").astype(int)
        out["is_Property_type_manu"] = (pt == "MH").astype(int)
        out["is_Property_type_pud"] = (pt == "PU").astype(int)
        out["is_Property_type_sing"] = (pt == "SF").astype(int)

        ps = chunk_f["Property_State"].fillna("XX").str.strip().str.upper()
        for state in ALL_STATES:
            out[f"is_property_state_{state}"] = (ps == state).astype(int)

        # 合并 performance 信息（dict 映射，快速）
        out["DFlag"] = out["Loanref"].map(default_dict).fillna(0).astype(int)

        # 缺失值
        for c in ["Credit_Score", "Mortgage_Insurance", "Number_of_units",
                  "CLoan_to_value", "Debt_to_income", "OLoan_to_value"]:
            out[c] = out[c].fillna(0)

        chunks_out.append(out)
        total_written += len(out)
        total_defaults += out["DFlag"].sum()

    if chunks_out:
        batch = pd.concat(chunks_out, ignore_index=True, copy=False)
        batch.to_csv(out_path, mode="w" if first_write else "a",
                     header=first_write, index=False, encoding="utf-8-sig")
        first_write = False
        print(f"  {qtr}: appended {len(batch):,} loans  (total: {total_written:,})", flush=True)
        del batch, chunks_out; gc.collect()
    else:
        print(f"  {qtr}: 0 loans after filtering", flush=True)

    chunks_out = []

print(f"\n  Total written: {total_written:,} loans")
print(f"  Total defaults: {int(total_defaults):,}")
print(f"  Default rate: {total_defaults/total_written*100:.2f}%")
print(f"  IR: {int((total_written - total_defaults)/total_defaults) if total_defaults > 0 else 'N/A'}")
print(f"  Output columns: 80 (expected)")

print(f"\n  -> {os.path.basename(out_path)}")
print(f"  File size: {os.path.getsize(out_path)/1024**2:.1f} MB")
print("\nDone.")
