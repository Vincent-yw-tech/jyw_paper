# -*- coding: utf-8 -*-
"""
Build speed labels (Early/Late Default) for 2022Q1-2023Q3 defaulted loans.
Early = first default month <= 12,  Late = first default month > 12.

Strategy: chunk-read Performance data, filter to only default loan IDs,
and track the FIRST month where DLQ>=3 or ZBC in {03,09}.
"""
import os
import numpy as np
import pandas as pd
import gc

data_root = os.path.dirname(os.path.abspath(__file__))

WINDOW = 24

# --- Load cleaned dataset to get default loan IDs from 2022Q1-2023Q3 ---
df = pd.read_csv(os.path.join(data_root, 'imbalanced_dataset_FM24_2022_2025_cleaned.csv'))

def pq(lr):
    y = int(lr[1:3]); q = lr[3:5]; return '20{:02d}{}'.format(y, q)
df['quarter'] = df['Loanref'].apply(pq)

train_q = ['2022Q1','2022Q2','2022Q3','2022Q4','2023Q1','2023Q2','2023Q3']
default_loans = df[(df['quarter'].isin(train_q)) & (df['DFlag'] == 1)]['Loanref'].values
default_set = set(default_loans)
print('Target default loans (2022Q1-2023Q3): {:,}'.format(len(default_set)))

# --- Performance columns ---
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

# --- Scan Performance data ---
QUARTERS = train_q
first_default = {}  # loan_id -> first_default_month

print('\nScanning Performance data for first default month...')
print('=' * 60)

for qtr in QUARTERS:
    subdir = os.path.join(data_root, 'historical_data_{}'.format(qtr))
    csv_f = os.path.join(subdir, 'historical_data_time_{}.csv'.format(qtr))
    txt_f = os.path.join(subdir, 'historical_data_time_{}.txt'.format(qtr))

    if os.path.exists(csv_f):
        ftype, fpath = 'csv', csv_f
    elif os.path.exists(txt_f):
        ftype, fpath = 'txt', txt_f
    else:
        print('  {}: SKIP (no data)'.format(qtr))
        continue

    print('  {} ({}) ...'.format(qtr, ftype), flush=True)

    reader_kwargs = dict(chunksize=500_000, low_memory=False, dtype=str)
    if ftype == 'txt':
        reader_kwargs.update(dict(sep='|', header=None, names=PERF_COLS, encoding='utf-8'))
    else:
        reader_kwargs['encoding'] = 'utf-8-sig'

    n_rows, n_matched = 0, 0
    for chunk in pd.read_csv(fpath, **reader_kwargs):
        n_rows += len(chunk)

        # Filter to only our target default loans
        mask = chunk['Loan_Sequence_Number'].isin(default_set)
        sub = chunk[mask]
        if len(sub) == 0:
            continue
        n_matched += len(sub)

        # Parse numeric columns
        loan_ids = sub['Loan_Sequence_Number'].values
        loan_ages = pd.to_numeric(sub['Loan_Age'], errors='coerce').fillna(0).values
        delinqs = pd.to_numeric(sub['Current_Loan_Delinquency_Status'], errors='coerce').fillna(0).values
        zbcs = sub['Zero_Balance_Code'].values

        # Find first default month for each row
        is_default_event = (delinqs >= 3) | (np.isin(zbcs, ['03', '09']))
        is_within_window = loan_ages <= WINDOW

        for i in range(len(sub)):
            lid = loan_ids[i]
            if is_default_event[i] and is_within_window[i]:
                age = int(loan_ages[i])
                if lid not in first_default or age < first_default[lid]:
                    first_default[lid] = age

    print('    rows={:,}  matched={:,}  found_first_default={:,}'.format(
        n_rows, n_matched, len(first_default)), flush=True)

print('\n' + '=' * 60)
print('Speed label construction complete')
print('=' * 60)

n_found = len(first_default)
n_missing = len(default_set) - n_found
print('Default loans:    {:,}'.format(len(default_set)))
print('First month found:  {:,}'.format(n_found))
print('Not found in window: {:,} (may be zb default outside window)'.format(n_missing))

# Distribution
ages = np.array(list(first_default.values()))
print('\nFirst default month distribution:')
print('  Mean:   {:.1f} months'.format(ages.mean()))
print('  Median: {:.0f} months'.format(np.median(ages)))
print('  Std:    {:.1f} months'.format(ages.std()))
print('  Min:    {} months'.format(ages.min()))
print('  Max:    {} months'.format(ages.max()))
print('')
print('  <= 6 months:  {:,} ({:.1f}%)'.format((ages <= 6).sum(), (ages <= 6).sum()/len(ages)*100))
print('  <=12 months:  {:,} ({:.1f}%)'.format((ages <= 12).sum(), (ages <= 12).sum()/len(ages)*100))
print('  13-24 months: {:,} ({:.1f}%)'.format((ages > 12).sum(), (ages > 12).sum()/len(ages)*100))

# Build labels
labels = {}
for lid in default_set:
    if lid in first_default and first_default[lid] <= 12:
        labels[lid] = 1   # Early Default
    elif lid in first_default and first_default[lid] > 12:
        labels[lid] = 0   # Late Default
    else:
        labels[lid] = -1  # Not determined (will be handled below)

# Summary
n_early = sum(1 for v in labels.values() if v == 1)
n_late = sum(1 for v in labels.values() if v == 0)
n_unknown = sum(1 for v in labels.values() if v == -1)
print('\nSpeed label summary:')
print('  Early Default (<=12m):  {:,} ({:.1f}%)'.format(n_early, n_early/(n_early+n_late)*100 if n_early+n_late>0 else 0))
print('  Late Default (>12m):   {:,} ({:.1f}%)'.format(n_late, n_late/(n_early+n_late)*100 if n_early+n_late>0 else 0))
print('  Unknown:               {:,}'.format(n_unknown))

# Save
speed_df = pd.DataFrame([
    {'Loanref': lid, 'first_default_month': first_default.get(lid, np.nan),
     'speed_label': labels[lid] if labels[lid] != -1 else np.nan}
    for lid in default_set
])
speed_df['speed_label'] = speed_df['speed_label'].astype('Int64')

out_path = os.path.join(data_root, 'speed_labels.csv')
speed_df.to_csv(out_path, index=False, encoding='utf-8-sig')
print('\nSaved: {}'.format(out_path))
print('  Columns: Loanref, first_default_month, speed_label')
print('  speed_label: 1=Early(<=12m), 0=Late(>12m)')
