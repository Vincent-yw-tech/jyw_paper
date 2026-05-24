# -*- coding: utf-8 -*-
"""
WOE encoding for remaining one-hot groups.
Replaces 14 one-hot columns -> 4 continuous WOE variables.
"""
import pandas as pd
import numpy as np

INPUT = r'data\imbalanced_dataset_FM24_2022_2025_cleaned.csv'
df = pd.read_csv(INPUT)
print('Before: {} features (excl. Loanref, DFlag)'.format(df.shape[1] - 2))

# --- Identify groups ---
GROUPS = {
    'Loan_Purpose': {
        'cols': ['is_Loan_purpose_purc', 'is_Loan_purpose_cash', 'is_Loan_purpose_noca'],
        'labels': ['Purchase', 'Cash-out Refi', 'No-cash Refi'],
    },
    'Occupancy': {
        'cols': ['is_Occupancy_status_prim', 'is_Occupancy_status_inve', 'is_Occupancy_status_seco'],
        'labels': ['Primary', 'Investment', 'Second'],
    },
    'Channel': {
        'cols': ['is_Origination_channel_reta', 'is_Origination_channel_brok', 'is_Origination_channel_corr'],
        'labels': ['Retail', 'Broker', 'Correspondent'],
    },
    'Property_Type': {
        'cols': ['is_Property_type_cond', 'is_Property_type_coop', 'is_Property_type_manu',
                 'is_Property_type_pud', 'is_Property_type_sing'],
        'labels': ['Condo', 'Coop', 'Manufactured', 'PUD', 'Single-Family'],
    },
}

# --- Parse quarter ---
def pq(lr):
    y = int(lr[1:3]); q = lr[3:5]; return '20{:02d}{}'.format(y, q)
df['quarter'] = df['Loanref'].apply(pq)
train_q = ['2022Q1','2022Q2','2022Q3','2022Q4','2023Q1','2023Q2','2023Q3']
train = df[df['quarter'].isin(train_q)]

print('\nWOE encoding (computed on 2022Q1-2023Q3 training data only):')
print('=' * 70)

all_onehot_to_drop = []

for group_name, info in GROUPS.items():
    cols = info['cols']
    labels = info['labels']
    woe_col = group_name + '_WOE'

    # Identify which category each loan belongs to
    def get_category(row):
        for c, label in zip(cols, labels):
            if row[c] == 1:
                return label
        return 'Other'

    train_cat = train.apply(get_category, axis=1)

    # Compute WOE per category from training data
    total_good = (train['DFlag'] == 0).sum()
    total_bad = (train['DFlag'] == 1).sum()

    woe_map = {}
    print('\n{}:'.format(group_name))
    print('  {:20s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}'.format(
        'Category', 'Loans', 'Defaults', 'Rate%', 'WOE'))

    for label in labels:
        mask = train_cat == label
        n_loans = mask.sum()
        n_bad = train.loc[mask, 'DFlag'].sum()
        n_good = n_loans - n_bad
        rate = n_bad / n_loans * 100 if n_loans > 0 else 0

        # WOE with Laplace smoothing
        pct_good = (n_good + 1) / (total_good + len(labels))
        pct_bad = (n_bad + 1) / (total_bad + len(labels))
        woe = np.log(pct_good / pct_bad)

        woe_map[label] = woe
        print('  {:20s}  {:>8,}  {:>8,}  {:>7.2f}  {:>+8.4f}'.format(
            label, n_loans, n_bad, rate, woe))

    # Apply WOE to all data
    all_cat = df.apply(get_category, axis=1)
    df[woe_col] = all_cat.map(woe_map).fillna(0)

    all_onehot_to_drop.extend(cols)

# --- Drop old one-hot columns ---
df.drop(columns=all_onehot_to_drop + ['quarter'], inplace=True)

n_features = df.shape[1] - 2
print('\n' + '=' * 70)
print('After: {} features (excl. Loanref, DFlag)'.format(n_features))
print('Reduction: 14 one-hot -> 4 WOE ({} features total)'.format(n_features))

# --- List final features ---
feat_cols = [c for c in df.columns if c not in ['Loanref', 'DFlag']]
print('\nFinal features:')
for i, c in enumerate(feat_cols, 1):
    print('  {:2d}. {}'.format(i, c))

# --- Save ---
df.to_csv(r'data\imbalanced_dataset_FM24_2022_2025_cleaned.csv', index=False)
import os
sz = os.path.getsize(r'data\imbalanced_dataset_FM24_2022_2025_cleaned.csv') / (1024*1024)
print('\nSaved. Size: {:.1f} MB'.format(sz))
