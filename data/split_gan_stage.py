# -*- coding: utf-8 -*-
"""
First-layer split: GAN stage 80/20
- Stratified by quarter + DFlag on complete observations (2022Q1-2023Q3)
- Right-censored defaults (2023Q4-2025Q2, 8,150 loans) all go to GAN training set
- Output: gan_train_set.csv, gan_val_set.csv
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

data_root = os.path.dirname(os.path.abspath(__file__))
SEED = 42

df = pd.read_csv(os.path.join(data_root, 'imbalanced_dataset_FM24_2022_2025_cleaned.csv'))

def pq(lr):
    y = int(lr[1:3]); q = lr[3:5]; return '20{:02d}{}'.format(y, q)

df['quarter'] = df['Loanref'].apply(pq)

# --- Separate right-censored defaults ---
train_q = ['2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3']
censored = df[~df['quarter'].isin(train_q)].copy()
complete = df[df['quarter'].isin(train_q)].copy()

# --- 80/20 stratified split ---
complete['stratify_key'] = complete['quarter'].astype(str) + '_' + complete['DFlag'].astype(str)

gan_train_complete, gan_val = train_test_split(
    complete, test_size=0.20, random_state=SEED,
    stratify=complete['stratify_key']
)

# --- Combine with censored ---
gan_train = pd.concat([gan_train_complete, censored], ignore_index=True)

# --- Clean up ---
for d in [gan_train, gan_val]:
    d.drop(columns=['quarter', 'stratify_key'], inplace=True, errors='ignore')

# --- Save ---
gan_train.to_csv(os.path.join(data_root, 'gan_train_set.csv'), index=False, encoding='utf-8-sig')
gan_val.to_csv(os.path.join(data_root, 'gan_val_set.csv'), index=False, encoding='utf-8-sig')

print('GAN Train: {:,} rows ({} defaults)'.format(len(gan_train), gan_train['DFlag'].sum()))
print('GAN Val:   {:,} rows ({} defaults)'.format(len(gan_val), gan_val['DFlag'].sum()))
print('Done.')
