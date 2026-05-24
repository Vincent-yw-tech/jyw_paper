# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv(r'data\imbalanced_dataset_FM24_2022_2025_cleaned.csv')

def pq(lr):
    y = int(lr[1:3]); q = lr[3:5]; return '20{:02d}{}'.format(y, q)
df['quarter'] = df['Loanref'].apply(pq)
train_q = ['2022Q1','2022Q2','2022Q3','2022Q4','2023Q1','2023Q2','2023Q3']
train = df[df['quarter'].isin(train_q)]
cens  = df[~df['quarter'].isin(train_q)]

num_cols = ['Credit_Score','Mortgage_Insurance','Number_of_units',
            'CLoan_to_value','Debt_to_income','OLoan_to_value',
            'Single_borrower','First_Time_Homebuyer_Flag','State_Default_LogOdds']
onehot_cols = [c for c in df.columns if c.startswith('is_')]

print('=' * 80)
print('1. DATASET OVERVIEW')
print('=' * 80)
print('Total samples:  {:,}'.format(len(df)))
print('Total columns:  {} ({} features + Loanref + DFlag)'.format(df.shape[1], len(num_cols)+len(onehot_cols)))
print('')
print('Defaults:       {:,} ({:.2f}%)'.format(df['DFlag'].sum(), df['DFlag'].mean()*100))
print('Non-defaults:   {:,} ({:.2f}%)'.format((df['DFlag']==0).sum(), (1-df['DFlag'].mean())*100))
print('Imbalance Ratio: {:.1f}'.format((df['DFlag']==0).sum() / df['DFlag'].sum()))
print('')
print('2022Q1-2023Q3 (complete obs): {:,}  defaults={:,}  rate={:.2f}%'.format(
    len(train), train['DFlag'].sum(), train['DFlag'].mean()*100))
print('2023Q4-2025Q2 (right-censored): {:,}  defaults={:,}  rate={:.2f}%'.format(
    len(cens), cens['DFlag'].sum(), cens['DFlag'].mean()*100))
print('File size: 75.2 MB')

print('')
print('=' * 80)
print('2. QUARTERLY DISTRIBUTION')
print('=' * 80)
qq = df.groupby('quarter').agg(n=('DFlag','count'), d=('DFlag','sum'))
qq['rate'] = (qq['d']/qq['n']*100).round(2)
qq['pct_total'] = (qq['n']/len(df)*100).round(1)
print('{:>8s}  {:>10s}  {:>10s}  {:>8s}  {:>8s}'.format('Quarter','Loans','Defaults','Rate%','%Total'))
print('-' * 50)
for idx, row in qq.iterrows():
    tag = ' *censored' if idx not in train_q else ''
    print('{:>8s}  {:>10,}  {:>10,}  {:>7.2f}  {:>7.1f}{}'.format(
        idx, int(row['n']), int(row['d']), row['rate'], row['pct_total'], tag))

print('')
print('=' * 80)
print('3. NUMERIC FEATURES (9) - All Data')
print('=' * 80)
print('{:>28s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}'.format(
    'Feature','Mean','Std','Min','25%','50%','75%','Max'))
print('-' * 80)
for c in num_cols:
    s = df[c]
    print('{:>28s}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}'.format(
        c, s.mean(), s.std(), s.min(), s.quantile(0.25), s.median(), s.quantile(0.75), s.max()))

print('')
print('=' * 80)
print('4. NUMERIC FEATURES - By Default Status (Training Set Only)')
print('=' * 80)
print('{:>28s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}'.format(
    'Feature','NonDef Mean','Def Mean','Diff','Diff/Std'))
print('-' * 80)
for c in num_cols:
    nd = train.loc[train['DFlag']==0, c]
    d  = train.loc[train['DFlag']==1, c]
    diff = d.mean() - nd.mean()
    pool_std = np.sqrt((nd.var() + d.var())/2)
    print('{:>28s}  {:>10.4f}  {:>10.4f}  {:>+10.4f}  {:>10.2f}'.format(
        c, nd.mean(), d.mean(), diff, diff/pool_std))

print('')
print('=' * 80)
print('5. ONE-HOT FEATURES (14) - Default Rate by Category')
print('=' * 80)
print('{:>40s}  {:>8s}  {:>10s}  {:>10s}'.format('Feature','Prop%','NonDef%','Def%'))
print('-' * 72)
for c in sorted(onehot_cols):
    prop = train[c].mean() * 100
    nd_rate = train.loc[train['DFlag']==0, c].mean() * 100
    d_rate  = train.loc[train['DFlag']==1, c].mean() * 100
    print('{:>40s}  {:>7.1f}  {:>9.1f}  {:>9.1f}'.format(c, prop, nd_rate, d_rate))

print('')
print('=' * 80)
print('6. QUARTERLY DEFAULT RATE TREND')
print('=' * 80)
for idx, row in qq[qq.index.isin(train_q)].iterrows():
    bar = '#' * int(row['rate'] * 3)
    print('  {}  {:>5.2f}%  {}'.format(idx, row['rate'], bar))

print('')
print('=' * 80)
print('7. FEATURE GROUP SUMMARY')
print('=' * 80)
print('Group                 Cols  Description')
print('-' * 60)
print('Numeric                  9  Credit_Score, MI, Units, CLTV, DTI, LTV, SingleBorr, FirstTime, StateLogOdds')
print('Loan Purpose             3  Purchase / Cash-out Refi / No-cash Refi')
print('Occupancy                3  Primary / Investment / Second')
print('Channel                  3  Retail / Broker / Correspondent')
print('Property Type            5  Condo / Coop / Manufactured / PUD / Single-Family')
print('State (encoded)          1  State_Default_LogOdds (54 states -> log-odds)')
print('-' * 60)
print('Total features          23')
