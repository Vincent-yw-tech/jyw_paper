# -*- coding: utf-8 -*-
"""
WOE验证 AUC对比图：78维 vs 13维 XGBoost AUC，3个随机种子取平均
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 加载数据 ──────────────────────────────────────────
df78 = pd.read_csv(r'data\imbalanced_dataset_FM24_2022_2025.csv')
df13 = pd.read_csv(r'data\imbalanced_dataset_FM24_2022_2025_cleaned.csv')

feat78 = [c for c in df78.columns if c not in ['Loanref', 'DFlag']]
feat13 = [c for c in df13.columns if c not in ['Loanref', 'DFlag', 'first_default_month', 'speed_label']]

X78 = df78[feat78].values.astype(np.float32)
X13 = df13[feat13].values.astype(np.float32)
y   = df78['DFlag'].values

print(f'78-dim features: {len(feat78)}')
print(f'13-dim features: {len(feat13)}')
print(f'Samples: {len(y):,}, Default rate: {y.mean():.4f}')

seeds = [42, 123, 2024]
auc78_list, auc13_list = [], []

xgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0,
    'n_jobs': -1,
}

print('\n' + '=' * 60)
print(f'{"Seed":<8} {"AUC_78dim":<14} {"AUC_13dim":<14} {"Diff":<10}')
print('-' * 60)

for seed in seeds:
    Xtr78, Xte78, Xtr13, Xte13, ytr, yte = train_test_split(
        X78, X13, y, test_size=0.2, stratify=y, random_state=seed)

    clf78 = xgb.XGBClassifier(random_state=seed, **xgb_params)
    clf78.fit(Xtr78, ytr)
    auc78 = roc_auc_score(yte, clf78.predict_proba(Xte78)[:, 1])

    clf13 = xgb.XGBClassifier(random_state=seed, **xgb_params)
    clf13.fit(Xtr13, ytr)
    auc13 = roc_auc_score(yte, clf13.predict_proba(Xte13)[:, 1])

    auc78_list.append(auc78)
    auc13_list.append(auc13)
    print(f'{seed:<8} {auc78:<14.4f} {auc13:<14.4f} {auc78 - auc13:<+10.4f}')

avg78 = np.mean(auc78_list)
avg13 = np.mean(auc13_list)
print('-' * 60)
print(f'{"Mean":<8} {avg78:<14.4f} {avg13:<14.4f} {avg78 - avg13:<+10.4f}')

# ── 画图 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

x_labels = ['78维特征', '13维特征']
colors = ['#3498db', '#e74c3c']
means = [avg78, avg13]
bar_positions = [0, 1]

bars = ax.bar(bar_positions, means, width=0.45, color=colors, edgecolor='white', linewidth=0.8, zorder=2)

for bar_obj, mean_val in zip(bars, means):
    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.002,
            f'{mean_val:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

offset = 0.06
for i, seed in enumerate(seeds):
    ax.scatter([bar_positions[0] - offset + i * offset, bar_positions[1] - offset + i * offset],
               [auc78_list[i], auc13_list[i]], c='#2c3e50', s=40, zorder=3, alpha=0.7)
    ax.plot([bar_positions[0] - offset + i * offset, bar_positions[1] - offset + i * offset],
            [auc78_list[i], auc13_list[i]], color='#95a5a6', linewidth=0.6, alpha=0.5)

ax.set_xticks(bar_positions)
ax.set_xticklabels(x_labels, fontsize=13)
ax.set_ylabel('AUC', fontsize=13)
ax.set_title('WOE编码验证：78维 vs 13维 XGBoost AUC对比', fontsize=14, fontweight='bold')
ax.set_ylim(0.70, 0.78)
ax.grid(axis='y', alpha=0.3, zorder=1)

fig.tight_layout()
fig.savefig(r'data\WOE验证_AUC对比.png', dpi=200, bbox_inches='tight')
print('\nSaved: data/WOE验证_AUC对比.png')
