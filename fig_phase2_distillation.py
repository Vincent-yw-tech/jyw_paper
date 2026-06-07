# -*- coding: utf-8 -*-
"""Phase 2 知识蒸馏可视化：散点密度图 + 边界一致性热力图"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150; plt.rcParams['savefig.dpi'] = 150

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
SEED = 42; DEVICE = torch.device('cpu')
os.makedirs(FIG_DIR, exist_ok=True)

FEATURE_COLS = [
    'Credit_Score', 'Mortgage_Insurance', 'Number_of_units',
    'CLoan_to_value', 'Debt_to_income', 'OLoan_to_value',
    'Single_borrower', 'State_Default_LogOdds', 'First_Time_Homebuyer_Flag',
    'Loan_Purpose_WOE', 'Occupancy_WOE', 'Channel_WOE', 'Property_Type_WOE'
]

# ===================================================================
# Load data
# ===================================================================
gan_train = pd.read_csv(os.path.join(DATA_DIR, 'gan_train_set.csv'))
X_train_full = gan_train[FEATURE_COLS].values.astype(np.float32)
y_train_full = gan_train['DFlag'].values.astype(np.int64)
minority_idx = np.where(y_train_full == 1)[0]
majority_idx = np.where(y_train_full == 0)[0]

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_minority_scaled = X_train_full_scaled[minority_idx]

print(f"Minority samples: {len(minority_idx):,}")

# ===================================================================
# Phase 2: XGBoost base classifier + NN surrogate (same as experiment)
# ===================================================================
print("Training XGBoost base classifier...")
n_maj_base = min(len(minority_idx) * 5, len(majority_idx))
maj_sub = np.random.choice(len(majority_idx), n_maj_base, replace=False)
X_base = np.vstack([X_minority_scaled, X_train_full_scaled[maj_sub]])
y_base = np.hstack([np.ones(len(minority_idx)), np.zeros(n_maj_base)])
shuf = np.random.permutation(len(X_base))
X_base, y_base = X_base[shuf], y_base[shuf]

base_xgb = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=SEED, eval_metric='logloss', verbosity=0
)
base_xgb.fit(X_base, y_base)

p_xgb = base_xgb.predict_proba(X_minority_scaled)[:, 1].astype(np.float32)
d_xgb = np.abs(p_xgb - 0.5)
print(f"XGBoost p: mean={p_xgb.mean():.4f} std={p_xgb.std():.4f}")

# ===================================================================
# NN Surrogate
# ===================================================================
class Surrogate(nn.Module):
    def __init__(self, in_dim=13, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.LeakyReLU(0.2),
            nn.Linear(hidden//2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

print("Training NN surrogate...")
surr = Surrogate().to(DEVICE)
opt = optim.Adam(surr.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()
Xm_t = torch.FloatTensor(X_minority_scaled)
p_t = torch.FloatTensor(p_xgb).unsqueeze(1)
dl = DataLoader(TensorDataset(Xm_t, p_t), batch_size=512, shuffle=True)

surr.train()
for epoch in range(300):
    for xb, pb in dl:
        xb, pb = xb.to(DEVICE), pb.to(DEVICE)
        opt.zero_grad(); loss_fn(surr(xb), pb).backward(); opt.step()
surr.eval()

with torch.no_grad():
    p_nn = surr(Xm_t).cpu().numpy().flatten()
d_nn = np.abs(p_nn - 0.5)
corr = np.corrcoef(p_xgb, p_nn)[0, 1]
mae = np.abs(p_xgb - p_nn).mean()
print(f"Surrogate: r={corr:.4f} MAE={mae:.4f}")

# Boundary agreement
xgb_boundary = d_xgb < 0.2
nn_boundary = d_nn < 0.2
agreement = (xgb_boundary == nn_boundary).mean()
print(f"Boundary agreement: {agreement:.4f}")

# Confusion matrix for boundary
tp = ((xgb_boundary) & (nn_boundary)).sum()     # both say "near boundary"
tn = ((~xgb_boundary) & (~nn_boundary)).sum()   # both say "far"
fp = ((~xgb_boundary) & (nn_boundary)).sum()    # NN says near, XGBoost says far
fn = ((xgb_boundary) & (~nn_boundary)).sum()    # NN says far, XGBoost says near
print(f"Boundary confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

# ===================================================================
# FIGURE A: Scatter density plot
# ===================================================================
print("Generating scatter density plot...")
fig, ax = plt.subplots(figsize=(7.5, 7))

# Use hexbin for density
hb = ax.hexbin(p_xgb, p_nn, gridsize=80, cmap='YlOrRd', mincnt=1, bins='log')

ax.plot([0, 1], [0, 1], 'b--', linewidth=1.2, alpha=0.7, label='y = x (perfect fit)')
ax.set_xlabel('XGBoost Predicted Probability', fontsize=12)
ax.set_ylabel('NN Surrogate Predicted Probability', fontsize=12)
ax.set_title('Knowledge Distillation: XGBoost vs NN Surrogate\n(35,411 minority samples)', fontweight='bold', fontsize=13)

# Stats box
stats_text = f"Pearson r = {corr:.4f}\nMAE = {mae:.4f}\nBoundary Agreement = {agreement*100:.1f}%"
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Shade boundary region
ax.axvspan(0.3, 0.7, alpha=0.06, color='green')
ax.axhspan(0.3, 0.7, alpha=0.06, color='green')
ax.annotate('Boundary region\np ∈ [0.3, 0.7]', xy=(0.5, 0.15), fontsize=9,
            ha='center', color='green', alpha=0.8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(loc='lower right')
cbar = plt.colorbar(hb, ax=ax, shrink=0.82)
cbar.set_label('Sample count (log scale)', fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '08_phase2_scatter_density.png'), bbox_inches='tight')
plt.close()
print("  Saved: 08_phase2_scatter_density.png")

# ===================================================================
# FIGURE B: Boundary confusion matrix heatmap
# ===================================================================
print("Generating boundary consistency heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Confusion matrix ---
ax = axes[0]
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, cmap='Blues', aspect='auto')

# Annotate
ax.text(0, 0, f'{tn}\n({tn/len(p_xgb)*100:.1f}%)', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
ax.text(1, 0, f'{fp}\n({fp/len(p_xgb)*100:.1f}%)', ha='center', va='center', fontsize=16, fontweight='bold', color='#333')
ax.text(0, 1, f'{fn}\n({fn/len(p_xgb)*100:.1f}%)', ha='center', va='center', fontsize=16, fontweight='bold', color='#333')
ax.text(1, 1, f'{tp}\n({tp/len(p_xgb)*100:.1f}%)', ha='center', va='center', fontsize=16, fontweight='bold', color='white')

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['d ≥ 0.2 (far)', 'd < 0.2 (near)'], fontsize=11)
ax.set_yticklabels(['d ≥ 0.2 (far)', 'd < 0.2 (near)'], fontsize=11)
ax.set_xlabel('NN Surrogate Prediction', fontsize=12)
ax.set_ylabel('XGBoost Ground Truth', fontsize=12)
ax.set_title('Boundary Classification Consistency\n(d < 0.2 = near decision boundary)', fontweight='bold', fontsize=13)

# Highlight agreement
for i in range(2):
    for j in range(2):
        color = '#28a745' if i == j else '#dc3545'
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor=color, linewidth=3)
        ax.add_patch(rect)

ax.text(0.5, -0.22, f'Overall Agreement: {agreement*100:.1f}%',
        transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
        color='#28a745')

# --- Right: Boundary distance comparison ---
ax = axes[1]
bins = np.linspace(0, 0.5, 50)
ax.hist(d_xgb, bins=bins, alpha=0.55, label=f'XGBoost (d<0.2: {(d_xgb<0.2).mean()*100:.1f}%)',
        color='#0366d6', edgecolor='white', linewidth=0.5)
ax.hist(d_nn, bins=bins, alpha=0.55, label=f'NN Surrogate (d<0.2: {(d_nn<0.2).mean()*100:.1f}%)',
        color='#28a745', edgecolor='white', linewidth=0.5)
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5, label='Boundary threshold (d=0.2)')
ax.set_xlabel('Boundary Distance d = |p - 0.5|', fontsize=12)
ax.set_ylabel('Number of Default Samples', fontsize=12)
ax.set_title(f'Boundary Distance Distribution\nXGBoost vs NN Surrogate (r={corr:.4f})', fontweight='bold', fontsize=13)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '09_phase2_boundary_consistency.png'), bbox_inches='tight')
plt.close()
print("  Saved: 09_phase2_boundary_consistency.png")

print("\nDone!")
print(f"Files in {FIG_DIR}:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('08_') or f.startswith('09_'):
        sz = os.path.getsize(os.path.join(FIG_DIR, f))
        print(f"  {f} ({sz/1024:.0f} KB)")
