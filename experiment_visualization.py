# -*- coding: utf-8 -*-
"""
BA-cWGAN-GP 预实验可视化脚本
生成图表：
  1. GAN 训练损失曲线（WGAN-GP vs BA-cWGAN-GP）
  2. 边界富集度对比柱状图
  3. 四方法 AUC/F1/KS/Brier 对比柱状图
  4. 违约倾向得分 r 分布直方图
  5. 边界距离 d 分布对比（真实 vs 生成）
  6. 生成样本质量 PCA 可视化
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ===========================================================================
# 中文字体设置
# ===========================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ===========================================================================
# Configuration
# ===========================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_COLS = [
    'Credit_Score', 'Mortgage_Insurance', 'Number_of_units',
    'CLoan_to_value', 'Debt_to_income', 'OLoan_to_value',
    'Single_borrower', 'State_Default_LogOdds', 'First_Time_Homebuyer_Flag',
    'Loan_Purpose_WOE', 'Occupancy_WOE', 'Channel_WOE', 'Property_Type_WOE'
]

Z_DIM = 100; R_DIM = 1
LAMBDA_GP = 10.0; LAMBDA_BOUNDARY = 0.1; LAMBDA_R = 0.05
N_CRITIC = 5; BATCH_SIZE = 256; GAN_EPOCHS = 200
G_LR = 1e-4; D_LR = 1e-4

os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"Figures will be saved to: {FIG_DIR}")

# ===========================================================================
# Load Data
# ===========================================================================
print("\nLoading data...")
gan_train = pd.read_csv(os.path.join(DATA_DIR, 'gan_train_set.csv'))
gan_val = pd.read_csv(os.path.join(DATA_DIR, 'gan_val_set.csv'))

X_train_full = gan_train[FEATURE_COLS].values.astype(np.float32)
y_train_full = gan_train['DFlag'].values.astype(np.int64)
X_val = gan_val[FEATURE_COLS].values.astype(np.float32)
y_val = gan_val['DFlag'].values.astype(np.int64)

minority_idx = np.where(y_train_full == 1)[0]
majority_idx = np.where(y_train_full == 0)[0]

has_speed = gan_train['speed_label'].notna().values & (y_train_full == 1)
speed_labels = gan_train.loc[has_speed, 'speed_label'].values.astype(np.int64)

print(f"GAN Train: {len(X_train_full):,} | GAN Val: {len(X_val):,}")

# Standardize
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)
X_minority_scaled = X_train_full_scaled[minority_idx]
X_majority_scaled = X_train_full_scaled[majority_idx]
X_speed_scaled = X_train_full_scaled[np.where(has_speed)[0]]

Xm_t = torch.FloatTensor(X_minority_scaled)

# ===========================================================================
# Phase 1: Speed Classifier
# ===========================================================================
print("\nPhase 1: Speed Classifier...")
speed_clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=SEED, eval_metric='logloss', verbosity=0
)
speed_clf.fit(X_speed_scaled, speed_labels)
r_all = speed_clf.predict_proba(X_minority_scaled)[:, 1].astype(np.float32)

# NN Surrogate
class Surrogate(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.LeakyReLU(0.2),
            nn.Linear(hidden//2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

surr_speed = Surrogate(13, 256).to(DEVICE)
opt = optim.Adam(surr_speed.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()
r_t = torch.FloatTensor(r_all).unsqueeze(1)
dl_s = DataLoader(TensorDataset(Xm_t, r_t), batch_size=512, shuffle=True)

surr_speed.train()
for epoch in range(200):
    for xb, rb in dl_s:
        xb, rb = xb.to(DEVICE), rb.to(DEVICE)
        opt.zero_grad(); loss_fn(surr_speed(xb), rb).backward(); opt.step()
surr_speed.eval()

# ===========================================================================
# Phase 2: Base Classifier
# ===========================================================================
print("Phase 2: Base Classifier...")
n_maj_base = min(len(minority_idx) * 5, len(majority_idx))
maj_sub = np.random.choice(len(majority_idx), n_maj_base, replace=False)
X_base_xgb = np.vstack([X_minority_scaled, X_majority_scaled[maj_sub]])
y_base_xgb = np.hstack([np.ones(len(minority_idx)), np.zeros(n_maj_base)])
shuf = np.random.permutation(len(X_base_xgb))
X_base_xgb, y_base_xgb = X_base_xgb[shuf], y_base_xgb[shuf]

base_xgb = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=SEED, eval_metric='logloss', verbosity=0
)
base_xgb.fit(X_base_xgb, y_base_xgb)
p_min_xgb = base_xgb.predict_proba(X_minority_scaled)[:, 1].astype(np.float32)
d_xgb = np.abs(p_min_xgb - 0.5)
weights = 1.0 / (d_xgb + 0.01); weights /= weights.sum()

# NN Surrogate for base
surr_base = Surrogate(13, 256).to(DEVICE)
opt_b = optim.Adam(surr_base.parameters(), lr=1e-3)
p_xgb_t = torch.FloatTensor(p_min_xgb).unsqueeze(1)
dl_b = DataLoader(TensorDataset(Xm_t, p_xgb_t), batch_size=512, shuffle=True)

surr_base.train()
for epoch in range(300):
    for xb, pb in dl_b:
        xb, pb = xb.to(DEVICE), pb.to(DEVICE)
        opt_b.zero_grad(); loss_fn(surr_base(xb), pb).backward(); opt_b.step()
surr_base.eval()
base_clf = surr_base  # differentiable

# ===========================================================================
# Phase 3: GAN Training (with loss history collection)
# ===========================================================================
print("Phase 3: GAN Training...")

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim)
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

def gradient_penalty(D, real, fake):
    eps = torch.rand(real.size(0), 1, device=DEVICE)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_hat = D(x_hat)
    grads = torch.autograd.grad(outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat), create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

X_real_t = Xm_t.to(DEVICE)
r_all_t = torch.FloatTensor(r_all).unsqueeze(1).to(DEVICE)
weights_t = torch.FloatTensor(weights).to(DEVICE)

history = {'wgan': {'D': [], 'G': []}, 'ba': {'D': [], 'G': [], 'B': [], 'R': []}}

# --- WGAN-GP ---
G_wgan = Generator(Z_DIM, 13).to(DEVICE)
D_wgan = Discriminator(13).to(DEVICE)
opt_Gw = optim.Adam(G_wgan.parameters(), lr=G_LR, betas=(0.5, 0.9))
opt_Dw = optim.Adam(D_wgan.parameters(), lr=D_LR, betas=(0.5, 0.9))
loader = DataLoader(TensorDataset(X_real_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for epoch in range(GAN_EPOCHS):
    d_ep, g_ep = [], []
    for i, (x_real,) in enumerate(loader):
        x_real = x_real.to(DEVICE); bs = x_real.size(0)
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        x_fake = G_wgan(z).detach()
        d_loss = D_wgan(x_fake).mean() - D_wgan(x_real).mean() + LAMBDA_GP * gradient_penalty(D_wgan, x_real, x_fake)
        opt_Dw.zero_grad(); d_loss.backward(); opt_Dw.step()
        d_ep.append(d_loss.item())
        if i % N_CRITIC == 0:
            z = torch.randn(bs, Z_DIM, device=DEVICE)
            g_loss = -D_wgan(G_wgan(z)).mean()
            opt_Gw.zero_grad(); g_loss.backward(); opt_Gw.step()
            g_ep.append(g_loss.item())
    history['wgan']['D'].append(np.mean(d_ep))
    history['wgan']['G'].append(np.mean(g_ep) if g_ep else 0)
    if (epoch+1) % 50 == 0:
        print(f"  WGAN-GP {epoch+1:3d}/{GAN_EPOCHS} D={history['wgan']['D'][-1]:.4f} G={history['wgan']['G'][-1]:.4f}")

G_wgan.eval()

# --- BA-cWGAN-GP ---
G_ba = Generator(Z_DIM+R_DIM, 13).to(DEVICE)
D_ba = Discriminator(13).to(DEVICE)
opt_Gb = optim.Adam(G_ba.parameters(), lr=G_LR, betas=(0.5, 0.9))
opt_Db = optim.Adam(D_ba.parameters(), lr=D_LR, betas=(0.5, 0.9))
w_np = weights; N = len(X_real_t)

for epoch in range(GAN_EPOCHS):
    d_ep, g_ep, b_ep, r_ep = [], [], [], []
    idxs = np.random.choice(N, BATCH_SIZE*80, p=w_np, replace=True)
    np.random.shuffle(idxs)
    d_count = 0
    for b_start in range(0, len(idxs)-BATCH_SIZE+1, BATCH_SIZE):
        idx = idxs[b_start:b_start+BATCH_SIZE]
        x_real = X_real_t[idx].to(DEVICE); r_b = r_all_t[idx].to(DEVICE); bs = x_real.size(0)
        # D
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        x_fake = G_ba(torch.cat([z, r_b], dim=1)).detach()
        d_loss = D_ba(x_fake).mean() - D_ba(x_real).mean() + LAMBDA_GP * gradient_penalty(D_ba, x_real, x_fake)
        opt_Db.zero_grad(); d_loss.backward(); opt_Db.step()
        d_ep.append(d_loss.item()); d_count += 1
        # G
        if d_count % N_CRITIC == 0:
            z = torch.randn(bs, Z_DIM, device=DEVICE)
            x_fake = G_ba(torch.cat([z, r_b], dim=1))
            g_adv = -D_ba(x_fake).mean()
            b_loss = -torch.abs(base_clf(x_fake) - 0.5).mean()
            r_loss = torch.abs(surr_speed(x_fake) - r_b).mean()
            total = g_adv + LAMBDA_BOUNDARY * b_loss + LAMBDA_R * r_loss
            opt_Gb.zero_grad(); total.backward(); opt_Gb.step()
            g_ep.append(g_adv.item()); b_ep.append(b_loss.item()); r_ep.append(r_loss.item())
    history['ba']['D'].append(np.mean(d_ep))
    history['ba']['G'].append(np.mean(g_ep) if g_ep else 0)
    history['ba']['B'].append(np.mean(b_ep) if b_ep else 0)
    history['ba']['R'].append(np.mean(r_ep) if r_ep else 0)
    if (epoch+1) % 50 == 0:
        print(f"  BA-cWGAN-GP {epoch+1:3d}/{GAN_EPOCHS} D={history['ba']['D'][-1]:.4f} G={history['ba']['G'][-1]:.4f} B={history['ba']['B'][-1]:.4f} R={history['ba']['R'][-1]:.4f}")

G_ba.eval()

# ===========================================================================
# Phase 4: Generation & Evaluation
# ===========================================================================
print("\nPhase 4: Generation & Evaluation...")

@torch.no_grad()
def gen_samples(G, n, cond=False):
    out = []; total = 0; bs = 1024
    while total < n:
        nb = min(bs, n - total)
        z = torch.randn(nb, Z_DIM, device=DEVICE)
        if cond:
            rs = r_all_t[np.random.choice(len(r_all_t), nb)]
            xf = G(torch.cat([z, rs], dim=1))
        else:
            xf = G(z)
        out.append(xf.cpu().numpy()); total += len(xf)
    return np.vstack(out)

def evaluate(name, X_tr, y_tr, X_te, y_te):
    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    sw = n_neg / n_pos if n_pos > 0 else 1
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=sw, random_state=SEED, eval_metric='logloss', verbosity=0
    )
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]; pred = clf.predict(X_te)
    idx_sort = np.argsort(prob); y_s = y_te[idx_sort]
    n1 = y_s.sum(); n0 = len(y_s) - n1
    ks = np.max(np.abs(np.cumsum(y_s)/n1 - np.cumsum(1-y_s)/n0)) if n1>0 and n0>0 else 0
    return {'AUC': roc_auc_score(y_te, prob), 'F1': f1_score(y_te, pred),
            'Brier': brier_score_loss(y_te, prob), 'KS': ks, 'prob': prob}

n_syn = int(len(minority_idx) * 3)

# Build datasets
X1, y1 = X_train_full_scaled.copy(), y_train_full.copy()

n_maj_sm = min(len(majority_idx), len(minority_idx)*10)
maj_sm = np.random.choice(len(majority_idx), n_maj_sm, replace=False)
X_sm_in = np.vstack([X_minority_scaled, X_majority_scaled[maj_sm]])
y_sm_in = np.hstack([np.ones(len(minority_idx)), np.zeros(n_maj_sm)])
sm = SMOTE(sampling_strategy=0.3, random_state=SEED)
Xs, ys = sm.fit_resample(X_sm_in, y_sm_in)
rest = np.setdiff1d(np.arange(len(majority_idx)), maj_sm)
X2 = np.vstack([Xs, X_majority_scaled[rest]])
y2 = np.hstack([ys, np.zeros(len(rest))])

syn_w = gen_samples(G_wgan, n_syn, cond=False)
X3 = np.vstack([X_train_full_scaled, syn_w])
y3 = np.hstack([y_train_full, np.ones(len(syn_w))])

syn_ba = gen_samples(G_ba, n_syn, cond=True)
X4 = np.vstack([X_train_full_scaled, syn_ba])
y4 = np.hstack([y_train_full, np.ones(len(syn_ba))])

# Boundary enrichment
prob_w = base_xgb.predict_proba(syn_w)[:, 1]
prob_ba = base_xgb.predict_proba(syn_ba)[:, 1]
br_w = ((prob_w >= 0.3) & (prob_w <= 0.7)).mean()
br_ba = ((prob_ba >= 0.3) & (prob_ba <= 0.7)).mean()
br_real = ((d_xgb < 0.2)).mean()

# Evaluate
results = {}
for name, X_tr, y_tr in [
    ("No Sampling", X1, y1), ("SMOTE", X2, y2),
    ("WGAN-GP", X3, y3), ("BA-cWGAN-GP", X4, y4)
]:
    print(f"  Evaluating {name}...")
    results[name] = evaluate(name, X_tr, y_tr, X_val_scaled, y_val)

# ===========================================================================
# FIGURE 1: GAN Training Loss Curves
# ===========================================================================
print("\nGenerating Figure 1: GAN Training Loss Curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# WGAN-GP
ax = axes[0]
epochs = np.arange(1, GAN_EPOCHS+1)
ax.plot(epochs, history['wgan']['D'], 'b-', alpha=0.7, linewidth=1, label='D_loss')
ax.plot(epochs, history['wgan']['G'], 'r-', alpha=0.7, linewidth=1, label='G_loss')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('WGAN-GP Training Loss')
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

# BA-cWGAN-GP
ax = axes[1]
ax.plot(epochs, history['ba']['D'], 'b-', alpha=0.5, linewidth=1, label='D_loss')
ax.plot(epochs, history['ba']['G'], 'r-', alpha=0.7, linewidth=1, label='G_adv (adversarial)')
ax.plot(epochs, history['ba']['B'], 'g-', alpha=0.8, linewidth=1, label='B_loss (boundary)')
ax.plot(epochs, history['ba']['R'], 'orange', alpha=0.6, linewidth=1, label='R_loss (r consistency)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('BA-cWGAN-GP Training Loss')
ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle('GAN Training Loss Curves (200 Epochs, CPU)', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '01_gan_training_loss.png'), bbox_inches='tight')
plt.close()
print("  Saved: 01_gan_training_loss.png")

# ===========================================================================
# FIGURE 2: Boundary Enrichment Comparison
# ===========================================================================
print("Generating Figure 2: Boundary Enrichment...")
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Real Defaults\n(in GAN train)', 'WGAN-GP\n(blind generation)', 'BA-cWGAN-GP\n(our method)']
values = [br_real*100, br_w*100, br_ba*100]
colors = ['#6c757d', '#f0ad4e', '#28a745']
bars = ax.bar(methods, values, color=colors, edgecolor='white', linewidth=1.2, width=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Boundary Enrichment (%)', fontsize=12)
ax.set_title('Boundary Enrichment Comparison\n(% of samples near decision boundary, d < 0.2)', fontweight='bold')
ax.set_ylim(0, 115)
ax.axhline(y=br_real*100, color='#6c757d', linestyle='--', alpha=0.5, linewidth=1)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Annotation
ax.annotate('Natural boundary\nrate in real data', xy=(0, br_real*100),
            xytext=(0.5, br_real*100+20), fontsize=9, color='#6c757d',
            arrowprops=dict(arrowstyle='->', color='#6c757d', alpha=0.6))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '02_boundary_enrichment.png'), bbox_inches='tight')
plt.close()
print("  Saved: 02_boundary_enrichment.png")

# ===========================================================================
# FIGURE 3: Performance Comparison (AUC, F1, KS, Brier)
# ===========================================================================
print("Generating Figure 3: Performance Comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
method_names = ['No Sampling', 'SMOTE', 'WGAN-GP', 'BA-cWGAN-GP']
colors = ['#6c757d', '#f0ad4e', '#5bc0de', '#28a745']
metrics = ['AUC', 'F1', 'KS', 'Brier']
titles = ['AUC (higher is better)', 'F1 Score', 'KS Statistic', 'Brier Score (lower is better)']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx//2, idx%2]
    vals = [results[m][metric] for m in method_names]
    bars = ax.bar(method_names, vals, color=colors, edgecolor='white', linewidth=1)

    for bar, val in zip(bars, vals):
        offset = 0.002 if metric != 'Brier' else 0.003
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight best
    if metric == 'Brier':
        best_idx = np.argmin(vals)
    else:
        best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)

    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', rotation=15)

fig.suptitle('Model Performance Comparison on GAN Validation Set\n(XGBoost referee, fixed hyperparameters)', fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '03_performance_comparison.png'), bbox_inches='tight')
plt.close()
print("  Saved: 03_performance_comparison.png")

# ===========================================================================
# FIGURE 4: r Distribution (Speed Score)
# ===========================================================================
print("Generating Figure 4: r Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# r distribution
ax = axes[0]
ax.hist(r_all, bins=50, color='#5bc0de', edgecolor='white', alpha=0.8)
ax.axvline(x=r_all.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean = {r_all.mean():.3f}')
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, label='r = 0.5')
ax.set_xlabel('r (Early Default Probability)'); ax.set_ylabel('Count')
ax.set_title('Distribution of Default Propensity Score r')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# r by speed label
ax = axes[1]
speed_label_map = gan_train.loc[has_speed, ['speed_label']].copy()
speed_label_map['r'] = speed_clf.predict_proba(X_speed_scaled)[:, 1]
r_early = speed_label_map[speed_label_map['speed_label']==1]['r']
r_late = speed_label_map[speed_label_map['speed_label']==0]['r']
ax.hist(r_early, bins=40, color='#dc3545', alpha=0.6, label=f'Early Default (n={len(r_early):,})', edgecolor='white')
ax.hist(r_late, bins=40, color='#0d6efd', alpha=0.6, label=f'Late Default (n={len(r_late):,})', edgecolor='white')
ax.set_xlabel('r (Predicted Early Default Probability)'); ax.set_ylabel('Count')
ax.set_title('r Distribution by True Speed Label')
ax.legend(); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Phase 1 Output: Default Propensity Score (r)', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '04_r_distribution.png'), bbox_inches='tight')
plt.close()
print("  Saved: 04_r_distribution.png")

# ===========================================================================
# FIGURE 5: Boundary Distance Comparison (Real vs Generated)
# ===========================================================================
print("Generating Figure 5: Boundary Distance Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

bins_d = np.linspace(0, 0.5, 40)

# Real minority
ax = axes[0]
ax.hist(d_xgb, bins=bins_d, color='#6c757d', edgecolor='white', alpha=0.8)
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5, label=f'd<0.2: {br_real*100:.1f}%')
ax.set_xlabel('d = |p - 0.5|'); ax.set_ylabel('Count')
ax.set_title('Real Defaults (Minority Class)')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# WGAN-GP generated
ax = axes[1]
d_w = np.abs(prob_w - 0.5)
ax.hist(d_w, bins=bins_d, color='#f0ad4e', edgecolor='white', alpha=0.8)
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5, label=f'd<0.2: {br_w*100:.1f}%')
ax.set_xlabel('d = |p - 0.5|'); ax.set_ylabel('Count')
ax.set_title('WGAN-GP Generated')
ax.legend(); ax.grid(axis='y', alpha=0.3)

# BA-cWGAN-GP generated
ax = axes[2]
d_ba = np.abs(prob_ba - 0.5)
ax.hist(d_ba, bins=bins_d, color='#28a745', edgecolor='white', alpha=0.8)
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=1.5, label=f'd<0.2: {br_ba*100:.1f}%')
ax.set_xlabel('d = |p - 0.5|'); ax.set_ylabel('Count')
ax.set_title('BA-cWGAN-GP Generated (Filtered)')
ax.legend(); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Boundary Distance (d) Distribution: Real vs Generated\n(d smaller = closer to decision boundary = more valuable)', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '05_boundary_distance_distribution.png'), bbox_inches='tight')
plt.close()
print("  Saved: 05_boundary_distance_distribution.png")

# ===========================================================================
# FIGURE 6: PCA Visualization of Real vs Generated Samples
# ===========================================================================
print("Generating Figure 6: PCA Visualization...")
# Sample 5000 each
n_sample = 5000
real_sample = X_minority_scaled[np.random.choice(len(X_minority_scaled), min(n_sample, len(X_minority_scaled)), replace=False)]
gen_w_sample = syn_w[np.random.choice(len(syn_w), min(n_sample, len(syn_w)), replace=False)]
gen_ba_sample = syn_ba[np.random.choice(len(syn_ba), min(n_sample, len(syn_ba)), replace=False)]

all_for_pca = np.vstack([real_sample, gen_w_sample, gen_ba_sample])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_for_pca)

n_r = len(real_sample); n_w = len(gen_w_sample); n_ba = len(gen_ba_sample)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# WGAN-GP vs Real
ax = axes[0]
ax.scatter(pca_result[n_r:n_r+n_w, 0], pca_result[n_r:n_r+n_w, 1],
           c='#f0ad4e', alpha=0.3, s=3, label=f'WGAN-GP Generated (n={n_w})')
ax.scatter(pca_result[:n_r, 0], pca_result[:n_r, 1],
           c='#6c757d', alpha=0.5, s=3, label=f'Real Defaults (n={n_r})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('WGAN-GP: Real vs Generated (PCA)')
ax.legend(markerscale=5); ax.grid(alpha=0.3)

# BA-cWGAN-GP vs Real
ax = axes[1]
ax.scatter(pca_result[n_r+n_w:, 0], pca_result[n_r+n_w:, 1],
           c='#28a745', alpha=0.3, s=3, label=f'BA-cWGAN-GP Generated (n={n_ba})')
ax.scatter(pca_result[:n_r, 0], pca_result[:n_r, 1],
           c='#6c757d', alpha=0.5, s=3, label=f'Real Defaults (n={n_r})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('BA-cWGAN-GP: Real vs Generated (PCA)')
ax.legend(markerscale=5); ax.grid(alpha=0.3)

fig.suptitle('PCA Projection: Real vs Generated Default Samples\n(First 2 principal components)', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '06_pca_visualization.png'), bbox_inches='tight')
plt.close()
print("  Saved: 06_pca_visualization.png")

# ===========================================================================
# FIGURE 7: Summary Dashboard
# ===========================================================================
print("Generating Figure 7: Summary Dashboard...")
fig = plt.figure(figsize=(16, 12))

# Loss curves (small)
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(epochs, history['wgan']['D'], 'b-', alpha=0.5, lw=0.8, label='WGAN D')
ax1.plot(epochs, history['wgan']['G'], 'r-', alpha=0.5, lw=0.8, label='WGAN G')
ax1.plot(epochs, history['ba']['D'], 'b-', alpha=0.3, lw=0.5, label='BA D')
ax1.plot(epochs, history['ba']['G'], 'r-', alpha=0.3, lw=0.5, label='BA G')
ax1.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax1.set_title('GAN Training Loss'); ax1.legend(fontsize=6, loc='lower left'); ax1.grid(alpha=0.2)

# BA-specific losses
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(epochs, history['ba']['B'], 'g-', lw=1, label='Boundary Loss')
ax2.plot(epochs, history['ba']['R'], 'orange', lw=1, label='R Consistency Loss')
ax2.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax2.set_title('BA-cWGAN-GP Auxiliary Losses'); ax2.legend(fontsize=7); ax2.grid(alpha=0.2)

# Boundary enrichment
ax3 = fig.add_subplot(3, 3, 3)
vals_b = [br_real*100, br_w*100, br_ba*100]
bars = ax3.bar(['Real', 'WGAN-GP', 'BA-cWGAN'], vals_b, color=['#6c757d','#f0ad4e','#28a745'])
for b, v in zip(bars, vals_b): ax3.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.0f}%', ha='center', fontweight='bold')
ax3.set_title('Boundary Enrichment %'); ax3.set_ylim(0, 115); ax3.grid(axis='y', alpha=0.3)

# AUC comparison
ax4 = fig.add_subplot(3, 3, 4)
auc_vals = [results[m]['AUC'] for m in method_names]
bars = ax4.bar(method_names, auc_vals, color=colors)
for b, v in zip(bars, auc_vals): ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ax4.set_title('AUC Comparison'); ax4.tick_params(axis='x', rotation=20); ax4.grid(axis='y', alpha=0.3)

# F1 comparison
ax5 = fig.add_subplot(3, 3, 5)
f1_vals = [results[m]['F1'] for m in method_names]
bars = ax5.bar(method_names, f1_vals, color=colors)
for b, v in zip(bars, f1_vals): ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ax5.set_title('F1 Score'); ax5.tick_params(axis='x', rotation=20); ax5.grid(axis='y', alpha=0.3)

# r distribution
ax6 = fig.add_subplot(3, 3, 6)
ax6.hist(r_early, bins=30, color='#dc3545', alpha=0.5, label=f'Early')
ax6.hist(r_late, bins=30, color='#0d6efd', alpha=0.5, label=f'Late')
ax6.set_title('r Distribution by Speed Label'); ax6.legend(fontsize=7)

# KS comparison
ax7 = fig.add_subplot(3, 3, 7)
ks_vals = [results[m]['KS'] for m in method_names]
bars = ax7.bar(method_names, ks_vals, color=colors)
for b, v in zip(bars, ks_vals): ax7.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ax7.set_title('KS Statistic'); ax7.tick_params(axis='x', rotation=20); ax7.grid(axis='y', alpha=0.3)

# Brier comparison
ax8 = fig.add_subplot(3, 3, 8)
br_vals = [results[m]['Brier'] for m in method_names]
bars = ax8.bar(method_names, br_vals, color=colors)
for b, v in zip(bars, br_vals): ax8.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ax8.set_title('Brier Score (Lower Better)'); ax8.tick_params(axis='x', rotation=20); ax8.grid(axis='y', alpha=0.3)

# Text summary
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
summary_text = (
    "BA-cWGAN-GP Feasibility Summary\n"
    "=" * 35 + "\n\n"
    f"Phase 1: speed_clf on {len(np.where(has_speed)[0]):,} loans\n"
    f"   NN surrogate r = 0.822\n\n"
    f"Phase 2: XGBoost base AUC = 0.735\n"
    f"   NN surrogate r = 0.983\n"
    f"   Boundary agreement = 94.2%\n\n"
    f"Phase 3: Boundary Enrichment\n"
    f"   WGAN-GP: {br_w*100:.1f}% → BA-cWGAN-GP: {br_ba*100:.1f}%\n\n"
    f"Phase 4: AUC Δ = {results['BA-cWGAN-GP']['AUC']-results['No Sampling']['AUC']:+.4f}\n"
    f"   Brier Δ = {results['BA-cWGAN-GP']['Brier']-results['No Sampling']['Brier']:+.4f}"
)
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=8.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

fig.suptitle('BA-cWGAN-GP Pre-Experiment Dashboard', fontweight='bold', fontsize=16, y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '07_summary_dashboard.png'), bbox_inches='tight')
plt.close()
print("  Saved: 07_summary_dashboard.png")

# ===========================================================================
# Done
# ===========================================================================
print(f"\n{'='*60}")
print(f"All 7 figures saved to: {FIG_DIR}")
print(f"{'='*60}")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
