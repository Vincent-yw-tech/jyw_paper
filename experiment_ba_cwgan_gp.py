# -*- coding: utf-8 -*-
"""
BA-cWGAN-GP Feasibility Experiment (Improved)
==============================================
Key improvement: XGBoost boundary → NN surrogate (knowledge distillation)
so the GAN boundary loss uses a XGBoost-quality boundary.

Compares: No Sampling | SMOTE | WGAN-GP | BA-cWGAN-GP
Fixed XGBoost classifier as referee. Evaluates on held-out GAN validation set.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ===========================================================================
# Configuration
# ===========================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
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

print(f"Device: {DEVICE}")
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ===========================================================================
# Load Data
# ===========================================================================
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

gan_train = pd.read_csv(os.path.join(DATA_DIR, 'gan_train_set.csv'))
gan_val = pd.read_csv(os.path.join(DATA_DIR, 'gan_val_set.csv'))

X_train_full = gan_train[FEATURE_COLS].values.astype(np.float32)
y_train_full = gan_train['DFlag'].values.astype(np.int64)
X_val = gan_val[FEATURE_COLS].values.astype(np.float32)
y_val = gan_val['DFlag'].values.astype(np.int64)

minority_idx = np.where(y_train_full == 1)[0]
majority_idx = np.where(y_train_full == 0)[0]

has_speed = gan_train['speed_label'].notna().values & (y_train_full == 1)
speed_idx = np.where(has_speed)[0]
speed_labels = gan_train.loc[has_speed, 'speed_label'].values.astype(np.int64)

print(f"GAN Train: {len(X_train_full):,} samples ({len(minority_idx):,} defaults)")
print(f"  Defaults with speed label: {len(speed_idx):,}")
print(f"GAN Val:   {len(X_val):,} samples ({y_val.sum():,} defaults)")

# Standardize
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)
X_minority_scaled = X_train_full_scaled[minority_idx]
X_majority_scaled = X_train_full_scaled[majority_idx]
X_speed_scaled = X_train_full_scaled[speed_idx]

# ===========================================================================
# Phase 1: Speed Classifier
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 1: SPEED CLASSIFIER (XGBoost + NN Surrogate)")
print("=" * 60)

speed_clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=SEED, eval_metric='logloss', verbosity=0
)
speed_clf.fit(X_speed_scaled, speed_labels)
r_all = speed_clf.predict_proba(X_minority_scaled)[:, 1].astype(np.float32)
print(f"Speed classifier: {len(speed_idx):,} labeled defaults")
print(f"r: mean={r_all.mean():.3f} std={r_all.std():.3f}")

# NN surrogate for speed_clf
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
Xm_t = torch.FloatTensor(X_minority_scaled)
r_t = torch.FloatTensor(r_all).unsqueeze(1)
ds = TensorDataset(Xm_t, r_t)
dl = DataLoader(ds, batch_size=512, shuffle=True)

surr_speed.train()
for epoch in range(200):
    tot = 0
    for xb, rb in dl:
        xb, rb = xb.to(DEVICE), rb.to(DEVICE)
        opt.zero_grad(); loss = loss_fn(surr_speed(xb), rb)
        loss.backward(); opt.step(); tot += loss.item()
    if (epoch+1) % 50 == 0:
        print(f"  Surrogate speed epoch {epoch+1:3d}/200 loss={tot/len(dl):.6f}")

surr_speed.eval()
with torch.no_grad():
    rp = surr_speed(Xm_t.to(DEVICE)).cpu().numpy().flatten()
    corr = np.corrcoef(r_all, rp)[0,1]
print(f"  Surrogate r={corr:.4f} MAE={np.abs(r_all-rp).mean():.4f}")

# ===========================================================================
# Phase 2: Base Classifier (XGBoost → NN Surrogate)
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 2: BASE CLASSIFIER (XGBoost + NN Surrogate)")
print("=" * 60)

# Use balanced subset for XGBoost base
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

# Evaluate XGBoost base
p_val_xgb = base_xgb.predict_proba(X_val_scaled)[:, 1]
auc_xgb = roc_auc_score(y_val, p_val_xgb)
print(f"XGBoost base classifier AUC on GAN val: {auc_xgb:.4f}")

# Boundary from XGBoost
p_min_xgb = base_xgb.predict_proba(X_minority_scaled)[:, 1].astype(np.float32)
d_xgb = np.abs(p_min_xgb - 0.5)
weights = 1.0 / (d_xgb + 0.01)
weights = weights / weights.sum()
n_boundary = (d_xgb < 0.2).sum()
print(f"Boundary (XGBoost): samples d<0.2 = {n_boundary}/{len(d_xgb)} ({n_boundary/len(d_xgb)*100:.1f}%)")

# Train NN surrogate to mimic XGBoost base predictions
surr_base = Surrogate(13, 256).to(DEVICE)
opt_b = optim.Adam(surr_base.parameters(), lr=1e-3)
p_xgb_t = torch.FloatTensor(p_min_xgb).unsqueeze(1)
ds_b = TensorDataset(Xm_t, p_xgb_t)
dl_b = DataLoader(ds_b, batch_size=512, shuffle=True)

surr_base.train()
for epoch in range(300):
    tot = 0
    for xb, pb in dl_b:
        xb, pb = xb.to(DEVICE), pb.to(DEVICE)
        opt_b.zero_grad(); loss = loss_fn(surr_base(xb), pb)
        loss.backward(); opt_b.step(); tot += loss.item()
    if (epoch+1) % 100 == 0:
        print(f"  Surrogate base epoch {epoch+1:3d}/300 loss={tot/len(dl_b):.6f}")

surr_base.eval()
with torch.no_grad():
    pb_pred = surr_base(Xm_t.to(DEVICE)).cpu().numpy().flatten()
    corr_b = np.corrcoef(p_min_xgb, pb_pred)[0,1]
    # Check boundary agreement
    d_surr = np.abs(pb_pred - 0.5)
    agree = ((d_xgb < 0.2) == (d_surr < 0.2)).mean()
print(f"  Surrogate r={corr_b:.4f} Boundary agreement={agree:.4f}")

# Use surrogate for GAN boundary loss
base_clf = surr_base  # differentiable!

# ===========================================================================
# Phase 3: GAN Models
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 3: GAN TRAINING")
print("=" * 60)

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
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat)
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

X_real_t = Xm_t.to(DEVICE)
r_all_t = torch.FloatTensor(r_all).unsqueeze(1).to(DEVICE)
weights_t = torch.FloatTensor(weights).to(DEVICE)

def train_wgan_gp(X_real, epochs, label):
    print(f"\n--- {label} ({epochs} epochs) ---")
    G = Generator(Z_DIM, 13).to(DEVICE)
    D = Discriminator(13).to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=G_LR, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=D_LR, betas=(0.5, 0.9))
    loader = DataLoader(TensorDataset(X_real), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        d_losses, g_losses = [], []
        for i, (x_real,) in enumerate(loader):
            x_real = x_real.to(DEVICE); bs = x_real.size(0)
            # D
            z = torch.randn(bs, Z_DIM, device=DEVICE)
            x_fake = G(z).detach()
            gp = gradient_penalty(D, x_real, x_fake)
            d_loss = D(x_fake).mean() - D(x_real).mean() + LAMBDA_GP * gp
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            d_losses.append(d_loss.item())
            # G
            if i % N_CRITIC == 0:
                z = torch.randn(bs, Z_DIM, device=DEVICE)
                g_loss = -D(G(z)).mean()
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()
                g_losses.append(g_loss.item())
        if (epoch+1) % 50 == 0:
            print(f"  {label} E {epoch+1:4d}/{epochs} D={np.mean(d_losses):.4f} G={np.mean(g_losses):.4f}")
    G.eval(); return G

def train_ba_cwgan_gp(X_real, r_vals, w_vals, epochs, label):
    print(f"\n--- {label} ({epochs} epochs) ---")
    G = Generator(Z_DIM+R_DIM, 13).to(DEVICE)
    D = Discriminator(13).to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=G_LR, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=D_LR, betas=(0.5, 0.9))

    w_np = w_vals.cpu().numpy()
    w_np = w_np / w_np.sum()
    N = len(X_real)

    for epoch in range(epochs):
        d_losses, g_losses, b_losses, r_losses = [], [], [], []
        idxs = np.random.choice(N, BATCH_SIZE*80, p=w_np, replace=True)
        np.random.shuffle(idxs)

        for b_start in range(0, len(idxs)-BATCH_SIZE+1, BATCH_SIZE):
            idx = idxs[b_start:b_start+BATCH_SIZE]
            x_real = X_real[idx].to(DEVICE); r_b = r_vals[idx].to(DEVICE)
            bs = x_real.size(0)

            # D
            z = torch.randn(bs, Z_DIM, device=DEVICE)
            zc = torch.cat([z, r_b], dim=1)
            x_fake = G(zc).detach()
            gp = gradient_penalty(D, x_real, x_fake)
            d_loss = D(x_fake).mean() - D(x_real).mean() + LAMBDA_GP * gp
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            d_losses.append(d_loss.item())

            # G
            if len(d_losses) % N_CRITIC == 0:
                z = torch.randn(bs, Z_DIM, device=DEVICE)
                zc = torch.cat([z, r_b], dim=1)
                x_fake = G(zc)
                g_adv = -D(x_fake).mean()
                # Boundary loss via NN surrogate (distilled from XGBoost)
                b_loss = -torch.abs(base_clf(x_fake) - 0.5).mean()
                # r consistency via NN surrogate
                r_loss = torch.abs(surr_speed(x_fake) - r_b).mean()
                total = g_adv + LAMBDA_BOUNDARY * b_loss + LAMBDA_R * r_loss
                opt_G.zero_grad(); total.backward(); opt_G.step()
                g_losses.append(g_adv.item())
                b_losses.append(b_loss.item())
                r_losses.append(r_loss.item())

        if (epoch+1) % 50 == 0:
            print(f"  {label} E {epoch+1:4d}/{epochs} D={np.mean(d_losses):.4f} "
                  f"G={np.mean(g_losses):.4f} B={np.mean(b_losses):.4f} R={np.mean(r_losses):.4f}")
    G.eval(); return G

t0 = time.time()
G_wgan = train_wgan_gp(X_real_t, GAN_EPOCHS, "WGAN-GP")
print(f"  Done in {time.time()-t0:.0f}s")

t0 = time.time()
G_ba = train_ba_cwgan_gp(X_real_t, r_all_t, weights_t, GAN_EPOCHS, "BA-cWGAN-GP")
print(f"  Done in {time.time()-t0:.0f}s")

# ===========================================================================
# Phase 4: Generation & Evaluation
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 4: GENERATION & EVALUATION")
print("=" * 60)

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
        out.append(xf.cpu().numpy())
        total += len(xf)
    return np.vstack(out)

@torch.no_grad()
def gen_filtered(G, n_target, cond=False):
    """Generate samples, filter by XGBoost boundary [0.3, 0.7]."""
    all_s = []; total = 0
    for _ in range(15):
        if total >= n_target: break
        batch = gen_samples(G, min(50000, n_target*3), cond)
        prob = base_xgb.predict_proba(batch)[:, 1]
        mask = (prob >= 0.3) & (prob <= 0.7)
        filtered = batch[mask]
        if len(filtered) > 0:
            all_s.append(filtered)
            total += len(filtered)
    if total < n_target:
        extra = gen_samples(G, n_target - total, cond)
        all_s.append(extra)
    return np.vstack(all_s)[:n_target]

def evaluate(name, X_tr, y_tr, X_te, y_te):
    n_neg = (y_tr==0).sum(); n_pos = y_tr.sum()
    sw = n_neg / n_pos if n_pos > 0 else 1
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=sw, random_state=SEED, eval_metric='logloss', verbosity=0
    )
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = clf.predict(X_te)
    # KS
    idx_sort = np.argsort(prob)
    y_s = y_te[idx_sort]; n1 = y_s.sum(); n0 = len(y_s)-n1
    if n1 > 0 and n0 > 0:
        ks = np.max(np.abs(np.cumsum(y_s)/n1 - np.cumsum(1-y_s)/n0))
    else:
        ks = 0
    return {
        'AUC': roc_auc_score(y_te, prob),
        'F1': f1_score(y_te, pred),
        'Brier': brier_score_loss(y_te, prob),
        'KS': ks
    }

n_syn = int(len(minority_idx) * 3)
print(f"\nReal defaults: {len(minority_idx):,}, Target synthetic: {n_syn:,}")

# Method 1: No Sampling
print("\n[1/4] No Sampling")
X1, y1 = X_train_full_scaled.copy(), y_train_full.copy()

# Method 2: SMOTE
print("[2/4] SMOTE")
n_maj_sm = min(len(majority_idx), len(minority_idx)*10)
maj_sm = np.random.choice(len(majority_idx), n_maj_sm, replace=False)
X_sm_in = np.vstack([X_minority_scaled, X_majority_scaled[maj_sm]])
y_sm_in = np.hstack([np.ones(len(minority_idx)), np.zeros(n_maj_sm)])
sm = SMOTE(sampling_strategy=0.3, random_state=SEED)
Xs, ys = sm.fit_resample(X_sm_in, y_sm_in)
rest = np.setdiff1d(np.arange(len(majority_idx)), maj_sm)
X2 = np.vstack([Xs, X_majority_scaled[rest]])
y2 = np.hstack([ys, np.zeros(len(rest))])
print(f"  SMOTE: {len(X2):,} samples (def rate={y2.mean():.4f})")

# Method 3: WGAN-GP
print("[3/4] WGAN-GP")
syn_w = gen_samples(G_wgan, n_syn, cond=False)
br_w = ((base_xgb.predict_proba(syn_w)[:,1] >= 0.3) & (base_xgb.predict_proba(syn_w)[:,1] <= 0.7)).mean()
print(f"  Generated {len(syn_w):,}, boundary enrichment (XGBoost): {br_w:.4f}")
X3 = np.vstack([X_train_full_scaled, syn_w])
y3 = np.hstack([y_train_full, np.ones(len(syn_w))])

# Method 4: BA-cWGAN-GP
print("[4/4] BA-cWGAN-GP")
syn_ba = gen_filtered(G_ba, n_syn, cond=True)
br_ba = ((base_xgb.predict_proba(syn_ba)[:,1] >= 0.3) & (base_xgb.predict_proba(syn_ba)[:,1] <= 0.7)).mean()
print(f"  Generated {len(syn_ba):,}, boundary enrichment (XGBoost): {br_ba:.4f}")
X4 = np.vstack([X_train_full_scaled, syn_ba])
y4 = np.hstack([y_train_full, np.ones(len(syn_ba))])

# ===========================================================================
# Results
# ===========================================================================
print("\n" + "=" * 60)
print("RESULTS (GAN Validation Set)")
print("=" * 60)

datasets = [
    ("No Sampling", X1, y1),
    ("SMOTE", X2, y2),
    ("WGAN-GP", X3, y3),
    ("BA-cWGAN-GP", X4, y4),
]

results = {}
for name, X_tr, y_tr in datasets:
    print(f"Evaluating {name}...")
    results[name] = evaluate(name, X_tr, y_tr, X_val_scaled, y_val)

br = {"No Sampling": float((d_xgb<0.2).mean()),
      "SMOTE": float((d_xgb<0.2).mean()),
      "WGAN-GP": br_w, "BA-cWGAN-GP": br_ba}

hdr = f"{'Method':<16s} {'AUC':>8s} {'F1':>8s} {'KS':>8s} {'Brier':>8s} {'Boundary':>10s}"
print("\n" + hdr)
print("-" * len(hdr))
for n in ["No Sampling","SMOTE","WGAN-GP","BA-cWGAN-GP"]:
    m = results[n]
    print(f"{n:<16s} {m['AUC']:>8.4f} {m['F1']:>8.4f} {m['KS']:>8.4f} "
          f"{m['Brier']:>8.4f} {br[n]:>10.4f}")

baseline = results["No Sampling"]["AUC"]
print(f"\nAUC improvement over No Sampling ({baseline:.4f}):")
for n in ["SMOTE","WGAN-GP","BA-cWGAN-GP"]:
    print(f"  {n:<16s}: {results[n]['AUC']-baseline:+.4f}")

# ===========================================================================
# Save
# ===========================================================================
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save({
    'G_wgan': G_wgan.state_dict(),
    'G_ba': G_ba.state_dict(),
    'surr_base': surr_base.state_dict(),
    'surr_speed': surr_speed.state_dict(),
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
}, os.path.join(MODEL_DIR, 'ba_cwgan_gp_models.pt'))
import joblib
joblib.dump(speed_clf, os.path.join(MODEL_DIR, 'speed_clf_xgb.pkl'))
joblib.dump(base_xgb, os.path.join(MODEL_DIR, 'base_clf_xgb.pkl'))
print(f"\nModels saved to {MODEL_DIR}")
print("Experiment complete!")
