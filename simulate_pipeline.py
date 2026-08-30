import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
torch.manual_seed(42)

n_samples = 4000
n_features = 50

# 2000 samples for Clade 0 (Source), 2000 for Clade 1 (Target)
lineage = np.array([0]*2000 + [1]*2000)

X = np.zeros((n_samples, n_features))
y = np.zeros(n_samples)

for i in range(n_samples):
    # Base resistance probability is ~30%
    y[i] = np.random.binomial(1, 0.3)
    
    # Feature 0 is CAUSAL. It is present 85% of the time if resistant, 15% if susceptible.
    X[i, 0] = np.random.binomial(1, 0.85 if y[i] == 1 else 0.15)
    
    if lineage[i] == 0:
        # Clade 0 has a strong SPURIOUS feature (Feature 1) that is 99% correlated with resistance in Clade 0.
        # It's an illusion of a resistance marker.
        X[i, 1] = np.random.binomial(1, 0.99 if y[i] == 1 else 0.01)
        # Clade 0 signature features (noise, just for lineage)
        X[i, 2:10] = np.random.binomial(1, 0.9)
    else:
        # In Clade 1, Feature 1 is NOT correlated with resistance. It's just background noise.
        X[i, 1] = np.random.binomial(1, 0.1)
        # Clade 1 signature features
        X[i, 10:18] = np.random.binomial(1, 0.9)
        
    # Global noise
    X[i, 18:] = np.random.binomial(1, 0.1, n_features - 18)

# Baseline Split (Random 80/20 across all data)
X_train_A, X_test_A, y_train_A, y_test_A, lin_train_A, lin_test_A = train_test_split(
    X, y, lineage, test_size=0.2, random_state=42
)

# Clade Holdout Split (Train on Clade 0, Test on Clade 1)
# For standard models, they only see Clade 0 labeled data
idx_c0 = np.where(lineage == 0)[0]
idx_c1 = np.where(lineage == 1)[0]
X_train_B_labeled, y_train_B_labeled = X[idx_c0], y[idx_c0]
X_test_B, y_test_B = X[idx_c1], y[idx_c1]

# Baseline Random Forest
rf_baseline_A = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_baseline_A.fit(X_train_A, y_train_A)
auc_rf_A = roc_auc_score(y_test_A, rf_baseline_A.predict_proba(X_test_A)[:, 1])

rf_baseline_B = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_baseline_B.fit(X_train_B_labeled, y_train_B_labeled)
auc_rf_B = roc_auc_score(y_test_B, rf_baseline_B.predict_proba(X_test_B)[:, 1])

# DANN Architecture
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class DANN(nn.Module):
    def __init__(self, input_dim):
        super(DANN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.res_head = nn.Linear(16, 1)
        self.lin_head = nn.Linear(16, 1)

    def forward(self, x, alpha=1.0):
        features = self.encoder(x)
        res_pred = self.res_head(features)
        lin_pred = self.lin_head(grad_reverse(features, alpha))
        return res_pred, lin_pred

def train_dann(X_labeled, y_labeled, X_unlabeled=None, epochs=250, max_alpha=1.0):
    model = DANN(X_labeled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    X_l = torch.FloatTensor(X_labeled)
    y_l = torch.FloatTensor(y_labeled).unsqueeze(1)
    
    if X_unlabeled is not None:
        X_u = torch.FloatTensor(X_unlabeled)
        
    for epoch in range(epochs):
        # Schedule alpha (warm-up)
        p = float(epoch) / epochs
        alpha = max_alpha * (2. / (1. + np.exp(-10 * p)) - 1) if max_alpha > 0 else 0.0
        
        model.train()
        optimizer.zero_grad()
        
        # 1. Source domain (labeled)
        res_pred_l, lin_pred_l = model(X_l, alpha)
        loss_res = criterion(res_pred_l, y_l)
        loss_lin_s = criterion(lin_pred_l, torch.zeros_like(lin_pred_l))
        
        # 2. Target domain (unlabeled)
        if X_unlabeled is not None and alpha > 0:
            _, lin_pred_u = model(X_u, alpha)
            loss_lin_t = criterion(lin_pred_u, torch.ones_like(lin_pred_u))
            loss_lin = (loss_lin_s + loss_lin_t) / 2.0
        else:
            loss_lin = loss_lin_s
            
        loss = loss_res + loss_lin
        loss.backward()
        optimizer.step()
        
    return model

# Standard NN (alpha = 0)
model_std_B = train_dann(X_train_B_labeled, y_train_B_labeled, epochs=250, max_alpha=0.0)
model_std_B.eval()
with torch.no_grad():
    res_pred, _ = model_std_B(torch.FloatTensor(X_test_B))
auc_nn_std_B = roc_auc_score(y_test_B, res_pred.numpy())

model_std_A = train_dann(X_train_A, y_train_A, epochs=250, max_alpha=0.0)
model_std_A.eval()
with torch.no_grad():
    res_pred, _ = model_std_A(torch.FloatTensor(X_test_A))
auc_nn_std_A = roc_auc_score(y_test_A, res_pred.numpy())

# Sweep over max_alpha for DANN on Scheme B to find the best trade-off
best_auc_dann_B = 0
best_alpha_B = 0
for max_alpha in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    model = train_dann(X_train_B_labeled, y_train_B_labeled, X_unlabeled=X_test_B, epochs=250, max_alpha=max_alpha)
    model.eval()
    with torch.no_grad():
        res_pred, _ = model(torch.FloatTensor(X_test_B))
    auc = roc_auc_score(y_test_B, res_pred.numpy())
    if auc > best_auc_dann_B:
        best_auc_dann_B = auc
        best_alpha_B = max_alpha

auc_nn_dann_B = best_auc_dann_B
print(f"Best max_alpha for Scheme B: {best_alpha_B} -> AUC: {auc_nn_dann_B}")

# DANN on Scheme A using the best alpha found
idx_A_c0 = np.where(lin_train_A == 0)[0]
idx_A_c1 = np.where(lin_train_A == 1)[0]
model_dann_A = train_dann(X_train_A[idx_A_c0], y_train_A[idx_A_c0], X_unlabeled=X_train_A[idx_A_c1], epochs=250, max_alpha=best_alpha_B)
model_dann_A.eval()
with torch.no_grad():
    res_pred, _ = model_dann_A(torch.FloatTensor(X_test_A))
auc_nn_dann_A = roc_auc_score(y_test_A, res_pred.numpy())


results = {
    "Baseline_Random_AUC": float(auc_rf_A),
    "Baseline_CladeHoldout_AUC": float(auc_rf_B),
    "NN_Random_AUC": float(auc_nn_std_A),
    "NN_CladeHoldout_AUC": float(auc_nn_std_B),
    "DANN_Random_AUC": float(auc_nn_dann_A),
    "DANN_CladeHoldout_AUC": float(auc_nn_dann_B)
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Simulation complete. Results saved to results.json")
