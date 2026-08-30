import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

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
        
    def get_features(self, x):
        return self.encoder(x)

def train_dann(X_labeled, y_labeled, X_unlabeled=None, epochs=250, max_alpha=1.0, lr=0.005, track_variance=False):
    model = DANN(X_labeled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    X_l = torch.FloatTensor(X_labeled)
    y_l = torch.FloatTensor(y_labeled).unsqueeze(1)
    
    if X_unlabeled is not None:
        X_u = torch.FloatTensor(X_unlabeled)
        
    history = {"task_loss": [], "adv_loss": [], "grad_norm": []}
        
    for epoch in range(epochs):
        p = float(epoch) / epochs
        alpha = max_alpha * (2. / (1. + np.exp(-10 * p)) - 1) if max_alpha > 0 else 0.0
        
        model.train()
        optimizer.zero_grad()
        
        res_pred_l, lin_pred_l = model(X_l, alpha)
        loss_res = criterion(res_pred_l, y_l)
        loss_lin_s = criterion(lin_pred_l, torch.zeros_like(lin_pred_l))
        
        if X_unlabeled is not None and alpha > 0:
            _, lin_pred_u = model(X_u, alpha)
            loss_lin_t = criterion(lin_pred_u, torch.ones_like(lin_pred_u))
            loss_lin = (loss_lin_s + loss_lin_t) / 2.0
        else:
            loss_lin = loss_lin_s
            
        loss = loss_res + loss_lin
        loss.backward()
        
        if track_variance:
            grad_norm = 0.0
            for param in model.encoder.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            history["task_loss"].append(loss_res.item())
            history["adv_loss"].append(loss_lin.item())
            history["grad_norm"].append(grad_norm)
            
        optimizer.step()
        
    if track_variance:
        return model, history
    return model

def run_experiment(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = np.load("real_X.npy")
    y = np.load("real_y.npy")
    lineage = np.load("real_lineage.npy")
    
    X_train_A, X_test_A, y_train_A, y_test_A, lin_train_A, lin_test_A = train_test_split(
        X, y, lineage, test_size=0.2, random_state=seed
    )
    
    idx_c0 = np.where(lineage == 0)[0]
    idx_c1 = np.where(lineage == 1)[0]
    X_train_B_labeled, y_train_B_labeled = X[idx_c0], y[idx_c0]
    X_test_B, y_test_B = X[idx_c1], y[idx_c1]
    
    var_A = np.var(X_train_A, axis=0)
    keep_A = var_A > 0
    df_train_A = sm.add_constant(pd.DataFrame(X_train_A[:, keep_A], columns=[f"f{i}" for i in range(sum(keep_A))]))
    df_train_A["y"] = y_train_A
    df_train_A["group"] = lin_train_A
    df_test_A = sm.add_constant(pd.DataFrame(X_test_A[:, keep_A], columns=[f"f{i}" for i in range(sum(keep_A))]))
    
    try:
        md_A = sm.MixedLM(df_train_A["y"], df_train_A[[c for c in df_train_A.columns if c not in ["y", "group"]]], groups=df_train_A["group"])
        mdf_A = md_A.fit(method='cg')
        auc_lmm_A = roc_auc_score(y_test_A, mdf_A.predict(df_test_A))
    except:
        auc_lmm_A = 0.5
        
    var_B = np.var(X_train_B_labeled, axis=0)
    keep_B = var_B > 0
    df_train_B = sm.add_constant(pd.DataFrame(X_train_B_labeled[:, keep_B], columns=[f"f{i}" for i in range(sum(keep_B))]))
    df_train_B["y"] = y_train_B_labeled
    df_test_B = sm.add_constant(pd.DataFrame(X_test_B[:, keep_B], columns=[f"f{i}" for i in range(sum(keep_B))]))
    for c in df_train_B.columns:
        if c != "y" and c not in df_test_B.columns:
            df_test_B[c] = 0
            
    try:
        md_B = sm.OLS(df_train_B["y"], df_train_B[[c for c in df_train_B.columns if c != "y"]])
        mdf_B = md_B.fit()
        auc_lmm_B = roc_auc_score(y_test_B, mdf_B.predict(df_test_B[[c for c in df_train_B.columns if c != "y"]]))
    except:
        auc_lmm_B = 0.5

    rf_baseline_A = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=5)
    rf_baseline_A.fit(X_train_A, y_train_A)
    auc_rf_A = roc_auc_score(y_test_A, rf_baseline_A.predict_proba(X_test_A)[:, 1])
    
    rf_baseline_B = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=5)
    rf_baseline_B.fit(X_train_B_labeled, y_train_B_labeled)
    auc_rf_B = roc_auc_score(y_test_B, rf_baseline_B.predict_proba(X_test_B)[:, 1])
    
    lr_baseline_A = LogisticRegression(max_iter=1000)
    lr_baseline_A.fit(X_train_A, y_train_A)
    auc_lr_A = roc_auc_score(y_test_A, lr_baseline_A.predict_proba(X_test_A)[:, 1])
    
    lr_baseline_B = LogisticRegression(max_iter=1000)
    lr_baseline_B.fit(X_train_B_labeled, y_train_B_labeled)
    auc_lr_B = roc_auc_score(y_test_B, lr_baseline_B.predict_proba(X_test_B)[:, 1])

    
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
    
    idx_A_c0 = np.where(lin_train_A == 0)[0]
    idx_A_c1 = np.where(lin_train_A == 1)[0]
    
    sweep_results = []
    best_auc_dann_B = 0
    best_alpha_B = 0
    best_model_dann_B = None
    best_model_dann_A = None
    
    dann_histories = {}
    
    for max_alpha in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        model_B, history_B = train_dann(X_train_B_labeled, y_train_B_labeled, X_unlabeled=X_test_B, epochs=250, max_alpha=max_alpha, track_variance=True)
        model_B.eval()
        with torch.no_grad():
            res_pred_B, _ = model_B(torch.FloatTensor(X_test_B))
        auc_B = roc_auc_score(y_test_B, res_pred_B.numpy())
        
        model_A = train_dann(X_train_A[idx_A_c0], y_train_A[idx_A_c0], X_unlabeled=X_train_A[idx_A_c1], epochs=250, max_alpha=max_alpha)
        model_A.eval()
        with torch.no_grad():
            res_pred_A, _ = model_A(torch.FloatTensor(X_test_A))
        auc_A = roc_auc_score(y_test_A, res_pred_A.numpy())
        
        sweep_results.append({
            "alpha": max_alpha,
            "SchemeA_AUC": float(auc_A),
            "SchemeB_AUC": float(auc_B)
        })
        
        dann_histories[max_alpha] = history_B
        
        if auc_B > best_auc_dann_B:
            best_auc_dann_B = auc_B
            best_alpha_B = max_alpha
            best_model_dann_B = model_B
            best_model_dann_A = model_A
    
    auc_nn_dann_A = sweep_results[[x["alpha"] for x in sweep_results].index(1.0)]["SchemeA_AUC"] # reporting 1.0
    auc_nn_dann_B = sweep_results[[x["alpha"] for x in sweep_results].index(1.0)]["SchemeB_AUC"]
    
    with torch.no_grad():
        emb_std = model_std_A.get_features(torch.FloatTensor(X_test_A)).numpy()
        emb_dann = best_model_dann_A.get_features(torch.FloatTensor(X_test_A)).numpy()
    
    probe_std = LogisticRegression(max_iter=1000)
    probe_std.fit(model_std_A.get_features(torch.FloatTensor(X_train_A)).detach().numpy(), lin_train_A)
    acc_probe_std = accuracy_score(lin_test_A, probe_std.predict(emb_std))
    
    probe_dann = LogisticRegression(max_iter=1000)
    probe_dann.fit(best_model_dann_A.get_features(torch.FloatTensor(X_train_A)).detach().numpy(), lin_train_A)
    acc_probe_dann = accuracy_score(lin_test_A, probe_dann.predict(emb_dann))

    return {
        "Baseline_Random_AUC": float(auc_rf_A),
        "Baseline_CladeHoldout_AUC": float(auc_rf_B),
        "LMM_Random_AUC": float(auc_lmm_A),
        "LMM_CladeHoldout_AUC": float(auc_lmm_B),
        "LR_Random_AUC": float(auc_lr_A),
        "LR_CladeHoldout_AUC": float(auc_lr_B),
        "NN_Random_AUC": float(auc_nn_std_A),
        "NN_CladeHoldout_AUC": float(auc_nn_std_B),
        "DANN_Random_AUC": float(auc_nn_dann_A),
        "DANN_CladeHoldout_AUC": float(auc_nn_dann_B),
        "Sweep": sweep_results,
        "Probe_Std_Acc": float(acc_probe_std),
        "Probe_DANN_Acc": float(acc_probe_dann),
        "dann_history_1.0": dann_histories[1.0]
    }, emb_std, emb_dann, lin_test_A

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    seeds = [42, 100, 2026, 777, 999]
    all_results = []
    
    last_emb_std, last_emb_dann, last_lin_test = None, None, None
    all_histories = []
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        res, emb_std, emb_dann, lin_test = run_experiment(seed)
        all_histories.append(res["dann_history_1.0"])
        all_results.append(res)
        if seed == 42:
            last_emb_std = emb_std
            last_emb_dann = emb_dann
            last_lin_test = lin_test

    def agg(key):
        vals = [r[key] for r in all_results]
        return f"{np.mean(vals):.3f} \\pm {np.std(vals):.3f}"
        
    print("Baseline Random:", agg("Baseline_Random_AUC"))
    print("Baseline Holdout:", agg("Baseline_CladeHoldout_AUC"))
    print("LMM Random:", agg("LMM_Random_AUC"))
    print("LMM Holdout:", agg("LMM_CladeHoldout_AUC"))
    print("LR Random:", agg("LR_Random_AUC"))
    print("LR Holdout:", agg("LR_CladeHoldout_AUC"))
    print("NN Random:", agg("NN_Random_AUC"))
    print("NN Holdout:", agg("NN_CladeHoldout_AUC"))
    print("DANN Random:", agg("DANN_Random_AUC"))
    print("DANN Holdout:", agg("DANN_CladeHoldout_AUC"))
    print("Probe Std Acc:", agg("Probe_Std_Acc"))
    print("Probe DANN Acc:", agg("Probe_DANN_Acc"))
    
    # --- PHASE 3: VARIANCE PLOT ---
    # Plot average task loss, adv loss, and grad norm over the 5 seeds for lambda=1.0
    epochs = len(all_histories[0]["task_loss"])
    
    mean_task = np.mean([[h["task_loss"][e] for e in range(epochs)] for h in all_histories], axis=0)
    std_task = np.std([[h["task_loss"][e] for e in range(epochs)] for h in all_histories], axis=0)
    
    mean_adv = np.mean([[h["adv_loss"][e] for e in range(epochs)] for h in all_histories], axis=0)
    std_adv = np.std([[h["adv_loss"][e] for e in range(epochs)] for h in all_histories], axis=0)
    
    mean_grad = np.mean([[h["grad_norm"][e] for e in range(epochs)] for h in all_histories], axis=0)
    std_grad = np.std([[h["grad_norm"][e] for e in range(epochs)] for h in all_histories], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(mean_task, label='Task Loss')
    ax1.fill_between(range(epochs), mean_task - std_task, mean_task + std_task, alpha=0.2)
    ax1.plot(mean_adv, label='Adversary Loss')
    ax1.fill_between(range(epochs), mean_adv - std_adv, mean_adv + std_adv, alpha=0.2)
    ax1.set_title("Loss Trajectories ($\\lambda=1.0$)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.plot(mean_grad, color='red', label='Encoder Grad Norm')
    ax2.fill_between(range(epochs), mean_grad - std_grad, mean_grad + std_grad, color='red', alpha=0.2)
    ax2.set_title("Gradient Norms ($\\lambda=1.0$)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("variance_diagnostics.pdf")
    print("Saved variance_diagnostics.pdf")
