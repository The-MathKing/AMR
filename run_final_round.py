"""
GEM Bio Final Round — Critical Experiments
============================================
1. Counterfactual x₁/x₀ intervention on each oracle
   - Directly measures whether oracles USE the shortcut, not just retain it
   - Toggle x₁ (spurious): Δ_x₁ = f(x₁=1) - f(x₁=0) holding all else fixed
   - Toggle x₀ (causal):  Δ_x₀ = f(x₀=1) - f(x₀=0) holding all else fixed
   - Since true SCM efficacy does NOT depend on x₁, any Δ_x₁ ≠ 0 is pure shortcut use

2. Prior-constrained covariate-shift control
   - Runs β=0.1 penalty optimization under SHARED mechanism (covariate shift)
   - Creates matched control for the significant mechanism-shift result (p=0.004)
   - Expected: CORAL should remain near-ceiling under shared mechanism + penalty

3. Seed-level paired plot for constrained CORAL vs Standard NN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.special import expit
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings, json
warnings.filterwarnings("ignore")

# ========== DATA LOADING ==========
X_bact = np.load("real_X_v2.npy")
Z_amp = np.load("Z_amp_prior.npy")
y = np.load("real_y_v2.npy")
lineage = np.load("real_lineage.npy")
X_full = np.concatenate([X_bact, Z_amp], axis=1)

idx_c0 = np.where(lineage == 0)[0]
idx_c1 = np.where(lineage == 1)[0]
X_train_B = X_full[idx_c0]
y_train_B = y[idx_c0]
X_test_B = X_full[idx_c1]

# ========== MODEL DEFINITIONS (same as run_constrained_design.py) ==========
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
def grad_reverse(x, alpha=1.0): return GradientReversal.apply(x, alpha)

class DANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.res_head = nn.Linear(16, 1)
        self.lin_head = nn.Linear(16, 1)
    def forward(self, x, alpha=1.0):
        f = self.encoder(x)
        return self.res_head(f), self.lin_head(grad_reverse(f, alpha)), f

def coral_loss(source, target):
    d = source.size(1)
    sc = source - torch.mean(source, 0, keepdim=True)
    tc = target - torch.mean(target, 0, keepdim=True)
    return torch.sum(((sc.t() @ sc)/(source.size(0)-1) - (tc.t() @ tc)/(target.size(0)-1))**2) / (4*d*d)

def train_oracle(X_l, y_l, X_u=None, mode="NN", seed=42):
    torch.manual_seed(seed)
    model = DANN(X_l.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    Xt = torch.FloatTensor(X_l); yt = torch.FloatTensor(y_l).unsqueeze(1)
    if X_u is not None: Xu = torch.FloatTensor(X_u)
    for epoch in range(250):
        p = float(epoch) / 250
        alpha = 1.0 * (2. / (1. + np.exp(-10 * p)) - 1) if mode == "DANN" else 0.0
        model.train(); optimizer.zero_grad()
        res_p, lin_p, f_s = model(Xt, alpha)
        loss = criterion(res_p, yt)
        if mode == "DANN" and X_u is not None:
            loss += criterion(lin_p, torch.zeros_like(lin_p))
            _, lin_u, _ = model(Xu, alpha)
            loss += criterion(lin_u, torch.ones_like(lin_u))
        elif mode == "CORAL" and X_u is not None:
            _, _, f_t = model(Xu, 0.0)
            loss += 1.0 * coral_loss(f_s, f_t)
        loss.backward(); optimizer.step()
    return model

# ========== TRUE EFFICACY FUNCTIONS ==========
def true_eff_mechanism(x_bact, z_amp):
    """Mechanism shift: target domain uses z₁ and z₀ differently."""
    return expit(4.0 * (x_bact[0] * z_amp[1]) - 4.0 * (x_bact[0] * z_amp[0]) - 1.0)

def true_eff_covariate(x_bact, z_amp):
    """Covariate shift: same mechanism as source (shared z₀ dependence)."""
    return expit(4.0 * (x_bact[0] * z_amp[0]) - 1.0)

# ========== CONSTRAINED OPTIMIZATION ==========
def optimize_constrained_penalty(oracle, x_target, steps=100, lr=0.1, beta=0.1,
                                  true_eff_fn=None, latent_seed=None):
    """Penalty method: max_Z f(x*,Z) - β||Z||²."""
    if latent_seed is not None:
        torch.manual_seed(latent_seed)
    z = torch.randn(1, 16, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    x_b = torch.FloatTensor(x_target).unsqueeze(0)
    for step in range(steps):
        optimizer.zero_grad()
        pred, _, _ = oracle(torch.cat([x_b, z], dim=1), 0.0)
        loss = -pred.squeeze() + beta * torch.sum(z ** 2)
        loss.backward()
        optimizer.step()
    final_z = z.detach().numpy()[0]
    final_eff = true_eff_fn(x_target, final_z) if true_eff_fn else None
    return final_z, final_eff

# ========== SETUP ==========
model_seeds = [42, 100, 2026, 777, 999, 10, 20, 30, 40, 50]
latent_seeds = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]
idx_c1_x0_1 = [i for i in idx_c1 if X_bact[i, 0] == 1.0]
BETA = 0.1

print("=" * 70)
print("FINAL ROUND: COUNTERFACTUAL INTERVENTION + CONSTRAINED COVARIATE SHIFT")
print(f"Model seeds: {len(model_seeds)}, Latent starts: {len(latent_seeds)}")
print("=" * 70)

# ========== STORAGE ==========
# Counterfactual intervention results
cf_results = {
    "x1_sensitivity": {"Standard_NN": [], "DANN": [], "CORAL": []},
    "x0_sensitivity": {"Standard_NN": [], "DANN": [], "CORAL": []},
    "x1_sensitivity_raw": {"Standard_NN": [], "DANN": [], "CORAL": []},
    "x0_sensitivity_raw": {"Standard_NN": [], "DANN": [], "CORAL": []},
}

# Constrained covariate-shift results
cov_shift_results = {
    "model_avgs": {"A": [], "B": [], "C": [], "D": []},
    "seed_diffs_C": [],  # DANN - NN
    "seed_diffs_D": [],  # CORAL - NN
}

# Also re-collect constrained mechanism-shift for the paired plot
mech_shift_results = {
    "model_avgs": {"A": [], "B": [], "C": [], "D": []},
    "seed_diffs_C": [],
    "seed_diffs_D": [],
}

for mi, ms in enumerate(model_seeds):
    print(f"\n  Model seed {mi+1}/10: {ms}")
    np.random.seed(ms); torch.manual_seed(ms)

    # Select 5 target bacteria (x0=1, from Clade 1)
    target_indices = np.random.choice(idx_c1_x0_1, size=5, replace=False)
    x_targets = [X_bact[ti].copy() for ti in target_indices]
    for xt in x_targets: assert xt[0] == 1.0

    # Generate y_covariate for Target-Informed Covariate Shift baseline
    y_covariate = y.copy().astype(float)
    for i in idx_c1:
        prob = true_eff_covariate(X_bact[i], Z_amp[i])
        y_covariate[i] = float(np.random.rand() < prob)

    # Train oracles (same as before)
    oA = train_oracle(X_full, y, mode="NN", seed=ms)        # Target-Informed (Mech)
    oA_cov = train_oracle(X_full, y_covariate, mode="NN", seed=ms) # Target-Informed (Cov)
    oB = train_oracle(X_train_B, y_train_B, mode="NN", seed=ms)  # Standard NN
    oC = train_oracle(X_train_B, y_train_B, X_test_B, mode="DANN", seed=ms)
    oD = train_oracle(X_train_B, y_train_B, X_test_B, mode="CORAL", seed=ms)
    for o in [oA, oA_cov, oB, oC, oD]: o.eval()

    oracles = {"Standard_NN": oB, "DANN": oC, "CORAL": oD}

    # ====================================================================
    # EXPERIMENT 1: COUNTERFACTUAL x₁ AND x₀ INTERVENTION
    # ====================================================================
    # Use a batch of target-domain bacteria + random Z vectors
    n_cf_samples = 200
    np.random.seed(ms + 1000)
    cf_indices = np.random.choice(idx_c1, size=min(n_cf_samples, len(idx_c1)), replace=False)

    for oracle_name, oracle in oracles.items():
        x1_deltas = []
        x0_deltas = []

        for ci in cf_indices:
            x_base = X_bact[ci].copy()
            z_rand = Z_amp[ci].copy()  # Use the paired Z

            # Build 66-dim input: [x_bact (50), z_amp (16)]
            # --- x₁ intervention: toggle x₁ while holding everything else fixed ---
            x_with_x1_0 = x_base.copy(); x_with_x1_0[1] = 0.0
            x_with_x1_1 = x_base.copy(); x_with_x1_1[1] = 1.0

            inp_x1_0 = torch.FloatTensor(np.concatenate([x_with_x1_0, z_rand])).unsqueeze(0)
            inp_x1_1 = torch.FloatTensor(np.concatenate([x_with_x1_1, z_rand])).unsqueeze(0)

            with torch.no_grad():
                pred_x1_0 = torch.sigmoid(oracle(inp_x1_0, 0.0)[0]).item()
                pred_x1_1 = torch.sigmoid(oracle(inp_x1_1, 0.0)[0]).item()

            x1_deltas.append(pred_x1_1 - pred_x1_0)

            # --- x₀ intervention: toggle x₀ while holding everything else fixed ---
            x_with_x0_0 = x_base.copy(); x_with_x0_0[0] = 0.0
            x_with_x0_1 = x_base.copy(); x_with_x0_1[0] = 1.0

            inp_x0_0 = torch.FloatTensor(np.concatenate([x_with_x0_0, z_rand])).unsqueeze(0)
            inp_x0_1 = torch.FloatTensor(np.concatenate([x_with_x0_1, z_rand])).unsqueeze(0)

            with torch.no_grad():
                pred_x0_0 = torch.sigmoid(oracle(inp_x0_0, 0.0)[0]).item()
                pred_x0_1 = torch.sigmoid(oracle(inp_x0_1, 0.0)[0]).item()

            x0_deltas.append(pred_x0_1 - pred_x0_0)

        # Store mean absolute sensitivity per seed
        cf_results["x1_sensitivity"][oracle_name].append(np.mean(np.abs(x1_deltas)))
        cf_results["x0_sensitivity"][oracle_name].append(np.mean(np.abs(x0_deltas)))
        # Store raw signed deltas for this seed
        cf_results["x1_sensitivity_raw"][oracle_name].append(np.mean(x1_deltas))
        cf_results["x0_sensitivity_raw"][oracle_name].append(np.mean(x0_deltas))

    # ====================================================================
    # EXPERIMENT 2: CONSTRAINED COVARIATE-SHIFT CONTROL (β=0.1)
    # ====================================================================
    cov_effs = {"A": [], "B": [], "C": [], "D": []}
    mech_effs = {"A": [], "B": [], "C": [], "D": []}

    for li, ls in enumerate(latent_seeds):
        for x_target in x_targets:
            # Covariate shift (shared mechanism)
            _, eff_A_cov = optimize_constrained_penalty(oA_cov, x_target, beta=BETA,
                                                          true_eff_fn=true_eff_covariate, latent_seed=ls)
            _, eff_B_cov = optimize_constrained_penalty(oB, x_target, beta=BETA,
                                                          true_eff_fn=true_eff_covariate, latent_seed=ls)
            _, eff_C_cov = optimize_constrained_penalty(oC, x_target, beta=BETA,
                                                          true_eff_fn=true_eff_covariate, latent_seed=ls)
            _, eff_D_cov = optimize_constrained_penalty(oD, x_target, beta=BETA,
                                                          true_eff_fn=true_eff_covariate, latent_seed=ls)
            cov_effs["A"].append(eff_A_cov)
            cov_effs["B"].append(eff_B_cov)
            cov_effs["C"].append(eff_C_cov)
            cov_effs["D"].append(eff_D_cov)

            # Mechanism shift (re-collect for paired plot)
            _, eff_A_mech = optimize_constrained_penalty(oA, x_target, beta=BETA,
                                                           true_eff_fn=true_eff_mechanism, latent_seed=ls)
            _, eff_B_mech = optimize_constrained_penalty(oB, x_target, beta=BETA,
                                                           true_eff_fn=true_eff_mechanism, latent_seed=ls)
            _, eff_C_mech = optimize_constrained_penalty(oC, x_target, beta=BETA,
                                                           true_eff_fn=true_eff_mechanism, latent_seed=ls)
            _, eff_D_mech = optimize_constrained_penalty(oD, x_target, beta=BETA,
                                                           true_eff_fn=true_eff_mechanism, latent_seed=ls)
            mech_effs["A"].append(eff_A_mech)
            mech_effs["B"].append(eff_B_mech)
            mech_effs["C"].append(eff_C_mech)
            mech_effs["D"].append(eff_D_mech)

    # Store seed-level means
    for k in ["A", "B", "C", "D"]:
        cov_shift_results["model_avgs"][k].append(np.mean(cov_effs[k]))
        mech_shift_results["model_avgs"][k].append(np.mean(mech_effs[k]))

    cov_shift_results["seed_diffs_C"].append(np.mean(cov_effs["C"]) - np.mean(cov_effs["B"]))
    cov_shift_results["seed_diffs_D"].append(np.mean(cov_effs["D"]) - np.mean(cov_effs["B"]))
    mech_shift_results["seed_diffs_C"].append(np.mean(mech_effs["C"]) - np.mean(mech_effs["B"]))
    mech_shift_results["seed_diffs_D"].append(np.mean(mech_effs["D"]) - np.mean(mech_effs["B"]))


# ========================================================================
# RESULTS
# ========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: COUNTERFACTUAL FEATURE SENSITIVITY")
print("=" * 70)
print("  (|Δ_x₁| = oracle sensitivity to spurious feature; should be ~0 if no shortcut use)")
print("  (|Δ_x₀| = oracle sensitivity to causal feature; should be >0 if causal signal used)")

for oracle_name in ["Standard_NN", "DANN", "CORAL"]:
    x1_sens = cf_results["x1_sensitivity"][oracle_name]
    x0_sens = cf_results["x0_sensitivity"][oracle_name]
    x1_raw = cf_results["x1_sensitivity_raw"][oracle_name]
    x0_raw = cf_results["x0_sensitivity_raw"][oracle_name]
    print(f"\n  {oracle_name}:")
    print(f"    |Δ_x₁| (spurious): {np.mean(x1_sens):.4f} ± {np.std(x1_sens):.4f}  "
          f"(signed mean: {np.mean(x1_raw):+.4f})")
    print(f"    |Δ_x₀| (causal):   {np.mean(x0_sens):.4f} ± {np.std(x0_sens):.4f}  "
          f"(signed mean: {np.mean(x0_raw):+.4f})")
    print(f"    Per-seed |Δ_x₁|: {[f'{v:.4f}' for v in x1_sens]}")
    print(f"    Per-seed |Δ_x₀|: {[f'{v:.4f}' for v in x0_sens]}")

print("\n" + "=" * 70)
print("EXPERIMENT 2: CONSTRAINED COVARIATE-SHIFT CONTROL (β=0.1)")
print("=" * 70)

print("\n  --- Covariate Shift (Shared Mechanism), Prior-Penalized ---")
for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
    vals = cov_shift_results["model_avgs"][k]
    print(f"    {name}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

diff_C_cov = np.array(cov_shift_results["seed_diffs_C"])
diff_D_cov = np.array(cov_shift_results["seed_diffs_D"])
try:
    _, p_C_cov = wilcoxon(diff_C_cov, alternative='two-sided')
except:
    p_C_cov = 1.0
try:
    _, p_D_cov = wilcoxon(diff_D_cov, alternative='two-sided')
except:
    p_D_cov = 1.0

# Holm correction
p_sorted = sorted([(p_C_cov, "DANN"), (p_D_cov, "CORAL")])
p_holm_cov = {}
for i, (p, name) in enumerate(p_sorted):
    p_holm_cov[name] = min(p * (2 - i), 1.0)

print(f"\n  --- Paired Wilcoxon (covariate shift, n=10) ---")
print(f"    DANN vs NN:  Δ={np.mean(diff_C_cov):.4f}  p_Holm={p_holm_cov['DANN']:.4f}  "
      f"signs: {sum(1 for d in diff_C_cov if d>0)}/10 positive")
print(f"    CORAL vs NN: Δ={np.mean(diff_D_cov):.4f}  p_Holm={p_holm_cov['CORAL']:.4f}  "
      f"signs: {sum(1 for d in diff_D_cov if d>0)}/10 positive")
print(f"    CORAL per-seed diffs: {[f'{d:+.4f}' for d in diff_D_cov]}")

print("\n  --- Mechanism Shift (Changed Mechanism), Prior-Penalized ---")
for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
    vals = mech_shift_results["model_avgs"][k]
    print(f"    {name}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

diff_C_mech = np.array(mech_shift_results["seed_diffs_C"])
diff_D_mech = np.array(mech_shift_results["seed_diffs_D"])
try:
    _, p_C_mech = wilcoxon(diff_C_mech, alternative='two-sided')
except:
    p_C_mech = 1.0
try:
    _, p_D_mech = wilcoxon(diff_D_mech, alternative='two-sided')
except:
    p_D_mech = 1.0

p_sorted = sorted([(p_C_mech, "DANN"), (p_D_mech, "CORAL")])
p_holm_mech = {}
for i, (p, name) in enumerate(p_sorted):
    p_holm_mech[name] = min(p * (2 - i), 1.0)

print(f"\n  --- Paired Wilcoxon (mechanism shift, n=10) ---")
print(f"    DANN vs NN:  Δ={np.mean(diff_C_mech):.4f}  p_Holm={p_holm_mech['DANN']:.4f}  "
      f"signs: {sum(1 for d in diff_C_mech if d>0)}/10 positive")
print(f"    CORAL vs NN: Δ={np.mean(diff_D_mech):.4f}  p_Holm={p_holm_mech['CORAL']:.4f}  "
      f"signs: {sum(1 for d in diff_D_mech if d>0)}/10 positive")
print(f"    CORAL per-seed diffs: {[f'{d:+.4f}' for d in diff_D_mech]}")

# ========================================================================
# FIGURE 1: COUNTERFACTUAL SENSITIVITY BAR CHART
# ========================================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

oracle_names = ["Standard NN", "DANN", "CORAL"]
oracle_keys = ["Standard_NN", "DANN", "CORAL"]
x_pos = np.arange(len(oracle_names))
width = 0.35

x1_means = [np.mean(cf_results["x1_sensitivity"][k]) for k in oracle_keys]
x1_stds  = [np.std(cf_results["x1_sensitivity"][k]) for k in oracle_keys]
x0_means = [np.mean(cf_results["x0_sensitivity"][k]) for k in oracle_keys]
x0_stds  = [np.std(cf_results["x0_sensitivity"][k]) for k in oracle_keys]

bars1 = ax.bar(x_pos - width/2, x1_means, width, yerr=x1_stds, label='$|\\Delta_{x_1}|$ (Spurious)',
               color='#E53935', alpha=0.85, capsize=4, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, x0_means, width, yerr=x0_stds, label='$|\\Delta_{x_0}|$ (Causal)',
               color='#1E88E5', alpha=0.85, capsize=4, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Oracle', fontsize=12)
ax.set_ylabel('Mean Absolute Prediction Change', fontsize=12)
ax.set_title('Counterfactual Feature Sensitivity of Design Oracles', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(oracle_names, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(x1_means), max(x0_means)) * 1.35)
ax.axhline(y=0, color='gray', linewidth=0.5)

# Add value labels
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("counterfactual_sensitivity.pdf", dpi=300, bbox_inches='tight')
print("\nSaved counterfactual_sensitivity.pdf")

# ========================================================================
# FIGURE 2: SEED-LEVEL PAIRED PLOT — Constrained CORAL vs Standard NN
# ========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel (a): Covariate shift (shared mechanism)
nn_cov = np.array(cov_shift_results["model_avgs"]["B"])
coral_cov = np.array(cov_shift_results["model_avgs"]["D"])

for i in range(10):
    axes[0].plot([0, 1], [nn_cov[i], coral_cov[i]], 'o-', color='#757575', alpha=0.5, markersize=5)
axes[0].plot([0, 1], [np.mean(nn_cov), np.mean(coral_cov)], 's-', color='#FF9800',
             markersize=10, linewidth=2.5, markeredgecolor='black', zorder=5, label='Mean')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Standard NN', 'CORAL'], fontsize=11)
axes[0].set_ylabel('True SCM Efficacy', fontsize=11)
axes[0].set_title('(a) Covariate Shift\n(Shared Mechanism, $\\beta=0.1$)', fontsize=11)
axes[0].set_xlim(-0.3, 1.3)
axes[0].legend(fontsize=9)

# Panel (b): Mechanism shift
nn_mech = np.array(mech_shift_results["model_avgs"]["B"])
coral_mech = np.array(mech_shift_results["model_avgs"]["D"])

for i in range(10):
    axes[1].plot([0, 1], [nn_mech[i], coral_mech[i]], 'o-', color='#757575', alpha=0.5, markersize=5)
axes[1].plot([0, 1], [np.mean(nn_mech), np.mean(coral_mech)], 's-', color='#FF9800',
             markersize=10, linewidth=2.5, markeredgecolor='black', zorder=5, label='Mean')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Standard NN', 'CORAL'], fontsize=11)
axes[1].set_ylabel('True SCM Efficacy', fontsize=11)
axes[1].set_title('(b) Mechanism Shift\n(Changed Mechanism, $\\beta=0.1$)', fontsize=11)
axes[1].set_xlim(-0.3, 1.3)

# Add p-value annotation
axes[1].annotate(f'$p_{{Holm}}$ = {p_holm_mech["CORAL"]:.3f}\n9/10 seeds negative',
                 xy=(0.5, max(max(nn_mech), max(coral_mech)) * 0.95),
                 ha='center', fontsize=9, fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("constrained_paired_plot.pdf", dpi=300, bbox_inches='tight')
print("Saved constrained_paired_plot.pdf")

# ========================================================================
# SAVE ALL RESULTS
# ========================================================================
output = {
    "counterfactual": {
        oracle_name: {
            "x1_sensitivity_mean": float(np.mean(cf_results["x1_sensitivity"][oracle_name])),
            "x1_sensitivity_std": float(np.std(cf_results["x1_sensitivity"][oracle_name])),
            "x1_sensitivity_per_seed": [float(v) for v in cf_results["x1_sensitivity"][oracle_name]],
            "x1_signed_mean": float(np.mean(cf_results["x1_sensitivity_raw"][oracle_name])),
            "x0_sensitivity_mean": float(np.mean(cf_results["x0_sensitivity"][oracle_name])),
            "x0_sensitivity_std": float(np.std(cf_results["x0_sensitivity"][oracle_name])),
            "x0_sensitivity_per_seed": [float(v) for v in cf_results["x0_sensitivity"][oracle_name]],
            "x0_signed_mean": float(np.mean(cf_results["x0_sensitivity_raw"][oracle_name])),
        }
        for oracle_name in ["Standard_NN", "DANN", "CORAL"]
    },
    "constrained_covariate_shift": {
        "model_avgs": {k: [float(v) for v in cov_shift_results["model_avgs"][k]] for k in ["A","B","C","D"]},
        "paired_tests": {
            "DANN_vs_NN": {"delta_mean": float(np.mean(diff_C_cov)), "p_Holm": float(p_holm_cov["DANN"]),
                           "n_positive": int(sum(1 for d in diff_C_cov if d>0))},
            "CORAL_vs_NN": {"delta_mean": float(np.mean(diff_D_cov)), "p_Holm": float(p_holm_cov["CORAL"]),
                            "n_positive": int(sum(1 for d in diff_D_cov if d>0)),
                            "per_seed_diffs": [float(d) for d in diff_D_cov]},
        },
        "summary": {k: {"mean": float(np.mean(cov_shift_results["model_avgs"][k])),
                         "std": float(np.std(cov_shift_results["model_avgs"][k]))}
                     for k in ["A","B","C","D"]}
    },
    "constrained_mechanism_shift": {
        "model_avgs": {k: [float(v) for v in mech_shift_results["model_avgs"][k]] for k in ["A","B","C","D"]},
        "paired_tests": {
            "DANN_vs_NN": {"delta_mean": float(np.mean(diff_C_mech)), "p_Holm": float(p_holm_mech["DANN"]),
                           "n_positive": int(sum(1 for d in diff_C_mech if d>0))},
            "CORAL_vs_NN": {"delta_mean": float(np.mean(diff_D_mech)), "p_Holm": float(p_holm_mech["CORAL"]),
                            "n_positive": int(sum(1 for d in diff_D_mech if d>0)),
                            "per_seed_diffs": [float(d) for d in diff_D_mech]},
        },
        "summary": {k: {"mean": float(np.mean(mech_shift_results["model_avgs"][k])),
                         "std": float(np.std(mech_shift_results["model_avgs"][k]))}
                     for k in ["A","B","C","D"]}
    },
}

with open("final_round_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved final_round_results.json")
print("\nDONE.")
