"""
GEM Bio Final — Constrained Latent Optimization + DANN Domain Probe
====================================================================
Addresses two critical reviewer concerns:
1. Constrained optimization: keeps Z near the VAE prior (||Z|| ≈ 4)
   - Projection method: Z ← min(1, r/||Z||) · Z  with r = sqrt(16) = 4.0
   - Penalty method:   max_Z f(x*,Z) - β||Z||²     with β = 0.1
2. Independent DANN domain probe (post-hoc logistic regression, separate
   from the adversarial classifier)
3. Seed-level paired-difference data for strip charts
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

# ========== MODEL DEFINITIONS ==========
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

# ========== CONSTRAINED OPTIMIZATION ==========
def optimize_constrained_projection(oracle, x_target, steps=100, lr=0.1, radius=4.0,
                                     true_eff_fn=None, latent_seed=None):
    """Projection method: after each step, project Z back to ||Z|| <= radius."""
    if latent_seed is not None:
        torch.manual_seed(latent_seed)
    z = torch.randn(1, 16, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    x_b = torch.FloatTensor(x_target).unsqueeze(0)
    pred_traj, true_traj, norm_traj = [], [], []
    for step in range(steps):
        optimizer.zero_grad()
        pred, _, _ = oracle(torch.cat([x_b, z], dim=1), 0.0)
        loss = -pred.squeeze()
        loss.backward()
        optimizer.step()
        # Project back to trust region
        with torch.no_grad():
            z_norm = torch.norm(z)
            if z_norm > radius:
                z.data = z.data * (radius / z_norm)
        pred_traj.append(torch.sigmoid(pred).item())
        norm_traj.append(float(torch.norm(z).item()))
        if true_eff_fn is not None:
            true_traj.append(true_eff_fn(x_target, z.detach().numpy()[0]))
    return z.detach().numpy()[0], pred_traj, true_traj, norm_traj

def optimize_constrained_penalty(oracle, x_target, steps=100, lr=0.1, beta=0.1,
                                  true_eff_fn=None, latent_seed=None):
    """Penalty method: max_Z f(x*,Z) - β||Z||²."""
    if latent_seed is not None:
        torch.manual_seed(latent_seed)
    z = torch.randn(1, 16, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    x_b = torch.FloatTensor(x_target).unsqueeze(0)
    pred_traj, true_traj, norm_traj = [], [], []
    for step in range(steps):
        optimizer.zero_grad()
        pred, _, _ = oracle(torch.cat([x_b, z], dim=1), 0.0)
        # Maximize predicted efficacy minus prior penalty
        loss = -pred.squeeze() + beta * torch.sum(z ** 2)
        loss.backward()
        optimizer.step()
        pred_traj.append(torch.sigmoid(pred).item())
        norm_traj.append(float(torch.norm(z).item()))
        if true_eff_fn is not None:
            true_traj.append(true_eff_fn(x_target, z.detach().numpy()[0]))
    return z.detach().numpy()[0], pred_traj, true_traj, norm_traj

def optimize_unconstrained(oracle, x_target, steps=100, lr=0.1,
                            true_eff_fn=None, latent_seed=None):
    """Original unconstrained optimization (for comparison)."""
    if latent_seed is not None:
        torch.manual_seed(latent_seed)
    z = torch.randn(1, 16, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    x_b = torch.FloatTensor(x_target).unsqueeze(0)
    pred_traj, true_traj, norm_traj = [], [], []
    for step in range(steps):
        optimizer.zero_grad()
        pred, _, _ = oracle(torch.cat([x_b, z], dim=1), 0.0)
        loss = -pred.squeeze()
        loss.backward()
        optimizer.step()
        pred_traj.append(torch.sigmoid(pred).item())
        norm_traj.append(float(torch.norm(z).item()))
        if true_eff_fn is not None:
            true_traj.append(true_eff_fn(x_target, z.detach().numpy()[0]))
    return z.detach().numpy()[0], pred_traj, true_traj, norm_traj

def true_eff_mechanism(x_bact, z_amp):
    return expit(4.0 * (x_bact[0] * z_amp[1]) - 4.0 * (x_bact[0] * z_amp[0]) - 1.0)

def true_eff_covariate(x_bact, z_amp):
    return expit(4.0 * (x_bact[0] * z_amp[0]) - 1.0)

# ========== VAE FOR DECODE ==========
AAs = list("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN = 30
aa_to_idx = {aa: i+1 for i, aa in enumerate(AAs)}
aa_to_idx['<PAD>'] = 0
idx_to_aa = {v: k for k, v in aa_to_idx.items()}
VOCAB_SIZE = len(aa_to_idx)

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 32)
        self.enc_conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc_fc_mu = nn.Linear(128 * MAX_LEN, latent_dim)
        self.enc_fc_var = nn.Linear(128 * MAX_LEN, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, 128 * MAX_LEN)
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(64, VOCAB_SIZE, kernel_size=3, padding=1)
    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 128, MAX_LEN)
        x = F.relu(self.dec_conv1(x))
        return self.dec_conv2(x)

vae = VAE()
vae.load_state_dict(torch.load("amp_vae.pt", map_location="cpu"))
vae.eval()

def decode_z(z_np):
    z_t = torch.FloatTensor(z_np).unsqueeze(0)
    with torch.no_grad():
        logits = vae.decode(z_t)
    tokens = logits.argmax(dim=1).squeeze().numpy()
    return "".join([idx_to_aa.get(t, "") for t in tokens if t != 0])

def compute_charge(seq):
    return seq.count('R') + seq.count('K') + seq.count('H') - seq.count('D') - seq.count('E')

def compute_hydrophobicity(seq):
    kd = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
          'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
          'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    vals = [kd.get(aa, 0) for aa in seq]
    return np.mean(vals) if vals else 0

# ========== SETUP ==========
model_seeds = [42, 100, 2026, 777, 999, 10, 20, 30, 40, 50]
latent_seeds = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]
idx_c1_susceptible = [i for i in idx_c1 if X_bact[i, 0] == 1.0]

RADIUS = 4.0  # sqrt(16) — typical ||z|| for 16-dim N(0,I)
BETA = 0.1    # Prior penalty weight

print("=" * 70)
print("CONSTRAINED LATENT OPTIMIZATION EXPERIMENT")
print(f"Projection radius: {RADIUS}, Penalty beta: {BETA}")
print(f"Model seeds: {len(model_seeds)}, Latent starts: {len(latent_seeds)}")
print("=" * 70)

# Storage for all three methods
methods = ["unconstrained", "projection", "penalty"]
method_fns = {
    "unconstrained": optimize_unconstrained,
    "projection": lambda o, x, **kw: optimize_constrained_projection(o, x, radius=RADIUS, **kw),
    "penalty": lambda o, x, **kw: optimize_constrained_penalty(o, x, beta=BETA, **kw),
}

# Results storage
results = {}
for method in methods:
    results[method] = {
        "model_avgs_mech": {"A": [], "B": [], "C": [], "D": []},
        "all_pred_traj": {"A": [], "B": [], "C": [], "D": []},
        "all_true_traj": {"A": [], "B": [], "C": [], "D": []},
        "all_norm_traj": {"A": [], "B": [], "C": [], "D": []},
        "vae_decode": {"A": [], "B": [], "C": [], "D": []},
        "seed_diffs_C": [],  # DANN - Standard NN per seed
        "seed_diffs_D": [],  # CORAL - Standard NN per seed
    }

# Independent DANN domain probe storage
dann_domain_probe_aucs = []
dann_adversary_accs = []

for mi, ms in enumerate(model_seeds):
    print(f"\n  Model seed {mi+1}/10: {ms}")
    np.random.seed(ms); torch.manual_seed(ms)

    # Select target bacterium (x0=1, from Clade 1)
    target_idx = np.random.choice(idx_c1_susceptible)
    x_target = X_bact[target_idx]
    assert x_target[0] == 1.0, f"Target x0 must be 1, got {x_target[0]}"

    # Train oracles
    oA = train_oracle(X_full, y, mode="NN", seed=ms)
    oB = train_oracle(X_train_B, y_train_B, mode="NN", seed=ms)
    oC = train_oracle(X_train_B, y_train_B, X_test_B, mode="DANN", seed=ms)
    oD = train_oracle(X_train_B, y_train_B, X_test_B, mode="CORAL", seed=ms)
    for o in [oA, oB, oC, oD]: o.eval()

    # ---- INDEPENDENT DANN DOMAIN PROBE ----
    # Extract frozen embeddings from DANN encoder
    with torch.no_grad():
        _, _, emb_dann_src = oC(torch.FloatTensor(X_train_B), 0.0)
        _, _, emb_dann_tgt = oC(torch.FloatTensor(X_test_B), 0.0)
    emb_all = np.concatenate([emb_dann_src.numpy(), emb_dann_tgt.numpy()])
    domain_labels = np.concatenate([np.zeros(len(X_train_B)), np.ones(len(X_test_B))])

    # Train/test split for probe
    probe_X_train, probe_X_test, probe_y_train, probe_y_test = train_test_split(
        emb_all, domain_labels, test_size=0.3, random_state=ms, stratify=domain_labels)
    probe_clf = LogisticRegression(max_iter=1000)
    probe_clf.fit(probe_X_train, probe_y_train)
    try:
        probe_auc = roc_auc_score(probe_y_test, probe_clf.predict_proba(probe_X_test)[:, 1])
    except:
        probe_auc = 0.5
    dann_domain_probe_aucs.append(probe_auc)

    # Also get DANN's own adversary accuracy for comparison
    with torch.no_grad():
        _, lin_src, _ = oC(torch.FloatTensor(X_train_B), 0.0)
        _, lin_tgt, _ = oC(torch.FloatTensor(X_test_B), 0.0)
        src_correct = (torch.sigmoid(lin_src) < 0.5).float().mean().item()
        tgt_correct = (torch.sigmoid(lin_tgt) > 0.5).float().mean().item()
        dann_adversary_accs.append((src_correct + tgt_correct) / 2)

    # ---- RUN ALL THREE OPTIMIZATION METHODS ----
    for method in methods:
        opt_fn = method_fns[method]
        effs = {"A": [], "B": [], "C": [], "D": []}

        for li, ls in enumerate(latent_seeds):
            zA, pt_A, tt_A, nt_A = opt_fn(oA, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
            zB, pt_B, tt_B, nt_B = opt_fn(oB, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
            zC, pt_C, tt_C, nt_C = opt_fn(oC, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
            zD, pt_D, tt_D, nt_D = opt_fn(oD, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)

            effs["A"].append(true_eff_mechanism(x_target, zA))
            effs["B"].append(true_eff_mechanism(x_target, zB))
            effs["C"].append(true_eff_mechanism(x_target, zC))
            effs["D"].append(true_eff_mechanism(x_target, zD))

            results[method]["all_pred_traj"]["A"].append(pt_A)
            results[method]["all_pred_traj"]["B"].append(pt_B)
            results[method]["all_pred_traj"]["C"].append(pt_C)
            results[method]["all_pred_traj"]["D"].append(pt_D)
            results[method]["all_true_traj"]["A"].append(tt_A)
            results[method]["all_true_traj"]["B"].append(tt_B)
            results[method]["all_true_traj"]["C"].append(tt_C)
            results[method]["all_true_traj"]["D"].append(tt_D)
            results[method]["all_norm_traj"]["A"].append(nt_A)
            results[method]["all_norm_traj"]["B"].append(nt_B)
            results[method]["all_norm_traj"]["C"].append(nt_C)
            results[method]["all_norm_traj"]["D"].append(nt_D)

            # VAE decode (first latent start only per model seed)
            if li == 0:
                for name, z_opt in [("A", zA), ("B", zB), ("C", zC), ("D", zD)]:
                    seq = decode_z(z_opt)
                    results[method]["vae_decode"][name].append({
                        "seq": seq, "len": len(seq),
                        "charge": compute_charge(seq),
                        "hydro": compute_hydrophobicity(seq),
                        "valid": all(aa in AAs for aa in seq) and len(seq) >= 5,
                        "z_norm": float(np.linalg.norm(z_opt))
                    })

        # Model-seed averages
        for k in ["A", "B", "C", "D"]:
            results[method]["model_avgs_mech"][k].append(np.mean(effs[k]))

        # Seed-level diffs
        results[method]["seed_diffs_C"].append(np.mean(effs["C"]) - np.mean(effs["B"]))
        results[method]["seed_diffs_D"].append(np.mean(effs["D"]) - np.mean(effs["B"]))

# ========== RESULTS ==========
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for method in methods:
    print(f"\n{'='*60}")
    print(f"  METHOD: {method.upper()}")
    print(f"{'='*60}")

    r = results[method]
    print("\n  --- Design Loop (Mechanism Shift) ---")
    for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
        vals = r["model_avgs_mech"][k]
        print(f"    {name}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Paired tests
    diff_C = np.array(r["seed_diffs_C"])
    diff_D = np.array(r["seed_diffs_D"])

    try:
        stat_C, p_C_raw = wilcoxon(diff_C, alternative='two-sided')
    except:
        p_C_raw = 1.0
    try:
        stat_D, p_D_raw = wilcoxon(diff_D, alternative='two-sided')
    except:
        p_D_raw = 1.0

    p_sorted = sorted([(p_C_raw, "DANN"), (p_D_raw, "CORAL")])
    p_holm = {}
    for i, (p, name) in enumerate(p_sorted):
        p_holm[name] = min(p * (2 - i), 1.0)

    print(f"\n  --- Paired Wilcoxon (model-seed level, n=10) ---")
    print(f"    DANN vs NN:  Δ={np.mean(diff_C):.4f}  p_Holm={p_holm['DANN']:.4f}  "
          f"signs: {sum(1 for d in diff_C if d>0)}/10 positive")
    print(f"    CORAL vs NN: Δ={np.mean(diff_D):.4f}  p_Holm={p_holm['CORAL']:.4f}  "
          f"signs: {sum(1 for d in diff_D if d>0)}/10 positive")

    # Seed-level diffs (for strip chart)
    print(f"\n  --- Seed-Level Paired Diffs ---")
    print(f"    DANN:  {[f'{d:+.4f}' for d in diff_C]}")
    print(f"    CORAL: {[f'{d:+.4f}' for d in diff_D]}")

    # Latent norms
    print(f"\n  --- Final Latent Norms ---")
    for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
        final_norms = [t[-1] for t in r["all_norm_traj"][k]]
        print(f"    {name}: {np.mean(final_norms):.2f} ± {np.std(final_norms):.2f}")

    # VAE decode
    print(f"\n  --- VAE Decode Validity ---")
    for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
        entries = r["vae_decode"][k]
        if entries:
            valid = sum(1 for e in entries if e["valid"])
            unique = len(set(e["seq"] for e in entries))
            charges = [e["charge"] for e in entries]
            hydros = [e["hydro"] for e in entries]
            seqs = [e["seq"] for e in entries]
            unique_seqs = set(seqs)
            print(f"    {name}: Valid={valid}/10, Unique={len(unique_seqs)}/10, "
                  f"Charge={np.mean(charges):+.1f}, Hydro={np.mean(hydros):.2f}, "
                  f"z_norm={np.mean([e['z_norm'] for e in entries]):.2f}")
            # Show first 3 sequences
            for s in list(unique_seqs)[:3]:
                print(f"      seq: {s[:40]}{'...' if len(s)>40 else ''}")

# ========== INDEPENDENT DANN DOMAIN PROBE ==========
print(f"\n{'='*60}")
print("INDEPENDENT DANN DOMAIN PROBE")
print(f"{'='*60}")
print(f"  DANN adversary accuracy:   {np.mean(dann_adversary_accs):.3f} ± {np.std(dann_adversary_accs):.3f}")
print(f"  Independent probe AUC:     {np.mean(dann_domain_probe_aucs):.3f} ± {np.std(dann_domain_probe_aucs):.3f}")
print(f"  (Chance = 0.5; closer to 0.5 = more domain-invariant)")
print(f"  Per-seed probe AUCs: {[f'{a:.3f}' for a in dann_domain_probe_aucs]}")

# ========== FIGURES ==========
# 1. Comparison trajectory figure: unconstrained vs projection vs penalty
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#F44336", "D": "#FF9800"}
names = {"A": "Target-Informed", "B": "Standard NN", "C": "DANN", "D": "CORAL"}

for col, method in enumerate(["unconstrained", "projection", "penalty"]):
    r = results[method]
    for k in ["A", "B", "C", "D"]:
        arr = np.array(r["all_true_traj"][k])
        mean = arr.mean(axis=0)
        ci_lo = np.percentile(arr, 2.5, axis=0)
        ci_hi = np.percentile(arr, 97.5, axis=0)
        axes[0, col].plot(mean, color=colors[k], label=names[k], linewidth=2)
        axes[0, col].fill_between(range(100), ci_lo, ci_hi, color=colors[k], alpha=0.12)
    axes[0, col].set_title(f"{method.title()}: True Efficacy", fontsize=11)
    axes[0, col].set_xlabel("Step"); axes[0, col].set_ylabel("True SCM Efficacy")
    axes[0, col].set_ylim(-0.05, 1.05); axes[0, col].legend(fontsize=7)

    for k in ["A", "B", "C", "D"]:
        arr = np.array(r["all_norm_traj"][k])
        mean = arr.mean(axis=0)
        ci_lo = np.percentile(arr, 2.5, axis=0)
        ci_hi = np.percentile(arr, 97.5, axis=0)
        axes[1, col].plot(mean, color=colors[k], label=names[k], linewidth=2)
        axes[1, col].fill_between(range(100), ci_lo, ci_hi, color=colors[k], alpha=0.12)
    axes[1, col].set_title(f"{method.title()}: Latent Norm", fontsize=11)
    axes[1, col].set_xlabel("Step"); axes[1, col].set_ylabel("$\\|Z\\|_2$")
    axes[1, col].legend(fontsize=7)
    if method == "projection":
        axes[1, col].axhline(y=RADIUS, color='gray', linestyle='--', alpha=0.5, label=f'r={RADIUS}')

plt.tight_layout()
plt.savefig("constrained_comparison.pdf", dpi=300, bbox_inches='tight')
print("\nSaved constrained_comparison.pdf")

# 2. Seed-level paired difference strip chart (for projection method)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, method in enumerate(methods):
    r = results[method]
    diff_C = np.array(r["seed_diffs_C"])
    diff_D = np.array(r["seed_diffs_D"])

    jitter = 0.05
    x_dann = np.ones(len(diff_C)) * 0 + np.random.randn(len(diff_C)) * jitter
    x_coral = np.ones(len(diff_D)) * 1 + np.random.randn(len(diff_D)) * jitter

    axes[col].scatter(x_dann, diff_C, color='#F44336', alpha=0.7, s=50, zorder=3)
    axes[col].scatter(x_coral, diff_D, color='#FF9800', alpha=0.7, s=50, zorder=3)
    axes[col].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[col].set_xticks([0, 1])
    axes[col].set_xticklabels(["DANN − NN", "CORAL − NN"])
    axes[col].set_ylabel("Δ True Efficacy (per model seed)")
    axes[col].set_title(f"{method.title()}")

    # Add mean markers
    axes[col].scatter([0], [np.mean(diff_C)], color='#F44336', marker='D', s=100,
                       edgecolors='black', linewidth=1.5, zorder=4)
    axes[col].scatter([1], [np.mean(diff_D)], color='#FF9800', marker='D', s=100,
                       edgecolors='black', linewidth=1.5, zorder=4)

plt.tight_layout()
plt.savefig("seed_level_diffs.pdf", dpi=300, bbox_inches='tight')
print("Saved seed_level_diffs.pdf")

# ========== SAVE ALL RESULTS ==========
output = {
    "methods": {},
    "dann_domain_probe": {
        "adversary_accuracy": {"mean": float(np.mean(dann_adversary_accs)),
                               "std": float(np.std(dann_adversary_accs)),
                               "values": [float(x) for x in dann_adversary_accs]},
        "independent_probe_auc": {"mean": float(np.mean(dann_domain_probe_aucs)),
                                   "std": float(np.std(dann_domain_probe_aucs)),
                                   "values": [float(x) for x in dann_domain_probe_aucs]},
    }
}

for method in methods:
    r = results[method]
    diff_C = np.array(r["seed_diffs_C"])
    diff_D = np.array(r["seed_diffs_D"])
    try:
        _, p_C = wilcoxon(diff_C, alternative='two-sided')
    except:
        p_C = 1.0
    try:
        _, p_D = wilcoxon(diff_D, alternative='two-sided')
    except:
        p_D = 1.0

    p_sorted = sorted([(p_C, "DANN"), (p_D, "CORAL")])
    p_holm = {}
    for i, (p, name) in enumerate(p_sorted):
        p_holm[name] = min(p * (2 - i), 1.0)

    output["methods"][method] = {
        "mechanism_shift": {
            k: {"mean": float(np.mean(r["model_avgs_mech"][k])),
                "std": float(np.std(r["model_avgs_mech"][k])),
                "values": [float(x) for x in r["model_avgs_mech"][k]]}
            for k in ["A", "B", "C", "D"]
        },
        "paired_tests": {
            "DANN_vs_NN": {"delta_mean": float(np.mean(diff_C)),
                           "p_Holm": float(p_holm["DANN"]),
                           "n_positive": int(sum(1 for d in diff_C if d > 0)),
                           "values": [float(x) for x in diff_C]},
            "CORAL_vs_NN": {"delta_mean": float(np.mean(diff_D)),
                            "p_Holm": float(p_holm["CORAL"]),
                            "n_positive": int(sum(1 for d in diff_D if d > 0)),
                            "values": [float(x) for x in diff_D]},
        },
        "latent_norms": {
            k: {"final_mean": float(np.mean([t[-1] for t in r["all_norm_traj"][k]])),
                "final_std": float(np.std([t[-1] for t in r["all_norm_traj"][k]]))}
            for k in ["A", "B", "C", "D"]
        },
        "vae_decode": {
            k: [{"seq": e["seq"], "charge": e["charge"], "hydro": e["hydro"],
                 "z_norm": e["z_norm"], "valid": e["valid"]}
                for e in r["vae_decode"][k]]
            for k in ["A", "B", "C", "D"]
        }
    }

with open("constrained_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved constrained_results.json")
print("\nDONE.")
