"""
GEM Bio Final Revision — Restructured Experiments
1. Nested 10 model seeds × 10 latent starts (proper statistical replication)
2. Probes on DESIGN-ORACLE embeddings (66-dim input, not diagnostic 50-dim)
3. Mean ± 95% CI trajectory across ALL replicates
4. Latent norm tracking during optimization
5. DANN domain accuracy + CORAL covariance distance
6. Two-sided Wilcoxon signed-rank + Holm correction
7. VAE validity/diversity metrics table
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
from sklearn.metrics import roc_auc_score, accuracy_score
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
    domain_accs = []
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
            # Track domain accuracy
            if epoch == 249:
                with torch.no_grad():
                    src_pred = (torch.sigmoid(lin_p) < 0.5).float()
                    tgt_pred = (torch.sigmoid(lin_u) > 0.5).float()
                    domain_accs.append(float((src_pred.sum() + tgt_pred.sum()) / (len(lin_p) + len(lin_u))))
        elif mode == "CORAL" and X_u is not None:
            _, _, f_t = model(Xu, 0.0)
            loss += 1.0 * coral_loss(f_s, f_t)
        loss.backward(); optimizer.step()
    return model, domain_accs

def compute_coral_distance(model, X_s, X_t):
    """Compute CORAL covariance distance after training"""
    model.eval()
    with torch.no_grad():
        _, _, f_s = model(torch.FloatTensor(X_s), 0.0)
        _, _, f_t = model(torch.FloatTensor(X_t), 0.0)
    d = f_s.size(1)
    sc = f_s - f_s.mean(0, keepdim=True)
    tc = f_t - f_t.mean(0, keepdim=True)
    cov_s = (sc.t() @ sc) / (f_s.size(0) - 1)
    cov_t = (tc.t() @ tc) / (f_t.size(0) - 1)
    return float(torch.sum((cov_s - cov_t)**2) / (4 * d * d))

def optimize_peptide_with_trajectory(oracle, x_target, steps=100, lr=0.1, true_eff_fn=None, latent_seed=None):
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

# Generate covariate-shift data
np.random.seed(42)
X_bact_cov = np.copy(X_bact)
y_cov = np.zeros(len(X_bact), dtype=int)
for i in range(len(X_bact)):
    x0 = np.random.binomial(1, 0.402)
    X_bact_cov[i, 0] = x0
    y_cov[i] = np.random.binomial(1, expit(4.0 * (x0 * Z_amp[i, 0]) - 1.0))
for i in range(len(X_bact)):
    if lineage[i] == 0:
        X_bact_cov[i, 1] = y_cov[i]
    else:
        X_bact_cov[i, 1] = np.random.binomial(1, 0.5)
X_full_cov = np.concatenate([X_bact_cov, Z_amp], axis=1)
X_train_cov = X_full_cov[idx_c0]
y_train_cov = y_cov[idx_c0]
X_test_cov = X_full_cov[idx_c1]
idx_c1_susceptible_cov = [i for i in idx_c1 if X_bact_cov[i, 0] == 1.0]

# ========== EXPERIMENT 1: NESTED 10x10 DESIGN ==========
print("=" * 60)
print("EXPERIMENT 1: Nested 10 model seeds × 10 latent starts")
print("=" * 60)

# Store per-model-seed averages for paired test
model_avgs_mech = {"A": [], "B": [], "C": [], "D": []}
model_avgs_cov = {"A": [], "B": [], "C": [], "D": []}
all_replicates_mech = {"A": [], "B": [], "C": [], "D": []}
all_replicates_cov = {"A": [], "B": [], "C": [], "D": []}

# Trajectory collection (all replicates)
all_pred_traj = {"A": [], "B": [], "C": [], "D": []}
all_true_traj = {"A": [], "B": [], "C": [], "D": []}
all_norm_traj = {"A": [], "B": [], "C": [], "D": []}

# DANN domain accuracy and CORAL covariance distance
dann_domain_accs = []
coral_cov_dists_before = []
coral_cov_dists_after = []
nn_cov_dists = []

# Design-oracle probe results
design_probe = {"NN": {"x0": [], "x1": []}, "DANN": {"x0": [], "x1": []}, "CORAL": {"x0": [], "x1": []}}

# VAE decode results (per oracle type, across model seeds)
vae_decode_results = {"A": [], "B": [], "C": [], "D": []}

for mi, ms in enumerate(model_seeds):
    print(f"\n  Model seed {mi+1}/10: {ms}")
    np.random.seed(ms); torch.manual_seed(ms)
    
    # Select target bacterium for this model seed
    target_idx = np.random.choice(idx_c1_susceptible)
    x_target = X_bact[target_idx]
    target_idx_cov = np.random.choice(idx_c1_susceptible_cov)
    x_target_cov = X_bact_cov[target_idx_cov]
    
    # Train oracles (mechanism shift)
    oA, _ = train_oracle(X_full, y, mode="NN", seed=ms)
    oB, _ = train_oracle(X_train_B, y_train_B, mode="NN", seed=ms)
    oC, dann_accs = train_oracle(X_train_B, y_train_B, X_test_B, mode="DANN", seed=ms)
    oD, _ = train_oracle(X_train_B, y_train_B, X_test_B, mode="CORAL", seed=ms)
    for o in [oA, oB, oC, oD]: o.eval()
    
    # DANN domain accuracy
    if dann_accs:
        dann_domain_accs.append(dann_accs[0])
    
    # CORAL/NN covariance distances
    coral_dist = compute_coral_distance(oD, X_train_B, X_test_B)
    nn_dist = compute_coral_distance(oB, X_train_B, X_test_B)
    coral_cov_dists_after.append(coral_dist)
    nn_cov_dists.append(nn_dist)
    
    # Design-oracle probes (on the 66-dim design oracles, NOT diagnostic 50-dim)
    idx_c1_train, idx_c1_test = train_test_split(idx_c1, test_size=0.3, random_state=ms)
    for oracle, model_name in [(oB, "NN"), (oC, "DANN"), (oD, "CORAL")]:
        with torch.no_grad():
            emb = oracle(torch.FloatTensor(X_full), 0.0)[2].numpy()
        for feat_idx, feat_name in [(0, "x0"), (1, "x1")]:
            clf = LogisticRegression(max_iter=1000)
            target_feat = X_bact[:, feat_idx]
            clf.fit(emb[idx_c1_train], target_feat[idx_c1_train])
            try:
                auc = roc_auc_score(target_feat[idx_c1_test], clf.predict_proba(emb[idx_c1_test])[:, 1])
            except:
                auc = 0.5
            design_probe[model_name][feat_name].append(auc)
    
    # Run 10 latent starts per model seed
    effs_mech = {"A": [], "B": [], "C": [], "D": []}
    effs_cov = {"A": [], "B": [], "C": [], "D": []}
    
    for li, ls in enumerate(latent_seeds):
        # Mechanism shift
        zA, pt_A, tt_A, nt_A = optimize_peptide_with_trajectory(oA, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
        zB, pt_B, tt_B, nt_B = optimize_peptide_with_trajectory(oB, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
        zC, pt_C, tt_C, nt_C = optimize_peptide_with_trajectory(oC, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
        zD, pt_D, tt_D, nt_D = optimize_peptide_with_trajectory(oD, x_target, true_eff_fn=true_eff_mechanism, latent_seed=ls)
        
        effs_mech["A"].append(true_eff_mechanism(x_target, zA))
        effs_mech["B"].append(true_eff_mechanism(x_target, zB))
        effs_mech["C"].append(true_eff_mechanism(x_target, zC))
        effs_mech["D"].append(true_eff_mechanism(x_target, zD))
        
        all_pred_traj["A"].append(pt_A); all_true_traj["A"].append(tt_A); all_norm_traj["A"].append(nt_A)
        all_pred_traj["B"].append(pt_B); all_true_traj["B"].append(tt_B); all_norm_traj["B"].append(nt_B)
        all_pred_traj["C"].append(pt_C); all_true_traj["C"].append(tt_C); all_norm_traj["C"].append(nt_C)
        all_pred_traj["D"].append(pt_D); all_true_traj["D"].append(tt_D); all_norm_traj["D"].append(nt_D)
        
        all_replicates_mech["A"].append(true_eff_mechanism(x_target, zA))
        all_replicates_mech["B"].append(true_eff_mechanism(x_target, zB))
        all_replicates_mech["C"].append(true_eff_mechanism(x_target, zC))
        all_replicates_mech["D"].append(true_eff_mechanism(x_target, zD))
        
        # Covariate shift (train separate oracles for cov shift)
        oA_c, _ = train_oracle(X_full_cov, y_cov, mode="NN", seed=ms)
        oB_c, _ = train_oracle(X_train_cov, y_train_cov, mode="NN", seed=ms)
        oC_c, _ = train_oracle(X_train_cov, y_train_cov, X_test_cov, mode="DANN", seed=ms)
        oD_c, _ = train_oracle(X_train_cov, y_train_cov, X_test_cov, mode="CORAL", seed=ms)
        for o in [oA_c, oB_c, oC_c, oD_c]: o.eval()
        
        zA_c, _, _, _ = optimize_peptide_with_trajectory(oA_c, x_target_cov, true_eff_fn=true_eff_covariate, latent_seed=ls)
        zB_c, _, _, _ = optimize_peptide_with_trajectory(oB_c, x_target_cov, true_eff_fn=true_eff_covariate, latent_seed=ls)
        zC_c, _, _, _ = optimize_peptide_with_trajectory(oC_c, x_target_cov, true_eff_fn=true_eff_covariate, latent_seed=ls)
        zD_c, _, _, _ = optimize_peptide_with_trajectory(oD_c, x_target_cov, true_eff_fn=true_eff_covariate, latent_seed=ls)
        
        effs_cov["A"].append(true_eff_covariate(x_target_cov, zA_c))
        effs_cov["B"].append(true_eff_covariate(x_target_cov, zB_c))
        effs_cov["C"].append(true_eff_covariate(x_target_cov, zC_c))
        effs_cov["D"].append(true_eff_covariate(x_target_cov, zD_c))
        
        all_replicates_cov["A"].append(true_eff_covariate(x_target_cov, zA_c))
        all_replicates_cov["B"].append(true_eff_covariate(x_target_cov, zB_c))
        all_replicates_cov["C"].append(true_eff_covariate(x_target_cov, zC_c))
        all_replicates_cov["D"].append(true_eff_covariate(x_target_cov, zD_c))
        
        # VAE decode for each optimized z (mechanism shift)
        if li == 0:  # Only first latent seed per model seed
            for name, z_opt in [("A", zA), ("B", zB), ("C", zC), ("D", zD)]:
                seq = decode_z(z_opt)
                vae_decode_results[name].append({
                    "seq": seq, "len": len(seq),
                    "charge": compute_charge(seq),
                    "hydro": compute_hydrophobicity(seq),
                    "valid": all(aa in AAs for aa in seq) and len(seq) >= 5,
                    "z_norm": float(np.linalg.norm(z_opt))
                })
    
    # Model-seed average
    for k in ["A", "B", "C", "D"]:
        model_avgs_mech[k].append(np.mean(effs_mech[k]))
        model_avgs_cov[k].append(np.mean(effs_cov[k]))

# ========== RESULTS ==========
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\n--- Design Loop (Mechanism Shift) ---")
for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
    all_vals = all_replicates_mech[k]
    model_vals = model_avgs_mech[k]
    print(f"  {name}: all={np.mean(all_vals):.3f}+/-{np.std(all_vals):.3f}  model-avg={np.mean(model_vals):.3f}+/-{np.std(model_vals):.3f}")

print("\n--- Design Loop (Covariate Shift) ---")
for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
    all_vals = all_replicates_cov[k]
    model_vals = model_avgs_cov[k]
    print(f"  {name}: all={np.mean(all_vals):.3f}+/-{np.std(all_vals):.3f}  model-avg={np.mean(model_vals):.3f}+/-{np.std(model_vals):.3f}")

# Paired TWO-SIDED Wilcoxon on MODEL-SEED averages
diff_C = np.array(model_avgs_mech["C"]) - np.array(model_avgs_mech["B"])
diff_D = np.array(model_avgs_mech["D"]) - np.array(model_avgs_mech["B"])

stat_C, p_C_raw = wilcoxon(diff_C, alternative='two-sided')
stat_D, p_D_raw = wilcoxon(diff_D, alternative='two-sided')

# Holm correction
p_sorted = sorted([(p_C_raw, "DANN"), (p_D_raw, "CORAL")])
p_holm = {}
for i, (p, name) in enumerate(p_sorted):
    p_holm[name] = min(p * (2 - i), 1.0)

print(f"\n--- Paired Wilcoxon (TWO-SIDED, model-seed level, n=10) ---")
print(f"  DANN vs Standard: delta={np.mean(diff_C):.3f} [{np.percentile(diff_C, 2.5):.3f}, {np.percentile(diff_C, 97.5):.3f}]  p_raw={p_C_raw:.6f}  p_Holm={p_holm['DANN']:.6f}")
print(f"  CORAL vs Standard: delta={np.mean(diff_D):.3f} [{np.percentile(diff_D, 2.5):.3f}, {np.percentile(diff_D, 97.5):.3f}]  p_raw={p_D_raw:.6f}  p_Holm={p_holm['CORAL']:.6f}")

# DANN domain accuracy
print(f"\n--- DANN Domain Accuracy (final epoch) ---")
print(f"  Mean: {np.mean(dann_domain_accs):.3f} +/- {np.std(dann_domain_accs):.3f}")

# CORAL covariance distance
print(f"\n--- Covariance Distance ---")
print(f"  Standard NN: {np.mean(nn_cov_dists):.6f} +/- {np.std(nn_cov_dists):.6f}")
print(f"  CORAL:       {np.mean(coral_cov_dists_after):.6f} +/- {np.std(coral_cov_dists_after):.6f}")
print(f"  Reduction:   {(1 - np.mean(coral_cov_dists_after)/np.mean(nn_cov_dists))*100:.1f}%")

# Design-oracle probes
print(f"\n--- Design-Oracle Probes (on 66-dim design oracles) ---")
for model_name in ["NN", "DANN", "CORAL"]:
    for feat_name in ["x0", "x1"]:
        vals = design_probe[model_name][feat_name]
        print(f"  {model_name} {feat_name}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

# Latent norms
print(f"\n--- Final Latent Norms ---")
for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
    final_norms = [traj[-1] for traj in all_norm_traj[k]]
    init_norms = [traj[0] for traj in all_norm_traj[k]]
    print(f"  {name}: init={np.mean(init_norms):.2f}+/-{np.std(init_norms):.2f}  final={np.mean(final_norms):.2f}+/-{np.std(final_norms):.2f}")

# VAE validity
print(f"\n--- VAE Decode Validity (10 model seeds) ---")
# Also encode 1000 random z from prior for novelty check
random_seqs = set()
for i in range(1000):
    z_rand = np.random.randn(16)
    s = decode_z(z_rand)
    if len(s) > 0:
        random_seqs.add(s)

for k, name in [("A","Target-Informed"), ("B","Standard NN"), ("C","DANN"), ("D","CORAL")]:
    entries = vae_decode_results[k]
    valid = sum(1 for e in entries if e["valid"])
    unique = len(set(e["seq"] for e in entries))
    novel = sum(1 for e in entries if e["seq"] not in random_seqs)
    charges = [e["charge"] for e in entries]
    hydros = [e["hydro"] for e in entries]
    lengths = [e["len"] for e in entries]
    z_norms = [e["z_norm"] for e in entries]
    print(f"  {name}: Valid={valid}/10, Unique={unique}/10, Novel={novel}/10, "
          f"AvgLen={np.mean(lengths):.1f}, Charge={np.mean(charges):+.1f}, "
          f"Hydro={np.mean(hydros):.2f}, z_norm={np.mean(z_norms):.2f}")


# ========== FIGURE: MEAN ± CI TRAJECTORY ==========
print("\n" + "=" * 60)
print("Generating averaged trajectory figure...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#F44336", "D": "#FF9800"}
names = {"A": "Target-Informed", "B": "Standard NN", "C": "DANN", "D": "CORAL"}

for k in ["A", "B", "C", "D"]:
    arr = np.array(all_pred_traj[k])
    mean = arr.mean(axis=0)
    ci_lo = np.percentile(arr, 2.5, axis=0)
    ci_hi = np.percentile(arr, 97.5, axis=0)
    axes[0].plot(mean, color=colors[k], label=names[k], linewidth=2)
    axes[0].fill_between(range(100), ci_lo, ci_hi, color=colors[k], alpha=0.15)
axes[0].set_xlabel("Optimization Step"); axes[0].set_ylabel("Oracle-Predicted Efficacy")
axes[0].set_title("(a) Predicted Efficacy"); axes[0].legend(fontsize=7); axes[0].set_ylim(-0.05, 1.05)

for k in ["A", "B", "C", "D"]:
    arr = np.array(all_true_traj[k])
    mean = arr.mean(axis=0)
    ci_lo = np.percentile(arr, 2.5, axis=0)
    ci_hi = np.percentile(arr, 97.5, axis=0)
    axes[1].plot(mean, color=colors[k], label=names[k], linewidth=2)
    axes[1].fill_between(range(100), ci_lo, ci_hi, color=colors[k], alpha=0.15)
axes[1].set_xlabel("Optimization Step"); axes[1].set_ylabel("True SCM Efficacy")
axes[1].set_title("(b) True Efficacy"); axes[1].legend(fontsize=7); axes[1].set_ylim(-0.05, 1.05)

for k in ["A", "B", "C", "D"]:
    arr = np.array(all_norm_traj[k])
    mean = arr.mean(axis=0)
    ci_lo = np.percentile(arr, 2.5, axis=0)
    ci_hi = np.percentile(arr, 97.5, axis=0)
    axes[2].plot(mean, color=colors[k], label=names[k], linewidth=2)
    axes[2].fill_between(range(100), ci_lo, ci_hi, color=colors[k], alpha=0.15)
axes[2].set_xlabel("Optimization Step"); axes[2].set_ylabel("$\\|Z\\|_2$")
axes[2].set_title("(c) Latent Norm"); axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("optimization_trajectory.pdf", dpi=300, bbox_inches='tight')
print("Saved optimization_trajectory.pdf")

# ========== FIGURE: PROBE BAR CHART (design oracle) ==========
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
models = ["Standard NN", "DANN", "CORAL"]
x0_means = [np.mean(design_probe[m]["x0"]) for m in ["NN", "DANN", "CORAL"]]
x0_stds = [np.std(design_probe[m]["x0"]) for m in ["NN", "DANN", "CORAL"]]
x1_means = [np.mean(design_probe[m]["x1"]) for m in ["NN", "DANN", "CORAL"]]
x1_stds = [np.std(design_probe[m]["x1"]) for m in ["NN", "DANN", "CORAL"]]
x = np.arange(len(models)); width = 0.35
ax.bar(x - width/2, x0_means, width, yerr=x0_stds, label='Causal $x_0$', color='#2196F3', capsize=5, alpha=0.85)
ax.bar(x + width/2, x1_means, width, yerr=x1_stds, label='Spurious $x_1$', color='#F44336', capsize=5, alpha=0.85)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax.set_ylabel('Probe AUC (Target Clade)')
ax.set_title('Design-Oracle Latent Probe: $x_0$ vs $x_1$ Decodability')
ax.set_xticks(x); ax.set_xticklabels(models)
ax.legend(loc='upper right'); ax.set_ylim(0.3, 1.1)
plt.tight_layout()
plt.savefig("probe_bar_chart.pdf", dpi=300, bbox_inches='tight')
print("Saved probe_bar_chart.pdf")

# ========== SAVE ALL RESULTS ==========
results_all = {
    "nested_design": {
        "n_model_seeds": 10,
        "n_latent_starts": 10,
        "mechanism_shift": {
            "model_seed_avgs": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in model_avgs_mech.items()},
            "all_replicates": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_replicates_mech.items()},
        },
        "covariate_shift": {
            "model_seed_avgs": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": [float(x) for x in v]} for k, v in model_avgs_cov.items()},
            "all_replicates": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_replicates_cov.items()},
        }
    },
    "paired_tests_model_level": {
        "test": "two-sided Wilcoxon signed-rank, Holm-corrected",
        "n": 10,
        "DANN_vs_NN": {"p_raw": float(p_C_raw), "p_Holm": float(p_holm["DANN"]),
                       "delta_mean": float(np.mean(diff_C))},
        "CORAL_vs_NN": {"p_raw": float(p_D_raw), "p_Holm": float(p_holm["CORAL"]),
                        "delta_mean": float(np.mean(diff_D))}
    },
    "alignment_metrics": {
        "dann_domain_accuracy": {"mean": float(np.mean(dann_domain_accs)), "std": float(np.std(dann_domain_accs))},
        "coral_cov_distance": {"mean": float(np.mean(coral_cov_dists_after)), "std": float(np.std(coral_cov_dists_after))},
        "nn_cov_distance": {"mean": float(np.mean(nn_cov_dists)), "std": float(np.std(nn_cov_dists))},
    },
    "design_oracle_probes": {m: {f: {"mean": float(np.mean(design_probe[m][f])),
                                      "std": float(np.std(design_probe[m][f]))}
                                 for f in ["x0", "x1"]} for m in ["NN", "DANN", "CORAL"]},
    "latent_norms": {k: {"init_mean": float(np.mean([t[0] for t in all_norm_traj[k]])),
                          "final_mean": float(np.mean([t[-1] for t in all_norm_traj[k]])),
                          "final_std": float(np.std([t[-1] for t in all_norm_traj[k]]))}
                     for k in ["A", "B", "C", "D"]}
}

with open("gem_final_results.json", "w") as f:
    json.dump(results_all, f, indent=2)
print("\nSaved gem_final_results.json")
print("\nDONE.")
