import numpy as np
from scipy.special import expit

np.random.seed(42)
X_bact = np.load("real_X_original.npy") if __import__('os').path.exists("real_X_original.npy") else np.load("real_X.npy")
lineage = np.load("real_lineage.npy")
Z_amp = np.load("Z_amp_prior.npy")
n_samples = X_bact.shape[0]

y = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    x0 = np.random.binomial(1, 0.402)
    X_bact[i, 0] = x0
    z0 = Z_amp[i, 0]
    z1 = Z_amp[i, 1]
    
    if lineage[i] == 0:
        logit = 4.0 * (x0 * z0) - 1.0
    else:
        # Revert to the strict mechanism shift (Target Shift)
        logit = 4.0 * (x0 * z1) - 4.0 * (x0 * z0) - 1.0
        
    prob = expit(logit)
    y[i] = np.random.binomial(1, prob)

for i in range(n_samples):
    if lineage[i] == 0:
        X_bact[i, 1] = y[i]
    else:
        X_bact[i, 1] = np.random.binomial(1, 0.5)

np.save("real_X_v2.npy", X_bact)
np.save("real_y_v2.npy", y)
