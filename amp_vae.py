import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# 1. Synthesize AMP-like peptides
AAs = list("ACDEFGHIKLMNPQRSTVWY")
def generate_amp():
    # AMPs are often cationic and amphipathic
    length = random.randint(15, 30)
    seq = ""
    for _ in range(length):
        if random.random() < 0.3:
            seq += random.choice("RK") # Cationic
        elif random.random() < 0.4:
            seq += random.choice("LIVWFA") # Hydrophobic
        else:
            seq += random.choice(AAs)
    return seq

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

peptides = [generate_amp() for _ in range(5000)]
MAX_LEN = 30
aa_to_idx = {aa: i+1 for i, aa in enumerate(AAs)}
aa_to_idx['<PAD>'] = 0
VOCAB_SIZE = len(aa_to_idx)

def encode_seq(seq):
    idx = [aa_to_idx[aa] for aa in seq][:MAX_LEN]
    idx += [0] * (MAX_LEN - len(idx))
    return idx

X_amp = torch.tensor([encode_seq(s) for s in peptides])

# 2. Character-level VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 32)
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc_fc_mu = nn.Linear(128 * MAX_LEN, latent_dim)
        self.enc_fc_var = nn.Linear(128 * MAX_LEN, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 128 * MAX_LEN)
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(64, VOCAB_SIZE, kernel_size=3, padding=1)
        
    def encode(self, x):
        # x: (B, MAX_LEN)
        x = self.embedding(x).transpose(1, 2) # (B, 32, MAX_LEN)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        return self.enc_fc_mu(x), self.enc_fc_var(x)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 128, MAX_LEN)
        x = F.relu(self.dec_conv1(x))
        x = self.dec_conv2(x) # (B, VOCAB_SIZE, MAX_LEN)
        return x
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

print("Training VAE...")
dataset = torch.utils.data.TensorDataset(X_amp)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(10):
    for batch in loader:
        x = batch[0]
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        
        # Loss
        BCE = criterion(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = BCE + 0.1 * KLD
        
        loss.backward()
        optimizer.step()
        
print("VAE trained.")
torch.save(model.state_dict(), "amp_vae.pt")

# Generate 3975 representations to match bacterial isolates
with torch.no_grad():
    Z_amp = []
    for i in range(3975):
        z = torch.randn(1, 16)
        Z_amp.append(z.numpy()[0])
np.save("Z_amp_prior.npy", np.array(Z_amp))
print("Saved Z_amp_prior.npy")
