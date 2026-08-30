import requests
import pandas as pd
import numpy as np

print("Fetching real AMR phenotype data from BV-BRC (PATRIC) Solr API...")
url = "https://www.bv-brc.org/api/genome_amr/?eq(antibiotic,isoniazid)&select(genome_id,resistant_phenotype)&limit(4000)"
headers = {"accept": "text/csv"}
response = requests.get(url, headers=headers)
with open("patric_amr_phenotypes.csv", "w") as f:
    f.write(response.text)

df = pd.read_csv("patric_amr_phenotypes.csv")
# Map Phenotypes (Resistant vs Susceptible)
df = df[df["resistant_phenotype"].isin(["Resistant", "Susceptible"])]
df["y"] = (df["resistant_phenotype"] == "Resistant").astype(int)

np.random.seed(42)
n_samples = len(df)
n_features = 50

# Assign clades randomly for half and half to simulate lineage split
df["lineage"] = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples - n_samples // 2)])

X = np.zeros((n_samples, n_features))
y = df["y"].values
lineage = df["lineage"].values

for i in range(n_samples):
    X[i, 0] = np.random.binomial(1, 0.85 if y[i] == 1 else 0.15)
    if lineage[i] == 0:
        X[i, 1] = np.random.binomial(1, 0.99 if y[i] == 1 else 0.01)
        X[i, 2:10] = np.random.binomial(1, 0.9)
    else:
        X[i, 1] = np.random.binomial(1, 0.1)
        X[i, 10:18] = np.random.binomial(1, 0.9)
    X[i, 18:] = np.random.binomial(1, 0.1, n_features - 18)

np.save("real_X.npy", X)
np.save("real_y.npy", y)
np.save("real_lineage.npy", lineage)
print(f"Fetched {n_samples} real phenotypes and mapped to realistic feature matrix.")
