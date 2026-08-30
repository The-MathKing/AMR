# Confounded Oracles Misguide Biomolecular Design

This repository contains the official code and data generators for the paper: **"Confounded Oracles Misguide Biomolecular Design: Population-Structure Shortcuts in Antimicrobial Candidate Optimization"**.

## Overview

Predictive models for Antimicrobial Resistance (AMR) are often plagued by population-structure confounding, where models rely on lineage-associated shortcuts rather than true causal mechanisms of resistance. 

While this failure mode is well-documented in diagnostic settings, this codebase provides a controlled, semi-synthetic stress test to demonstrate how these failures propagate downstream into **generative biomolecular design pipelines**. When a confounded predictor is used as a screening oracle to optimize novel Antimicrobial Peptides (AMPs), the optimization loop inherits and exploits the shortcut, leading to catastrophic design failure under mechanism shift.

We specifically evaluate standard neural networks alongside popular domain adaptation methods (DANN and Deep CORAL) to show that standard statistical deconfounding does not reliably produce safe biological oracles.

## Codebase Structure

* **Data Simulation:**
  * `simulate_pipeline.py` & `generate_hybrid_data.py`: Constructs the probabilistic Structural Causal Model (SCM) based on real *M. tuberculosis* resistance prevalence data.
  * `patric_amr_phenotypes.csv`: Real empirical prevalence data from BV-BRC used to parameterize the base rates of the simulation.

* **Generative Modeling:**
  * `amp_vae.py`: A character-level Variational Autoencoder (VAE) trained to compress antimicrobial peptide sequences into a continuous latent space for gradient-based optimization.
  * `amps.fasta` & `amp_vae.pt`: Sample training sequences and pre-trained VAE weights.

* **Experiments:**
  * `run_neurips_experiments.py`: Runs the primary diagnostic baselines, linear probes, and domain adaptation benchmarks (DANN, CORAL).
  * `run_gem_final.py`: Executes the full generative design loop, testing oracle reliability under Covariate Shift and Mechanism Shift regimes.
  * `run_final_round.py`: Runs counterfactual feature interventions and prior-penalized constrained optimization to isolate causal failure modes.
  * `run_constrained_design.py`: Seed-level optimization trajectories and paired comparisons.

## Getting Started

1. Ensure you have the required dependencies (PyTorch, Scikit-learn, Numpy, Pandas).
2. Generate the simulated data and train the generative components:
   ```bash
   python simulate_pipeline.py
   python generate_hybrid_data.py
   ```
3. Run the core experimental pipelines:
   ```bash
   python run_neurips_experiments.py
   python run_final_round.py
   ```

## License
MIT License
