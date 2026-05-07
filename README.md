# GAPSBI: Gaps in data for Simulation-Based Inference
A repository to create benchmark datasets for simulation-based inference problems with missing data.

## GAPSBI-core (v1): Benchmark Design

### Goal
Build a benchmark for simulation-based inference under missing data, focusing on:

- posterior calibration (SBC, TARP)
- robustness under MCAR / MAR / MNAR
- direct inference from (x_obs, mask) (no imputation focus)

### Benchmark Problems (6 total)
#### Core (from RISE-style SBI benchmarks)

##### Ricker model
- Nonlinear time series, low-dimensional θ
- Good for debugging + calibration sanity checks
- 
##### Ornstein–Uhlenbeck (OUP)
- Simple stochastic time series
- Tests noise handling under missingness
- 
##### GLM
- Static vector observations, ~10D θ
- Breaks sequence inductive bias
- 
##### GLU
- Gaussian vector model, ~10D θ
- Controlled high-dimensional baseline

#### New additions (GAPSBI-specific)

##### Spatial SIR (2D grid)
- Spatiotemporal epidemic dynamics
- Enables:
  - spatial block masking
  - region-based MAR
  - realistic structured missingness

##### Physics / cosmology-lite task
- keep lightweight for v1
- Options:
  - toy Lyα skewer (1D spectrum)
  - Gaussian random field
  - simple lensing toy model
- Purpose: connect to real scientific inference

Missing Data Modalities (for each task)
MCAR (baseline)
Mask independent of x and θ
Random point masking
Random block masking
MAR
Mask depends on observed/context features
Examples:
time window missingness
spatial region dropout
masking based on noisy proxy or neighbors
MNAR
Mask depends on true values or θ
Examples:
threshold censoring (x < τ or x > τ)
value-dependent dropout
parameter-dependent missingness
Dataset Structure (per sample)
θ: parameters
x_full: complete data
mask: binary (1 = observed, 0 = missing)
x_obs = mask ⊙ x_full

No imputation stored.

Baselines
1. Wang et al. (mask augmentation baseline)
Input: (x with fill value, mask)
Strong, simple, robust baseline
Must be included for all tasks
2. RISE
Imputation-based + inference
Apply mainly to:
Ricker
OUP
GLM / GLU
Serves as comparison to imputation-based approaches
3. Simformer
Flexible conditional generative model
Handles missing/unstructured data
Apply to:
selected representative tasks (not necessarily all)
4. Simple transformer ratio estimator (yours)
Input: (x_obs, mask)
No imputation
Direct posterior / ratio learning
Apply to all tasks (main method of interest)
Key Design Principle
Fix simulator: p(x∣θ)
Vary only missingness: p(m∣x,θ)

→ isolates effect of missing data on inference

Positioning
Not about reconstruction
Not about imputation quality

Core question:

Do methods produce calibrated posteriors under realistic missingness?

This is a strong, clean v1 benchmark: small, controlled, and actually useful.
