# Partial-data reference artifacts

These artifacts provide the final benchmark target described in the
[canonical workflow guide](../../docs/workflows.md):
`p(theta | x_obs, mask)` for the persisted partial observation and mask.
The directory name is retained as immutable provenance; it is not the intended
long-term public-facing workflow name.

This directory holds immutable, locally generated partial-data reference
posteriors conditioned on the realized observation and mask. Full-data
artifacts under `reference_posteriors_v1/` are unchanged.

Each artifact stores:

```text
/observations/theta_true
/observations/x_full
/observations/x_obs                 # x_full * mask
/observations/mask                  # 1=observed, 0=missing
/reference_posterior/theta_samples  # samples from p(theta | x_obs, mask)
```

Artifacts currently cover GLU, GLM, OUP, and Lotka--Volterra under MCAR and
MNAR for 10%, 25%, and 50% nominal missingness. They are deliberately ignored
by Git: the complete local collection is about 126 MB and includes posterior
samples, MCMC chains/provenance, and numerical diagnostics. Recreate them with
the corresponding `scripts/generate_*_partial_data_reference_posteriors.py`
script, using the committed source and the campaign metadata under `outputs/`.

All files use the generic partial-reference HDF5 schema. Metadata records the
original full-reference provenance, realized mask seed, sampler/grid settings,
and validation diagnostics. The default `reference_mask_seed` is `987654`
where configured by the original data pipeline.

The LV/MNAR 50% artifact is intentionally sparse:

```text
lotka_volterra_mnar_eps050_successful_references.h5
reference_indices = [0, 1, 3, 4, 5, 6, 8, 9]
```

Original indices 2 and 7 have preserved failure diagnostics because the exact
importance sampler produced zero feasible weight. They are not replaced with
full-data references or fabricated draws.

For example, regenerate GLU/MCAR at 25% with:

```bash
PYTHONPATH=src python scripts/generate_glu_mcar_partial_data_reference_posteriors.py \
  --missing-fraction 0.25
```
