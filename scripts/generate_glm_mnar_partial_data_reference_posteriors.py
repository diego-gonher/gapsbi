"""Generate exact finite-completion GLM/MNAR partial-reference artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import logsumexp

from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask
from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    find_glm_observed_map,
    glm_mnar_compatible_completions,
    glm_log_likelihood,
    glm_observed_laplace_covariance,
    load_reference_posteriors_hdf5,
    run_glm_observed_independence_mh,
    save_partial_data_reference_posteriors_hdf5,
)
from gapsbi.simulators import GLMSimulator


def log_evidence_is(x: np.ndarray, log_mask: float, simulator: GLMSimulator, seed: int, proposals: int) -> tuple[float, float]:
    """Laplace-proposal IS estimate of log p(x,m), with proposal ESS."""
    full_mask = np.ones_like(x, dtype=np.int8)
    mode, _ = find_glm_observed_map(x, full_mask, simulator)
    covariance = glm_observed_laplace_covariance(mode, x, full_mask, simulator)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mode, covariance, size=proposals)
    delta = draws - mode
    precision = np.linalg.inv(covariance)
    _, logdet = np.linalg.slogdet(covariance)
    log_q = -0.5 * (simulator.theta_dim * np.log(2 * np.pi) + logdet + np.einsum('ni,ij,nj->n', delta, precision, delta))
    log_w = simulator.log_prior(draws) + glm_log_likelihood(draws, x, simulator) + log_mask - log_q
    log_z = float(logsumexp(log_w) - np.log(proposals))
    normalized = np.exp(log_w - logsumexp(log_w))
    return log_z, float(1.0 / np.sum(normalized**2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missing-fraction', type=float, required=True)
    parser.add_argument('--proposals', type=int, default=100_000)
    parser.add_argument('--samples', type=int, default=10_000)
    parser.add_argument('--reference-mask-seed', type=int, default=987_654)
    parser.add_argument('--seed', type=int, default=45_678)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    eps_tag = f'{round(args.missing_fraction * 100):03d}'
    dataset = Path(f'data/canonical_v1/glm/mnar/glm_raw_mnar_self_censoring_mean_normalized_identity_eps{eps_tag}_seed123.h5')
    with h5py.File(dataset) as handle:
        mask_meta = json.loads(handle.attrs['mask'])
    full_obs, full_theta, full_meta, _ = load_reference_posteriors_hdf5('references/reference_posteriors_v1/glm_references.h5')
    with h5py.File('references/reference_posteriors_v1/glm_references.h5') as handle:
        full_samples = np.asarray(handle['reference_posterior/theta_samples'])
    sim_meta = full_meta['simulator']
    simulator = GLMSimulator(dim=int(sim_meta['dim']), duration=int(sim_meta['duration']), stimulus_seed=int(sim_meta['stimulus_seed']), summary='raw')
    generator = MeanNormalizedSelfCensoringMNARMask(**{key: mask_meta[key] for key in ('missing_fraction', 'eps', 'score_transform')})
    x_full = np.asarray(full_obs['x_full'], dtype=np.int8)
    theta_true = np.asarray(full_obs['theta_true'], dtype=float)
    mask = generator.generate(x_full, theta_true, np.random.default_rng(args.reference_mask_seed))
    x_obs = x_full * mask
    rng = np.random.default_rng(args.seed)
    theta_samples = np.empty((10, args.samples, simulator.theta_dim))
    component_weights = np.full((10, 2), np.nan)
    component_log_evidence = np.full((10, 2, 2), np.nan)
    component_ess = np.full((10, 2, 2), np.nan)
    completions_store = np.full((10, 2, x_full.shape[1]), -1, dtype=np.int8)
    component_count = np.zeros(10, dtype=np.int8)
    selected_alt = np.zeros(10, dtype=np.int64)
    for i in range(10):
        completions, log_mask = glm_mnar_compatible_completions(x_obs[i], mask[i], missing_fraction=args.missing_fraction, eps=float(mask_meta['eps']))
        component_count[i] = len(completions); completions_store[i, :len(completions)] = completions
        for c, (completion, lp) in enumerate(zip(completions, log_mask)):
            for run in range(2):
                component_log_evidence[i, c, run], component_ess[i, c, run] = log_evidence_is(completion, float(lp), simulator, int(rng.integers(2**32)), args.proposals)
        log_z = np.mean(component_log_evidence[i, :len(completions)], axis=1)
        weights = np.exp(log_z - logsumexp(log_z)); component_weights[i, :len(completions)] = weights
        labels = rng.choice(len(completions), size=args.samples, p=weights)
        dominant = int(np.argmax(weights))
        # Full-reference samples are exactly p(theta|completion) for the dominant completion.
        if full_samples.shape[1] != args.samples:
            raise ValueError('Dominant full-reference sample count must equal requested sample count.')
        theta_samples[i] = full_samples[i].copy()
        for c in range(len(completions)):
            n = int(np.sum(labels == c))
            if c == dominant or n == 0:
                continue
            chain = run_glm_observed_independence_mh(completions[c], np.ones_like(mask[i]), simulator=simulator, num_steps=max(30_000, n * 20), burn_in_steps=2_000, thin=2, rng=np.random.default_rng(rng.integers(2**32)))['chain']
            theta_samples[i, labels == c] = chain[rng.choice(len(chain), size=n, replace=len(chain) < n)]
            selected_alt[i] += n
        if not np.allclose(completions[dominant], x_full[i]):
            raise RuntimeError('Dominant completion does not match full reference; refusing sample reuse.')
    observations = {'theta_true': theta_true, 'x_full': x_full, 'x_obs': x_obs, 'mask': mask}
    metadata = {
        'problem': 'glm', 'mechanism': 'mnar', 'missing_fraction': args.missing_fraction,
        'mask': mask_meta, 'reference_mask_seed': args.reference_mask_seed,
        'posterior_method': 'exact_binary_completion_mixture', 'dominant_component_samples': 'reference_posteriors_v1/glm_references.h5',
        'evidence_method': 'two_independent_laplace_proposal_importance_runs', 'proposals_per_run': args.proposals,
        'posterior_seed': args.seed, 'simulator': sim_meta, 'theta_scaled': False, 'x_scaled': False,
    }
    diagnostics = {'component_count': component_count, 'component_completions': completions_store, 'component_weights': component_weights, 'component_log_evidence': component_log_evidence, 'component_importance_ess': component_ess, 'selected_alternative_samples': selected_alt}
    output = Path(f'references/{DEFAULT_PARTIAL_DATA_REFERENCE_VERSION}/glm_mnar_eps{eps_tag}_references.h5')
    save_partial_data_reference_posteriors_hdf5(output, observations=observations, theta_samples=theta_samples, metadata=metadata, diagnostics=diagnostics, reference_version=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION, overwrite=args.overwrite)
    print(output)


if __name__ == '__main__':
    main()
