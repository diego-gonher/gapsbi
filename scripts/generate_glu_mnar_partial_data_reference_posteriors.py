"""Generate quality-gated GLU/MNAR partial-data reference posteriors by IS."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import h5py
import numpy as np
from gapsbi.masks import MeanNormalizedSelfCensoringMNARMask
from gapsbi.references import (
    DEFAULT_PARTIAL_DATA_REFERENCE_VERSION,
    default_full_reference_path,
    default_partial_reference_path,
    load_reference_posteriors_hdf5,
    partial_reference_observations,
    sample_glu_mnar_missing_importance,
    sample_glu_reference_posterior,
    save_partial_data_reference_posteriors_hdf5,
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--missing-fraction', type=float, required=True)
    parser.add_argument('--full-reference', type=Path, default=default_full_reference_path('glu'))
    parser.add_argument('--dataset', type=Path, help='Canonical MNAR dataset used only to recover mask metadata.')
    parser.add_argument('--num-proposals', '--proposals', dest='num_proposals', type=int, default=1_000_000)
    parser.add_argument('--num-reference-samples', '--samples', dest='num_reference_samples', type=int, default=10_000)
    parser.add_argument('--posterior-seed', '--seed', dest='posterior_seed', type=int, default=456789)
    parser.add_argument('--reference-mask-seed', type=int, default=987654)
    parser.add_argument('--reference-version', default=DEFAULT_PARTIAL_DATA_REFERENCE_VERSION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    full_obs, _, full_meta, _ = load_reference_posteriors_hdf5(args.full_reference)
    dataset = args.dataset or Path(
        f'data/canonical_v1/glu/mnar/glu_mnar_self_censoring_mean_normalized_identity_eps{round(args.missing_fraction * 100):03d}_seed123.h5'
    )
    with h5py.File(dataset) as handle:
        mask_meta = json.loads(handle.attrs['mask'])
    generator = MeanNormalizedSelfCensoringMNARMask(
        missing_fraction=float(mask_meta['missing_fraction']), eps=float(mask_meta['eps']),
        score_transform=str(mask_meta['score_transform']),
    )
    mask = generator.generate(full_obs['x_full'], full_obs['theta_true'], np.random.default_rng(args.reference_mask_seed))
    observations = partial_reference_observations(full_obs, mask)
    x_obs = observations['x_obs']
    rng = np.random.default_rng(args.posterior_seed)
    theta = np.empty((10, args.num_reference_samples, 10))
    ess, ess_fraction, max_weight, runtime = [], [], [], []

    for index in range(10):
        started = time.perf_counter()
        missing = np.flatnonzero(mask[index] == 0)
        if len(missing):
            latent, diagnostics = sample_glu_mnar_missing_importance(
                x_obs=x_obs[index], mask=mask[index], prior_low=np.full(10, -1.),
                prior_high=np.full(10, 1.), simulator_scale=.1,
                missing_fraction=args.missing_fraction, num_proposals=args.num_proposals,
                num_samples=args.num_reference_samples, rng=rng,
            )
            if diagnostics['ess'] < 20_000 or diagnostics['ess_fraction'] < .02 or diagnostics['max_weight'] > .001:
                raise RuntimeError(f'observation {index} failed IS quality gate: {diagnostics}')
            complete = np.repeat(x_obs[index][None], args.num_reference_samples, axis=0)
            complete[:, missing] = latent
            ess.append(diagnostics['ess']); ess_fraction.append(diagnostics['ess_fraction']); max_weight.append(diagnostics['max_weight'])
        else:
            complete = np.repeat(x_obs[index][None], args.num_reference_samples, axis=0)
            ess.append(float(args.num_proposals)); ess_fraction.append(1.); max_weight.append(1 / args.num_proposals)
        theta[index] = sample_glu_reference_posterior(
            x_full=complete, prior_low=np.full(10, -1.), prior_high=np.full(10, 1.),
            simulator_scale=.1, num_reference_samples=1, rng=rng,
        )[:, 0]
        runtime.append(time.perf_counter() - started)

    # Independent IS check on the most-missing observation, without retaining a second artifact.
    validation_index = int(np.argmax((mask == 0).sum(1)))
    if np.any(mask[validation_index] == 0):
        latent, _ = sample_glu_mnar_missing_importance(
            x_obs=x_obs[validation_index], mask=mask[validation_index], prior_low=np.full(10, -1.),
            prior_high=np.full(10, 1.), simulator_scale=.1, missing_fraction=args.missing_fraction,
            num_proposals=args.num_proposals, num_samples=args.num_reference_samples,
            rng=np.random.default_rng(args.posterior_seed + 10_000),
        )
        complete = np.repeat(x_obs[validation_index][None], args.num_reference_samples, axis=0)
        complete[:, mask[validation_index] == 0] = latent
        validation_samples = sample_glu_reference_posterior(
            x_full=complete, prior_low=np.full(10, -1.), prior_high=np.full(10, 1.),
            simulator_scale=.1, num_reference_samples=1, rng=np.random.default_rng(args.posterior_seed + 10_001),
        )[:, 0]
        validation_mean_delta = float(np.abs(validation_samples.mean(0) - theta[validation_index].mean(0)).max())
    else:
        validation_mean_delta = 0.

    metadata = {'problem':'glu','missingness_mechanism':'mnar','posterior_method':'prior_predictive_importance_sampling_then_exact_truncated_normal','posterior_conditioning':'theta_given_x_obs_mask','mask':mask_meta,'simulator':full_meta['simulator'],'prior_low':[-1.]*10,'prior_high':[1.]*10,'simulator_scale':.1,'reference_mask_seed':args.reference_mask_seed,'proposal_seed':args.posterior_seed,'num_proposals':args.num_proposals,'resampling':'numpy_choice_weighted_with_replacement','quality_gate_ess':20000,'quality_gate_ess_fraction':.02,'quality_gate_max_weight':.001,'reference_observation_source':str(args.full_reference)}
    diagnostics = {'importance_ess':np.array(ess),'importance_ess_fraction':np.array(ess_fraction),'importance_max_normalized_weight':np.array(max_weight),'importance_num_proposals':np.full(10,args.num_proposals),'runtime_sec':np.array(runtime),'independent_validation_observation_indices':np.array([validation_index]),'independent_validation_theta_mean_abs_max_delta':np.array([validation_mean_delta])}
    output = args.output or default_partial_reference_path('glu', 'mnar', args.missing_fraction, reference_version=args.reference_version)
    save_partial_data_reference_posteriors_hdf5(output, observations=observations, theta_samples=theta, metadata=metadata, diagnostics=diagnostics, reference_version=args.reference_version, overwrite=args.overwrite)
    print(json.dumps({'path':str(output),'min_ess':min(ess),'min_frac':min(ess_fraction),'max_weight':max(max_weight),'runtime':sum(runtime),'validation_delta':validation_mean_delta}))


if __name__ == '__main__':
    main()
