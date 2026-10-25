"""Generate one validated OUP/MNAR partial-data reference artifact."""
from __future__ import annotations
import argparse
from pathlib import Path
from gapsbi.references import DEFAULT_PARTIAL_DATA_REFERENCE_VERSION, generate_oup_mnar_partial_data_reference_posteriors, save_partial_data_reference_posteriors_hdf5

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--missing-fraction',type=float,required=True); p.add_argument('--full-reference',type=Path,default=Path('references/reference_posteriors_v1/oup_references.h5'))
    p.add_argument('--reference-mask-seed',type=int,default=987654); p.add_argument('--posterior-seed',type=int,default=23456); p.add_argument('--num-proposals',type=int,default=1_000_000); p.add_argument('--num-reference-samples',type=int,default=10_000); p.add_argument('--grid-resolution',type=int,default=4800); p.add_argument('--output',type=Path); p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); obs,meta,data=generate_oup_mnar_partial_data_reference_posteriors(full_reference_path=a.full_reference,missing_fraction=a.missing_fraction,reference_mask_seed=a.reference_mask_seed,posterior_seed=a.posterior_seed,num_proposals=a.num_proposals,num_reference_samples=a.num_reference_samples,grid_resolution=a.grid_resolution)
    out=a.output or Path('references')/DEFAULT_PARTIAL_DATA_REFERENCE_VERSION/f"oup_mnar_eps{round(a.missing_fraction*100):03d}_references.h5"
    save_partial_data_reference_posteriors_hdf5(out,observations=obs,theta_samples=data.pop('theta_samples'),metadata=meta,diagnostics=data,overwrite=a.overwrite)
    print(f'Saved: {out}')
if __name__=='__main__': main()
