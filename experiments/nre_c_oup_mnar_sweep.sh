#!/usr/bin/env bash
# Sequential, resumable OUP NRE-C MNAR-10% campaign. Run from any directory.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
exec conda run --no-capture-output -n gapsbi_env python scripts/run_nre_c_oup_mcar_sweep.py \
  --matrix experiments/nre_c_oup_mnar_sweep.yaml "$@"
