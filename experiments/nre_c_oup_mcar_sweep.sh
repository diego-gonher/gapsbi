#!/bin/bash
# Sequential, resumable OUP NRE-C campaign. Run from any directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec conda run --no-capture-output -n gapsbi_env python scripts/run_nre_c_oup_mcar_sweep.py "$@"
