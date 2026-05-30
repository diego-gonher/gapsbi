#!/usr/bin/env bash

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "Generating Campaign 1 canonical datasets: 90k / 10k / 1k"
echo "Started at $(date)"
echo "Repo root: $(pwd)"
echo "============================================================"

N_TRAIN=90000
N_VAL=10000
N_TEST=1000
SEED=123

run_cmd () {
  echo
  echo "------------------------------------------------------------"
  echo "$*"
  echo "------------------------------------------------------------"
  "$@"
}

for EPS in 010 025 050; do
  case "$EPS" in
    010) FRAC="0.10" ;;
    025) FRAC="0.25" ;;
    050) FRAC="0.50" ;;
  esac

  # GLU
  run_cmd python scripts/generate_dataset.py --task glu --dim 10 --simulator-scale 0.1 --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/glu/mcar/glu_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task glu --dim 10 --simu  run_cmd python smas  run_cmd pythar --  run_cmd pyreasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL  run_cmd python scripts/generate_dataset.py --task glu --dim 1/  run_cmd python scriptste  run_cmd python scriptee  run_cmd python scripts/generate_dataset.py --task glu --dim 10 ---t  run_cmd pym   run_cmd at  run_cmd0.1  run_c se  run_cmd pg_  run--  run_cmd python scripts/generate_dataset.py --task glu --dim 10 --s$N  run_cmd n-  run_cmd python scripts/generate_dataset.py --task glu --dim 10 --sca  runglu/m  run_cmd python scripts/ng  run_cmd pythoEP  run_cmd python scripri  run_cmd python scrippy  run_cmd python scri_da  run.py  run_k glm --summary raw --dim 10 --prior-bound 2.0 --duration 100 --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/glm/mcar/glm_raw_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task glm --summary raw --dim 10 --prior-bound 2.0 --duration 100 --mask coordinate_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/glm/mar/glm_raw_mar_coordinate_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task glm --summary raw --dim 10 --prior-bound 2.0 --duration 100 --mask self_censoring_mnar --mnar-score-transform identity --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/glm/mnar/glm_raw_mnar_self_censoring_identity_eps${EPS}_seed123.h5" --overwrite

  # OUP
  run_cmd python scripts/generate_dataset.py --task oup --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/oup/mcar/oup_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task oup --mask coordinate_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/oup/mar/oup_mar_coordinate_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task oup --mask self_censoring_mnar --mnar-score-transform identity --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/oup/mnar/oup_mnar_self_censoring_identity_eps${EPS}_seed123.h5" --overwrite

  # Ricker
  run_cmd python scripts/generate_dataset.py --task ricker --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/ricker/mcar/ricker_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task ricker --mask coordinate_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/ricker/mar/ricker_mar_coordinate_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd python scripts/generate_dataset.py --task ricker --mask self_censoring_mnar --mnar-score-transform log1p --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "data/canonical_v1/ricker/mnar/ricker_mnar_self_censoring_log1p_eps${EPS}_seed123.h5" --overwrite
done

echo
echo "============================================================"
echo "Finished at $(date)"
echo "============================================================"
