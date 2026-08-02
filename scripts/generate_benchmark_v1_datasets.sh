#!/usr/bin/env bash

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "Generating benchmark_v1 canonical datasets: 90k / 10k / 1k"
echo "Started at $(date)"
echo "Repo root: $(pwd)"
echo "============================================================"

N_TRAIN=90000
N_VAL=10000
N_TEST=1000
SEED=123
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/canonical_v1}"
GENERIC_MNAR_MASK="${GENERIC_MNAR_MASK:-self_censoring_mnar_mean_normalized}"
GENERIC_MNAR_STEM="${GENERIC_MNAR_STEM:-mnar_self_censoring_mean_normalized_identity}"

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

  # GLU: Gaussian linear uniform, bounded prior, vector observations.
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task glu --dim 10 --simulator-scale 0.1 --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/glu/mcar/glu_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task glu --dim 10 --simulator-scale 0.1 --mask coordinate_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/glu/mar/glu_mar_coordinate_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task glu --dim 10 --simulator-scale 0.1 --mask "$GENERIC_MNAR_MASK" --mnar-score-transform identity --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/glu/mnar/glu_${GENERIC_MNAR_STEM}_eps${EPS}_seed123.h5" --overwrite

  # GLM: raw Bernoulli spike train with SBIBM-style Gaussian prior.
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task glm --summary raw --dim 10 --duration 100 --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/glm/mcar/glm_raw_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task glm --summary raw --dim 10 --duration 100 --mask coordinate_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/glm/mar/glm_raw_mar_coordinate_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task glm --summary raw --dim 10 --duration 100 --mask "$GENERIC_MNAR_MASK" --mnar-score-transform identity --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/glm/mnar/glm_raw_${GENERIC_MNAR_STEM}_eps${EPS}_seed123.h5" --overwrite

  # OUP: RISE-style Ornstein-Uhlenbeck process with widened log-theta2 prior.
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task oup --mask point_mcar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/oup/mcar/oup_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task oup --mask coordinate_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/oup/mar/oup_mar_coordinate_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task oup --mask "$GENERIC_MNAR_MASK" --mnar-score-transform identity --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/oup/mnar/oup_${GENERIC_MNAR_STEM}_eps${EPS}_seed123.h5" --overwrite

  # Lotka-Volterra: timestamp-level masks over interleaved prey/predator observations.
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task lotka_volterra --mask lv_time_block_mcar --missing-fraction "$FRAC" --block-size 5 --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/lotka_volterra/mcar/lotka_volterra_time_block_mcar_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task lotka_volterra --mask lv_time_mar --mar-mode increasing --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/lotka_volterra/mar/lotka_volterra_time_mar_increasing_eps${EPS}_seed123.h5" --overwrite
  run_cmd env PYTHONPATH=src "$PYTHON_BIN" scripts/generate_dataset.py --task lotka_volterra --mask lv_log_total_mnar --missing-fraction "$FRAC" --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --seed "$SEED" --output "${OUTPUT_ROOT}/lotka_volterra/mnar/lotka_volterra_log_total_mnar_eps${EPS}_seed123.h5" --overwrite
done

echo
echo "============================================================"
echo "Finished at $(date)"
echo "============================================================"
