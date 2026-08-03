#!/usr/bin/env bash
set -u
set -o pipefail

METHOD="npe_imputation"
IMPUTATION="mean"
BUDGET="mid_sim_budget"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PROBLEMS=("oup" "glm" "glu" "lotka_volterra")
MECHANISMS=("mcar" "mar" "mnar")
EPSILONS=("010" "025" "050")

mkdir -p "logs/${METHOD}/${BUDGET}/${IMPUTATION}_imputation"

for PROBLEM in "${PROBLEMS[@]}"; do
  for MECHANISM in "${MECHANISMS[@]}"; do
    for EPS in "${EPSILONS[@]}"; do
      CONFIG="${IMPUTATION}_imputation/${BUDGET}/${PROBLEM}/${PROBLEM}_${IMPUTATION}_${MECHANISM}_eps${EPS}_config.yaml"
      NAME="${IMPUTATION}_${PROBLEM}_${MECHANISM}_eps${EPS}"
      LOG="logs/${METHOD}/${BUDGET}/${IMPUTATION}_imputation/${NAME}.log"

      echo "============================================================"
      echo "Running ${METHOD} :: ${BUDGET} :: ${NAME}"
      echo "Started at $(date)"
      echo "============================================================"

      PYTHONPATH=src python "experiments/${METHOD}/train.py" \
          --config "experiments/${METHOD}/${CONFIG}" \
          2>&1 | tee "$LOG"

      STATUS=${PIPESTATUS[0]}
      if [ "$STATUS" -ne 0 ]; then
          echo "FAILED ${NAME} with exit code ${STATUS}"
          exit "$STATUS"
      fi

      echo "Finished ${NAME} at $(date)"
    done
  done
done

echo "ALL ${IMPUTATION} IMPUTATION EXPERIMENTS FINISHED"
