#!/usr/bin/env bash
set -u

BUDGET="mid_sim_budget"
METHOD="npe_masked_transformer_embedding"
PROBLEMS=("oup" "glm" "glu" "lotka_volterra")
MECHANISMS=("mcar" "mar" "mnar")
EPSILONS=("010" "025" "050")

mkdir -p "logs/${METHOD}/${BUDGET}"

for PROBLEM in "${PROBLEMS[@]}"; do
  for MECHANISM in "${MECHANISMS[@]}"; do
    for EPS in "${EPSILONS[@]}"; do
      CONFIG="experiments/${METHOD}/${BUDGET}/${PROBLEM}/${PROBLEM}_${METHOD}_${MECHANISM}_eps${EPS}_config.yaml"
      LOG="logs/${METHOD}/${BUDGET}/${PROBLEM}_${MECHANISM}_eps${EPS}.log"

      echo "============================================================"
      echo "Running ${METHOD} :: ${BUDGET} :: ${PROBLEM} :: ${MECHANISM} :: eps${EPS}"
      echo "Started at $(date)"
      echo "============================================================"

      PYTHONPATH=src python experiments/${METHOD}/train.py --config "${CONFIG}" 2>&1 | tee "${LOG}"
      STATUS=${PIPESTATUS[0]}
      if [ "${STATUS}" -ne 0 ]; then
        echo "FAILED ${PROBLEM} ${MECHANISM} eps${EPS} with exit code ${STATUS}"
        exit "${STATUS}"
      fi
    done
  done
done

echo "All ${METHOD} ${BUDGET} jobs completed."
