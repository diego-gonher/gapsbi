#!/usr/bin/env bash
set -u
set -o pipefail

METHOD="rise"
QUEUE_NAME="rise_oup"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
  "full_sim_budget/oup/oup_rise_mcar_eps010_config.yaml"
  "full_sim_budget/oup/oup_rise_mcar_eps025_config.yaml"
  "full_sim_budget/oup/oup_rise_mcar_eps050_config.yaml"
  "full_sim_budget/oup/oup_rise_mar_eps010_config.yaml"
  "full_sim_budget/oup/oup_rise_mar_eps025_config.yaml"
  "full_sim_budget/oup/oup_rise_mar_eps050_config.yaml"
  "full_sim_budget/oup/oup_rise_mnar_eps010_config.yaml"
  "full_sim_budget/oup/oup_rise_mnar_eps025_config.yaml"
  "full_sim_budget/oup/oup_rise_mnar_eps050_config.yaml"
)

mkdir -p "logs/${METHOD}/${QUEUE_NAME}"

for CONFIG in "${CONFIGS[@]}"; do
    NAME="$(basename "$CONFIG" _config.yaml)"
    LOG="logs/${METHOD}/${QUEUE_NAME}/${NAME}.log"

    echo "============================================================"
    echo "Running ${METHOD} :: ${QUEUE_NAME} :: ${NAME}"
    echo "Started at $(date)"
    echo "Repo root: ${REPO_ROOT}"
    echo "Config: experiments/${METHOD}/${CONFIG}"
    echo "Log: ${LOG}"
    echo "============================================================"

    PYTHONPATH=src python "experiments/${METHOD}/train.py" \
        --config "experiments/${METHOD}/${CONFIG}" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "FAILED ${METHOD} :: ${QUEUE_NAME} :: ${NAME} at $(date) with exit code ${STATUS}"
        exit "$STATUS"
    fi

    echo "FINISHED ${METHOD} :: ${QUEUE_NAME} :: ${NAME} at $(date)"
done

echo "============================================================"
echo "ALL ${METHOD} :: ${QUEUE_NAME} EXPERIMENTS FINISHED at $(date)"
echo "============================================================"
