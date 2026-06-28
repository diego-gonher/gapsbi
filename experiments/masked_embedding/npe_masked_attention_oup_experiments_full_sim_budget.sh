#!/usr/bin/env bash
set -u
set -o pipefail

METHOD_DIR="masked_embedding"
METHOD_NAME="npe_masked_attention"
QUEUE_NAME="npe_masked_attention_oup"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mar_eps010_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mar_eps025_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mar_eps050_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mcar_eps010_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mcar_eps025_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mcar_eps050_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mnar_eps010_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mnar_eps025_config.yaml"
  "masked_attention/full_sim_budget/oup/oup_npe_masked_attention_mnar_eps050_config.yaml"
)

mkdir -p "logs/${METHOD_DIR}/${METHOD_NAME}/${QUEUE_NAME}"

for CONFIG in "${CONFIGS[@]}"; do
    NAME="$(basename "$CONFIG" _config.yaml)"
    LOG="logs/${METHOD_DIR}/${METHOD_NAME}/${QUEUE_NAME}/${NAME}.log"

    echo "============================================================"
    echo "Running ${METHOD_NAME} :: ${QUEUE_NAME} :: ${NAME}"
    echo "Started at $(date)"
    echo "Repo root: ${REPO_ROOT}"
    echo "Config: experiments/${METHOD_DIR}/${CONFIG}"
    echo "Log: ${LOG}"
    echo "============================================================"

    PYTHONPATH=src python "experiments/${METHOD_DIR}/train.py" \
        --config "experiments/${METHOD_DIR}/${CONFIG}" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "FAILED ${METHOD_NAME} :: ${QUEUE_NAME} :: ${NAME} at $(date) with exit code ${STATUS}"
        exit "$STATUS"
    fi

    echo "FINISHED ${METHOD_NAME} :: ${QUEUE_NAME} :: ${NAME} at $(date)"
done

echo "============================================================"
echo "ALL ${METHOD_NAME} :: ${QUEUE_NAME} EXPERIMENTS FINISHED at $(date)"
echo "============================================================"
