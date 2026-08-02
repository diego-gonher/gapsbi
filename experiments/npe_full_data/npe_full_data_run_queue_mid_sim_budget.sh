#!/usr/bin/env bash
set -u
set -o pipefail

METHOD="npe_full_data"
BUDGET="mid_sim_budget"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
  "${BUDGET}/oup_config.yaml"
  "${BUDGET}/glm_config.yaml"
  "${BUDGET}/glu_config.yaml"
  "${BUDGET}/lotka_volterra_config.yaml"
)

mkdir -p "logs/${METHOD}/${BUDGET}"

for CONFIG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG" _config.yaml)
    LOG="logs/${METHOD}/${BUDGET}/${NAME}.log"

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

echo "ALL EXPERIMENTS FINISHED"
