#!/usr/bin/env bash
set -u
set -o pipefail

METHOD="npe_full_data"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
  "low_sim_budget/oup_config.yaml"
  "low_sim_budget/glm_config.yaml"
  "low_sim_budget/glu_config.yaml"
  "low_sim_budget/ricker_config.yaml"
)

mkdir -p "logs/${METHOD}"

for CONFIG in "${CONFIGS[@]}"; do

    NAME=$(basename "$CONFIG" _config.yaml)

    LOG="logs/${METHOD}/${NAME}.log"

    echo "============================================================"
    echo "Running ${METHOD} :: ${NAME}"
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
