#!/usr/bin/env bash
set -u
set -o pipefail

METHOD="npe_full_data"

CONFIGS=(
  "full_sim_budget/oup_config.yaml"
  "full_sim_budget/glm_config.yaml"
  "full_sim_budget/glu_config.yaml"
  "full_sim_budget/ricker_config.yaml"
)

mkdir -p ../../logs/${METHOD}

for CONFIG in "${CONFIGS[@]}"; do

    NAME=$(basename "$CONFIG" _config.yaml)

    LOG="../../logs/${METHOD}/${NAME}.log"

    echo "============================================================"
    echo "Running ${METHOD} :: ${NAME}"
    echo "Started at $(date)"
    echo "============================================================"

    python train.py \
        --config "$CONFIG" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "FAILED ${NAME} with exit code ${STATUS}"
        exit "$STATUS"
    fi

    echo "Finished ${NAME} at $(date)"

done

echo "ALL EXPERIMENTS FINISHED"
