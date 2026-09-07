#!/usr/bin/env bash
set -euo pipefail

set -o allexport
if [ -f .env ]; then
    source .env
fi
set +o allexport

MODEL_NAME="${1:-amortized_tm}"

# Reports ROC-AUC / PR-AUC on the held-out `split == "test"` repertoires, the same
# metrics for the Emerson burden-score baseline, per-topic separation between CMV+
# and CMV- repertoires, and the top-ranked candidate sequences.
evaluate-model \
    --input_dir "${PROCESSED_DATA_DIR}" \
    --model "${MODEL_DIR}${MODEL_NAME}/model.pt" \
    --output_dir "${MODEL_DIR}${MODEL_NAME}/evaluation" \
    "${@:2}"
