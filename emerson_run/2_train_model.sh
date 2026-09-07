#!/usr/bin/env bash
set -euo pipefail

set -o allexport
if [ -f .env ]; then
    source .env
fi
set +o allexport

MODEL_NAME="${1:-amortized_tm}"

# Add --repertoire_slice 0:32 for a quick shakedown run on a few repertoires.
train-model \
    --input_dir "${PROCESSED_DATA_DIR}" \
    --output_dir "${MODEL_DIR}${MODEL_NAME}" \
    --config "${CONFIG_PATH}" \
    "${@:2}"
