set -o allexport
if [ -f .env ]; then
    source .env || true
fi
set +o allexport

MODEL_NAME="default"

train-model --input_dir "${PROJECT_DIR}processed_data" --output_dir "${MODEL_DIR}${MODEL_NAME}" --config $CONFIG_PATH
