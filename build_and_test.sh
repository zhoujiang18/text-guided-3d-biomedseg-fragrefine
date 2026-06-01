#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-text-guided-3d-biomedseg-fragrefine:latest}"
INPUT_DIR="${INPUT_DIR:-$(pwd)/demo_inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/demo_outputs}"
MEMORY="${MEMORY:-32G}"
GPU_DEVICE="${GPU_DEVICE:-0}"

if [[ ! -f model_weights/medai_finetuned_decoder_state_dict.ckpt ]]; then
  echo "Missing checkpoint: model_weights/medai_finetuned_decoder_state_dict.ckpt"
  echo "Run scripts/download_weights.sh or place the checkpoint manually before building the challenge image."
  exit 1
fi

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"
python scripts/create_synthetic_case.py --out_dir "${INPUT_DIR}"

echo "Building ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Running smoke-test inference"
docker run --rm --gpus "device=${GPU_DEVICE}" -m "${MEMORY}" \
  -v "${INPUT_DIR}":/workspace/inputs \
  -v "${OUTPUT_DIR}":/workspace/outputs \
  "${IMAGE_NAME}" /bin/bash -c "sh predict.sh"

echo "Output files:"
ls -lh "${OUTPUT_DIR}"
