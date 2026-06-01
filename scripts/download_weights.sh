#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/model_weights"
MODE="${1:-runtime}"

download_google_drive() {
  local file_id="$1"
  local output_file="$2"
  local tmp_dir cookies first_response confirm_token

  if ! command -v wget >/dev/null 2>&1; then
    echo "Google Drive download requires wget. Install wget or set MEDAI_FINETUNED_WEIGHTS_URL to a direct URL."
    exit 1
  fi

  tmp_dir="$(mktemp -d)"
  cookies="${tmp_dir}/cookies.txt"
  first_response="${tmp_dir}/response"

  echo "Downloading Google Drive file id ${file_id} to ${output_file}"
  wget --quiet --save-cookies "${cookies}" --keep-session-cookies --no-check-certificate \
    "https://drive.google.com/uc?export=download&id=${file_id}" \
    -O "${first_response}"

  confirm_token="$(sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' "${first_response}" | head -n 1 || true)"
  if [[ -n "${confirm_token}" ]]; then
    wget --load-cookies "${cookies}" --no-check-certificate \
      "https://drive.google.com/uc?export=download&confirm=${confirm_token}&id=${file_id}" \
      -O "${output_file}"
    rm -rf "${tmp_dir}"
    return 0
  fi

  if grep -qi "<html" "${first_response}"; then
    echo "Google Drive returned an HTML page instead of the checkpoint."
    echo "Download manually from: https://drive.google.com/file/d/${file_id}/view"
    echo "Then place it at: ${output_file}"
    rm -rf "${tmp_dir}"
    exit 1
  fi

  mv "${first_response}" "${output_file}"
  rm -rf "${tmp_dir}"
}

if [[ "${MODE}" == "runtime" ]]; then
  OUT_FILE="${OUT_DIR}/medai_finetuned_decoder_state_dict.ckpt"
  GOOGLE_DRIVE_ID="${MEDAI_FINETUNED_GOOGLE_DRIVE_ID:-1XW0-vNX3WQiyfFTYzDFI85pt3JyUPRNk}"
  WEIGHTS_URL="${MEDAI_FINETUNED_WEIGHTS_URL:-}"
elif [[ "${MODE}" == "base" ]]; then
  OUT_FILE="${OUT_DIR}/biomedparse_v2.ckpt"
  GOOGLE_DRIVE_ID=""
  WEIGHTS_URL="${BIOMEDPARSE_V2_URL:-https://huggingface.co/microsoft/BiomedParse/resolve/main/biomedparse_v2.ckpt}"
else
  echo "Usage: bash scripts/download_weights.sh [runtime|base]"
  exit 1
fi

mkdir -p "${OUT_DIR}"

if [[ -f "${OUT_FILE}" ]]; then
  echo "Checkpoint already exists: ${OUT_FILE}"
  exit 0
fi

if [[ -n "${WEIGHTS_URL}" ]]; then
  echo "Downloading checkpoint to ${OUT_FILE}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${OUT_FILE}" "${WEIGHTS_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${WEIGHTS_URL}" -o "${OUT_FILE}"
  else
    echo "Neither wget nor curl is available."
    exit 1
  fi
  exit 0
fi

if [[ -n "${GOOGLE_DRIVE_ID}" ]]; then
  download_google_drive "${GOOGLE_DRIVE_ID}" "${OUT_FILE}"
  exit 0
fi

echo "No download URL is configured for mode: ${MODE}"
echo "Download the checkpoint manually and place it at:"
echo "  ${OUT_FILE}"
if [[ "${MODE}" == "runtime" ]]; then
  echo "Runtime checkpoint share link:"
  echo "  https://drive.google.com/file/d/1XW0-vNX3WQiyfFTYzDFI85pt3JyUPRNk/view?usp=sharing"
else
  echo "Base checkpoint:"
  echo "  https://huggingface.co/microsoft/BiomedParse/blob/main/biomedparse_v2.ckpt"
fi
  exit 1
