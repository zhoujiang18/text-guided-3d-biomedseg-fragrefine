# Text-Guided 3D BioMedSeg FragRefine

Official code release for our accuracy-track submission to the **CVPR 2026 Foundation Models for Text-guided 3D Biomedical Image Segmentation**.

This repository packages the Docker inference code, model definition, configuration files, transfer fine-tuning utilities, challenge-format evaluation helpers, and reproducibility notes for the text-guided 3D biomedical segmentation system used in the accuracy track.

## Method Summary

![Method overview](assets/method-bg-final.png)

The submission uses a text-guided 3D adaptation of a BioMedParse/SEEM-style segmentation model. For each 3D case, the challenge input provides:

- `imgs`: a 3D biomedical image volume.
- `spacing`: physical voxel spacing used by the official evaluator.
- `text_prompts`: class IDs mapped to text prompts, plus `instance_label`.

The model encodes the text prompts, runs multi-view 2D slice inference over the 3D volume, applies per-slice non-maximum suppression, merges class masks into the original label IDs, and writes a challenge-compatible `.npz` file containing `segs`.

## Repository Structure

```text
.
├── configs/                  # Hydra model configs
├── docs/                     # Reproducibility and data format notes
├── evaluation/               # Official-like text-track evaluation scripts
├── model_weights/            # Tokenizer files and checkpoint instructions
├── scripts/                  # Fine-tuning, checkpoint, and utility scripts
├── src/                      # Model implementation
├── Dockerfile                # Challenge Docker image
├── inference.py              # Batch inference entry point
├── predict.sh                # Challenge runtime entry point
├── requirements.txt          # Runtime Python dependencies
└── build_and_test.sh         # Example Docker build/run helper
```

## Reproducibility Status

This repository is organized against the MICCAI-style machine-learning code checklist:

- Environment and Docker runtime are specified.
- Public dataset source and expected folder structure are documented.
- Preprocessing, inference, postprocessing, evaluation, training, and checkpoint conversion commands are included.
- The large fine-tuned runtime checkpoint is distributed externally and verified by SHA256.
- Known limitations are documented explicitly.

See [docs/REPRODUCIBILITY_CHECKLIST.md](docs/REPRODUCIBILITY_CHECKLIST.md) for the itemized checklist mapping.

## Environment and Requirements

Recommended environment:

- Ubuntu 22.04
- NVIDIA GPU with at least 16 GB memory for inference; 24 GB was used for validation
- CUDA 11.8
- Python 3.10 in Docker
- PyTorch 2.3.1 + cu118

Docker is the recommended execution path because it matches the challenge runtime.

For a local Python environment:

```bash
conda create -n fragrefine python=3.10 -y
conda activate fragrefine
pip install --upgrade pip setuptools==65.6.3 packaging==23.0
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git
```

## Model Weights

The runtime checkpoint is large and is not tracked by Git:

```text
model_weights/medai_finetuned_decoder_state_dict.ckpt
```

This is our checkpoint after parameter-efficient transfer fine-tuning from the official BiomedParse v2 checkpoint. For Docker-size reasons, the released runtime checkpoint is stored as a raw `state_dict` only, without training metadata. The model loader supports this format directly.

Place the checkpoint at the path above before running inference or building the final Docker image. The tokenizer files under `model_weights/local_tokenizer/` are small and are included in this repository.

Download the processed fine-tuned checkpoint:

```bash
bash scripts/download_weights.sh
```

The default runtime checkpoint link is:

```text
https://drive.google.com/file/d/1XW0-vNX3WQiyfFTYzDFI85pt3JyUPRNk/view?usp=sharing
```

The original BiomedParse v2 checkpoint is the base model from Microsoft, not our trained checkpoint:

```text
https://huggingface.co/microsoft/BiomedParse/blob/main/biomedparse_v2.ckpt
```

## Dataset

The public challenge data are hosted on Hugging Face:

```text
https://huggingface.co/datasets/junma/CVPR-BiomedSegFM
```

The dataset contains 3D biomedical segmentation cases across CT, MR, PET, Ultrasound, and Microscopy. The Hugging Face dataset card lists the dataset license as `cc-by-nc-sa-4.0`. For the text-guided accuracy track, the relevant files are:

```text
3D_train_npz_all/                       # complete training set
3D_train_npz_random_10percent_16G/      # optional coreset subset
3D_val_npz/                             # validation inputs
3D_val_gt/                              # validation ground truth
CVPR25_TextSegFMData_with_class.json    # dataset-wise text prompts for training
```

Each `.npz` case contains an image volume and metadata. Training files include `gts`; validation/test files do not include `gts` in the input folder. Validation labels are provided separately under `3D_val_gt/`.

Expected local layout for evaluation:

```text
data/
├── 3D_val_npz/
├── 3D_val_gt/
└── CVPR25_TextSegFMData_with_class.json
```

Expected local layout for transfer fine-tuning:

```text
data/
├── 3D_train_npz_all/
└── CVPR25_TextSegFMData_with_class.json
```

See [docs/DATASET.md](docs/DATASET.md) and [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for details.

## Data Format

Each input file is a compressed `.npz` with at least:

```python
imgs: np.ndarray
spacing: np.ndarray
text_prompts: dict
```

Example `text_prompts`:

```python
{
    "1": "CT imaging of the spleen within the abdomen",
    "2": "Presence of the right kidney detected in abdominal CT images",
    "instance_label": 0
}
```

The output file has the same filename and contains:

```python
segs: np.ndarray
```

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for details.

## Preprocessing

The inference code performs lightweight runtime preprocessing rather than writing a separate preprocessed dataset:

- Load each `.npz` volume from `--input_dir`.
- Select a valid slicing axis and convert the volume into a slice stack.
- Normalize/pad/crop each slice stack to the configured 512 in-plane resolution.
- Preserve original volume shape and orientation metadata for output restoration.

The core implementation is in `utils.py::process_input`.

## Inference

Run local inference:

```bash
python inference.py \
  --input_dir /path/to/input_npz \
  --output_dir /path/to/output_npz \
  --model_path model_weights/medai_finetuned_decoder_state_dict.ckpt
```

The checkpoint can also be specified with:

```bash
export BIOMEDPARSE_CKPT=/path/to/medai_finetuned_decoder_state_dict.ckpt
```

Challenge-style execution:

```bash
sh predict.sh
```

The script expects:

```text
/workspace/inputs/
/workspace/outputs/
```

## Docker

Build the Docker image:

```bash
docker build -t text-guided-3d-biomedseg-fragrefine:latest .
```

Run on local input/output folders:

```bash
docker run --rm --gpus "device=0" -m 32G \
  -v /path/to/inputs:/workspace/inputs \
  -v /path/to/outputs:/workspace/outputs \
  text-guided-3d-biomedseg-fragrefine:latest \
  /bin/bash -c "sh predict.sh"
```

Save for challenge submission:

```bash
docker save text-guided-3d-biomedseg-fragrefine:latest | gzip > text-guided-3d-biomedseg-fragrefine.tar.gz
```

For a quick local smoke test, place the runtime checkpoint under `model_weights/` and run:

```bash
bash build_and_test.sh
```

## Evaluation

The repository includes official-like evaluation scripts in `evaluation/`. To evaluate a Docker image on a validation set:

```bash
conda create -n cvpr_segfm_eval python=3.11 -y
conda activate cvpr_segfm_eval
pip install -r evaluation/requirements_eval.txt

python evaluation/CVPR25_text_eval.py \
  --docker_folder_path /path/to/team_docker \
  --test_img_path /path/to/3D_val_npz \
  --validation_gts_path /path/to/3D_val_gt \
  --save_path /path/to/eval_outputs
```

The official-like metrics include semantic Dice, semantic normalized surface Dice, instance F1_50, and instance DSC_TP.

## Postprocessing

Postprocessing is implemented in `inference.py` and `utils.py`:

- Apply sigmoid mask scoring.
- Filter predictions using object-existence logits.
- Apply slice-wise non-maximum suppression for multiple text prompts.
- Merge prompt-wise binary masks back into challenge label IDs.
- Restore predictions to the original 3D volume shape and save lossless `.npz` outputs.

## Validation Results

On the local official-like validation run from 2026-05-16:

| Method | Cases | Semantic DSC | Semantic NSD | Instance F1_50 | Instance DSC-TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiomedParse v2 official baseline | - | 0.8000 | 0.8829 | 0.4609 | 0.6882 |
| Ours | 2144 | 0.8272 | 0.8925 | 0.4904 | 0.7152 |

Our values are from `judge_full_accuracy_20260516/medai_summary_fixed.json`, produced by our local official-like validation harness. The BiomedParse v2 baseline is the official all-data baseline reported by the challenge organizers. The exact challenge leaderboard score may differ from local validation depending on the hidden test set and evaluator version.

## Training

The released runtime checkpoint was produced by transfer fine-tuning from the official BiomedParse v2 checkpoint on the challenge training data. We do not claim to pretrain BiomedParse v2 from scratch. The repository includes the lightweight fine-tuning and checkpoint-processing scripts needed to reproduce our transfer-refinement path:

- Base checkpoint: official Microsoft `biomedparse_v2.ckpt`.
- Fine-tuning script: `scripts/finetune_cvpr_npz_slices.py`.
- Checkpoint stripping script: `scripts/prepare_state_dict_checkpoint.py`.
- Runtime checkpoint: `medai_finetuned_decoder_state_dict.ckpt`.
- Runtime input resolution: padded to 512 in-plane size.
- Inference slice batch size: 4.
- Post-processing: sigmoid thresholding with object-existence filtering and slice-wise NMS.

See `docs/TRAINING_AND_CHECKPOINTS.md` for the fine-tuning command, selected checkpoint, state-dict-only conversion, and validation metrics.

## Limitations

- The released training script reproduces our lightweight transfer fine-tuning path, not the full original BiomedParse v2 pretraining procedure.
- Hidden-test leaderboard scores may differ from local official-like validation because hidden data and evaluator versions are controlled by the challenge organizers.
- The runtime checkpoint is hosted externally because it is too large for Git.
- Fragmentation-aware refinement is conservative by design; it improves stability for fragmented predictions but may not recover missing structures when the base model fails to detect them.

## Reproducibility Checklist

See [docs/REPRODUCIBILITY_CHECKLIST.md](docs/REPRODUCIBILITY_CHECKLIST.md). The repository follows the MICCAI code checklist for environment, dataset, preprocessing, training, inference, postprocessing, evaluation, results, Docker, and limitations.

## Contributing

Issues and pull requests are welcome for bug fixes, documentation improvements, and reproducibility updates. For changes that affect metrics, please include the exact command, dataset split, checkpoint, and evaluator output used to verify the change.

## License

This repository is released under the Apache-2.0 license. See [LICENSE](LICENSE).

## Acknowledgements

We thank the organizers of the CVPR 2025-2026 Foundation Models for Text-guided 3D Biomedical Image Segmentation Challenge. We also thank all the data owners for making the medical images publicly available.
