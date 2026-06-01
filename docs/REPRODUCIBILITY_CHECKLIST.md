# Reproducibility Checklist

This document maps the repository to the MICCAI-style code checklist.

## Environment and Requirements

- OS: Ubuntu 22.04 Docker base image.
- CUDA: 11.8 runtime.
- Python: 3.10 in Docker.
- GPU: validated on NVIDIA RTX 3090 24 GB; at least 16 GB GPU memory is recommended.
- Dependencies: `requirements.txt` plus PyTorch 2.3.1 cu118 and Detectron2.

## Dataset

- Public dataset URL: `https://huggingface.co/datasets/junma/CVPR-BiomedSegFM`.
- Dataset license listed on Hugging Face: `cc-by-nc-sa-4.0`.
- The public challenge validation/test data are not redistributed in this repository.
- Dataset preparation and expected folder layout are documented in `docs/DATASET.md`.
- Input/output format is documented in `docs/DATA_FORMAT.md`.
- The official-like evaluator expects the challenge folder structure described in `evaluation/CVPR25_text_eval.py`.

## Preprocessing

Implemented in `utils.py`:

- Select a valid slicing axis for each 3D volume.
- Pad/crop in-plane size to 512.
- Convert processed volume to model input tensor.
- Preserve original volume shape for output restoration.

## Model and Architecture

Implemented in `src/`:

- `src/seem_3D.py`: text-guided 3D BioMedParse/SEEM-style segmentation model.
- `src/backbone/`: focal/ConvNeXt-style backbone blocks.
- `src/pixel_decoder/`: multi-scale feature decoding.
- `src/transformer_decoder/`: prompt-conditioned mask decoding.
- `configs/`: Hydra configuration for the submitted architecture.

## Transfer Fine-tuning

The current release includes a lightweight transfer fine-tuning script for the challenge accuracy track. We initialize from the official Microsoft BiomedParse v2 checkpoint and fine-tune decoder/head components on the official challenge training data. The released runtime checkpoint should be downloaded separately and placed under `model_weights/`.

Included:

- `scripts/finetune_cvpr_npz_slices.py`: parameter-efficient transfer fine-tuning on challenge-format 3D NPZ volumes.
- `scripts/prepare_state_dict_checkpoint.py`: convert a training checkpoint with metadata into the state-dict-only runtime checkpoint used by the Docker submission.
- `model_weights/medai_finetuned_decoder_state_dict.meta.json`: provenance and validation summary for the processed checkpoint.

The original BiomedParse v2 pretraining code and data are not part of this repository.

Training protocol:

- Base checkpoint: official Microsoft BiomedParse v2 checkpoint.
- Trainable parameters: decoder/head components, with image backbone and text encoder frozen.
- Optimizer: AdamW.
- Learning rate: `1e-6` for the selected sweep run.
- Batch size: `1`, gradient accumulation: `8`.
- Input resolution: 512 in-plane size.
- Prompt sampling: dataset-wise text prompts from `CVPR25_TextSegFMData_with_class.json`.
- Selected checkpoint: `checkpoint_step000500.ckpt`, selected by full local validation trade-off.

See `docs/TRAINING_AND_CHECKPOINTS.md` for the exact command.

## Inference

Primary entry point:

```bash
python inference.py --input_dir /path/to/inputs --output_dir /path/to/outputs
```

Challenge entry point:

```bash
sh predict.sh
```

Docker entry point:

```bash
docker run --rm --gpus "device=0" -m 32G \
  -v /path/to/inputs:/workspace/inputs \
  -v /path/to/outputs:/workspace/outputs \
  text-guided-3d-biomedseg-fragrefine:latest \
  /bin/bash -c "sh predict.sh"
```

Testing details:

- Inference processes all `.npz` files in the input directory.
- Outputs are written as `.npz` files with the same filenames and a `segs` key.
- The runtime checkpoint is loaded from `BIOMEDPARSE_CKPT` or `model_weights/medai_finetuned_decoder_state_dict.ckpt`.

## Postprocessing

Implemented in `inference.py` and `utils.py`:

- Sigmoid mask scoring.
- Object-existence filtering.
- Slice-wise non-maximum suppression.
- Multi-class mask merging.
- Conversion back to the original 3D orientation and shape.

## Evaluation

Official-like text-track evaluator:

```bash
python evaluation/CVPR25_text_eval.py \
  --docker_folder_path /path/to/team_docker \
  --test_img_path /path/to/3D_val_npz \
  --validation_gts_path /path/to/3D_val_gt \
  --save_path /path/to/eval_outputs
```

Metrics:

- Semantic cases: Dice and normalized surface Dice.
- Instance cases: F1_50 and Dice on matched true-positive instances.
- Runtime per case.

## Results

Local official-like validation run:

| Method | Cases | Semantic DSC | Semantic NSD | Instance F1_50 | Instance DSC-TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiomedParse v2 official baseline | - | 0.8000 | 0.8829 | 0.4609 | 0.6882 |
| Ours | 2144 | 0.8272 | 0.8925 | 0.4904 | 0.7152 |

The hidden-test challenge score may differ from the local validation score.

## Known Limitations

- The repository does not include the original BiomedParse v2 pretraining pipeline.
- The official challenge hidden test set is not publicly available, so hidden-test leaderboard reproduction is not possible from this repository alone.
- The large fine-tuned checkpoint is distributed through Google Drive rather than Git.
- Fine-tuning gains are small on aggregate metrics; the final stack mainly improves robust full-volume execution and fragmentation-prone instance cases.

## Checklist Audit

| Checklist item | Status | Location |
| --- | --- | --- |
| Environment and requirements | Complete | `README.md`, `Dockerfile`, `requirements.txt` |
| Dataset source and structure | Complete | `README.md`, `docs/DATASET.md` |
| Data format | Complete | `docs/DATA_FORMAT.md` |
| Preprocessing | Complete | `README.md`, `utils.py` |
| Model architecture | Complete | `src/`, `configs/` |
| Training / fine-tuning command | Complete | `docs/TRAINING_AND_CHECKPOINTS.md`, `scripts/finetune_cvpr_npz_slices.py` |
| Trained model download | Complete | `model_weights/README.md`, `scripts/download_weights.sh` |
| Inference command | Complete | `README.md`, `inference.py`, `predict.sh` |
| Docker execution | Complete | `Dockerfile`, `build_and_test.sh` |
| Evaluation command | Complete | `README.md`, `evaluation/` |
| Quantitative results | Complete | `README.md`, `docs/RESULTS.md` |
| Limitations | Complete | `README.md`, this file |
| Hidden-test reproduction | Not possible from public data alone | Hidden test data are controlled by challenge organizers |
| Original BiomedParse v2 pretraining | Not included | External Microsoft checkpoint is used as initialization |
| Qualitative gallery | Not included in this code package | Presentation figures are maintained separately |
