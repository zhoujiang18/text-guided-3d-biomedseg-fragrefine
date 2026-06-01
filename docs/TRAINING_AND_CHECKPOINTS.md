# Training and Checkpoint Processing

This repository does not claim to pretrain BiomedParse v2 from scratch. The base checkpoint is the official Microsoft BiomedParse v2 model:

```text
https://huggingface.co/microsoft/BiomedParse/blob/main/biomedparse_v2.ckpt
```

Our accuracy-track checkpoint is obtained by parameter-efficient transfer fine-tuning on the official challenge training data, followed by a checkpoint repackaging step for Docker-size efficiency.

The challenge dataset can be downloaded from:

```text
https://huggingface.co/datasets/junma/CVPR-BiomedSegFM
```

## Runtime Checkpoint

Expected runtime path:

```text
model_weights/medai_finetuned_decoder_state_dict.ckpt
```

This file is a raw PyTorch `state_dict`:

- No outer `{"state_dict": ...}` wrapper.
- No embedded `metadata`.
- Directly loadable by `src/seem_3D.py::load_pretrained`.

The metadata removed from the binary checkpoint is stored separately in:

```text
model_weights/medai_finetuned_decoder_state_dict.meta.json
```

## Fine-tuning Command

Download the official base checkpoint:

```bash
bash scripts/download_weights.sh base
```

Run decoder/head transfer fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_cvpr_npz_slices.py \
  --train_root /path/to/3D_train_npz_all \
  --class_map /path/to/CVPR25_TextSegFMData_with_class.json \
  --checkpoint model_weights/biomedparse_v2.ckpt \
  --output_dir outputs/finetune_decoder_lr1e-6_steps1000 \
  --trainable decoder \
  --lr 1e-6 \
  --max_steps 1000 \
  --batch_size 1 \
  --accum_steps 8 \
  --foreground_prob 0.85 \
  --num_prompts 1 \
  --save_every 50
```

The selected checkpoint in our validation sweep was:

```text
checkpoint_step000500.ckpt
```

It was selected because it gave the best safe trade-off on full local validation:

```text
Mean DSC:     0.827152
Mean F1_50:   0.476835
Mean DSC_TP:  0.717066
```

## Repackaging for Docker

Training checkpoints are saved as:

```python
{"state_dict": ..., "metadata": ...}
```

The final Docker runtime checkpoint removes the metadata wrapper:

```bash
python scripts/prepare_state_dict_checkpoint.py \
  --input outputs/finetune_decoder_lr1e-6_steps1000/checkpoint_step000500.ckpt \
  --output model_weights/medai_finetuned_decoder_state_dict.ckpt \
  --meta-output model_weights/medai_finetuned_decoder_state_dict.meta.json
```

The processed runtime checkpoint generated in this workspace has SHA256:

```text
c91aec0a3062bdd25bbd61ede9a48107cddcec8fa51993847f49f1c8518d8c58
```

The public runtime checkpoint is hosted at:

```text
https://drive.google.com/file/d/1XW0-vNX3WQiyfFTYzDFI85pt3JyUPRNk/view?usp=sharing
```

Download it with:

```bash
bash scripts/download_weights.sh
```
