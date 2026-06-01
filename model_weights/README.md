# Model Weights

Expected runtime checkpoint path:

```text
model_weights/medai_finetuned_decoder_state_dict.ckpt
```

This is our processed fine-tuned checkpoint for the accuracy-track runtime. It is saved as a raw `state_dict` only, without the training `metadata` wrapper, matching the final Docker packaging style.

The checkpoint is approximately 1.4 GB and is intentionally not tracked in Git.

Before running inference or building the final Docker image, download the checkpoint to this directory:

```bash
bash scripts/download_weights.sh
```

The script downloads the processed runtime checkpoint from:

```text
https://drive.google.com/file/d/1XW0-vNX3WQiyfFTYzDFI85pt3JyUPRNk/view?usp=sharing
```

To override the runtime checkpoint source, set either:

```bash
export MEDAI_FINETUNED_GOOGLE_DRIVE_ID=your_google_drive_file_id
export MEDAI_FINETUNED_WEIGHTS_URL=https://your/direct/download/url
```

The original BiomedParse v2 checkpoint from Microsoft is used only as the base checkpoint for transfer fine-tuning:

```text
https://huggingface.co/microsoft/BiomedParse/blob/main/biomedparse_v2.ckpt
```

For fine-tuning experiments, download it as:

```bash
bash scripts/download_weights.sh base
```

The processed runtime checkpoint currently has this local SHA256:

```text
c91aec0a3062bdd25bbd61ede9a48107cddcec8fa51993847f49f1c8518d8c58  medai_finetuned_decoder_state_dict.ckpt
```

The sidecar provenance file `medai_finetuned_decoder_state_dict.meta.json` is tracked by Git; the large `.ckpt` file is not.

Included tokenizer files:

```text
local_tokenizer/merges.txt
local_tokenizer/special_tokens_map.json
local_tokenizer/tokenizer.json
local_tokenizer/tokenizer_config.json
local_tokenizer/vocab.json
```
