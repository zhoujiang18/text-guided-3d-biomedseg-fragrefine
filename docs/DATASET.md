# Dataset

The public challenge dataset is hosted at:

```text
https://huggingface.co/datasets/junma/CVPR-BiomedSegFM
```

This dataset is the CVPR 2025 BiomedSegFM dataset for 3D biomedical image segmentation. It covers five 3D biomedical modalities:

- CT
- MR
- PET
- Ultrasound
- Microscopy

The Hugging Face dataset card lists the dataset license as `cc-by-nc-sa-4.0`. Follow the dataset license and challenge terms when downloading or redistributing derived artifacts.

## Expected Files

For the text-guided accuracy track, the key files and folders are:

```text
3D_train_npz_all/
3D_train_npz_random_10percent_16G/
3D_val_npz/
3D_val_gt/
CVPR25_TextSegFMData_with_class.json
```

`3D_train_npz_all/` is the complete training set. `3D_train_npz_random_10percent_16G/` is a smaller coreset subset. `3D_val_npz/` contains validation inputs, and `3D_val_gt/` contains validation labels.

## NPZ Keys

Training `.npz` files contain:

```text
imgs
gts
spacing
```

Validation and test input `.npz` files contain:

```text
imgs
spacing
text_prompts
```

For text-guided training, `CVPR25_TextSegFMData_with_class.json` provides dataset-wise text prompts and `instance_label` metadata.

## Recommended Local Layout

```text
data/
├── 3D_train_npz_all/
├── 3D_val_npz/
├── 3D_val_gt/
└── CVPR25_TextSegFMData_with_class.json
```

The repository does not redistribute the dataset. Download it from Hugging Face and pass local paths to the training, inference, and evaluation commands.
