# Data Format

The public dataset is available at:

```text
https://huggingface.co/datasets/junma/CVPR-BiomedSegFM
```

The accuracy-track inference code consumes one or more `.npz` files in an input directory.

## Input Keys

Required keys:

- `imgs`: 3D image array with shape `[D, H, W]`.
- `spacing`: voxel spacing array used by the evaluator.
- `text_prompts`: Python dictionary serialized inside the `.npz`.

Optional keys:

- `gts`: ground-truth mask, used only by evaluation code.

## Text Prompt Dictionary

Semantic example:

```python
{
    "1": "CT imaging of the spleen within the abdomen",
    "2": "Presence of the right kidney detected in abdominal CT images",
    "instance_label": 0
}
```

Instance example:

```python
{
    "1": "liver tumor lesion",
    "instance_label": 1
}
```

The model predicts one mask per text prompt and maps the internal prompt order back to the original integer label IDs.

## Output Keys

Each output `.npz` uses the same filename as the input and contains:

- `segs`: predicted segmentation mask with shape matching `imgs`.
