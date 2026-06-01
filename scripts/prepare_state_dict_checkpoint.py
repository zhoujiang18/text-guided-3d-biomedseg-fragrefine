#!/usr/bin/env python3
"""Convert a training checkpoint into the state-dict-only runtime format.

The fine-tuning script stores checkpoints as:

    {"state_dict": ..., "metadata": ...}

The final challenge Docker image used a smaller runtime checkpoint containing
only the raw PyTorch state_dict. This script reproduces that packaging step and
writes any provenance information into a sidecar JSON file instead of embedding
it in the large checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def public_metadata(metadata):
    """Keep reproducible hyperparameters while dropping local absolute paths."""
    if not isinstance(metadata, dict):
        return metadata

    args = metadata.get("args") if isinstance(metadata.get("args"), dict) else {}
    public_args = {
        key: value
        for key, value in args.items()
        if key
        in {
            "max_steps",
            "batch_size",
            "accum_steps",
            "num_workers",
            "lr",
            "weight_decay",
            "image_size",
            "num_prompts",
            "foreground_prob",
            "trainable",
            "amp",
            "seed",
            "limit_volumes",
            "save_every",
            "log_every",
        }
    }
    return {
        "script": Path(str(metadata.get("script", "scripts/finetune_cvpr_npz_slices.py"))).name,
        "base_checkpoint": "official Microsoft BiomedParse v2 checkpoint",
        "training_data": "official challenge training data",
        "records": metadata.get("records"),
        "trainable_params": metadata.get("trainable_params"),
        "total_params": metadata.get("total_params"),
        "created_time": metadata.get("created_time"),
        "args": public_args,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Training checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="State-dict-only output checkpoint.")
    parser.add_argument("--meta-output", type=Path, default=None, help="Optional sidecar JSON metadata path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output checkpoint.")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {args.output}. Use --force to overwrite.")

    checkpoint = torch.load(args.input, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        source_metadata = checkpoint.get("metadata")
        source_format = "wrapped_state_dict_with_metadata"
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        source_metadata = None
        source_format = "state_dict_only"
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict dict, got {type(state_dict)!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)

    meta = {
        "runtime_checkpoint_name": args.output.name,
        "checkpoint_format": "state_dict-only",
        "metadata_inside_ckpt": False,
        "source_checkpoint": args.input.name,
        "source_format": source_format,
        "source_metadata": public_metadata(source_metadata),
        "num_state_dict_keys": len(state_dict),
        "sha256": sha256_file(args.output),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
