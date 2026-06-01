#!/usr/bin/env python3
"""Create a tiny challenge-format NPZ file for smoke testing.

The synthetic image is only intended to verify file I/O and Docker wiring. It
is not meaningful for measuring segmentation quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="demo_inputs")
    parser.add_argument("--name", default="synthetic_case.npz")
    parser.add_argument("--shape", default="16,96,96", help="D,H,W")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(x) for x in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must be D,H,W")

    rng = np.random.default_rng(2025)
    imgs = (rng.normal(80, 20, size=shape).clip(0, 255)).astype(np.uint8)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    text_prompts = {
        "1": "synthetic foreground object in a 3D biomedical image",
        "instance_label": 0,
    }

    out_path = out_dir / args.name
    np.savez_compressed(out_path, imgs=imgs, spacing=spacing, text_prompts=text_prompts)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

