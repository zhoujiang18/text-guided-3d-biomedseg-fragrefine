#!/usr/bin/env python3
"""Lightweight transfer fine-tuning on CVPR challenge-format 3D NPZ data.

This script initializes from the official BiomedParse v2 checkpoint and updates
only selected decoder/head parameters. It samples neighboring 2D slices from 3D
volumes and pairs them with the provided text prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VolumeRecord:
    path: str
    dataset_key: str
    modality: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_prompt(prompt_entry) -> str:
    if isinstance(prompt_entry, str):
        return prompt_entry
    if isinstance(prompt_entry, list):
        return random.choice(prompt_entry)
    raise TypeError(f"Unsupported prompt entry type: {type(prompt_entry)!r}")


def build_volume_records(train_root: Path, class_map: dict, limit_volumes: int = 0) -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    skipped_missing_prompts = 0
    for npz_path in sorted(train_root.rglob("*.npz")):
        dataset_key = npz_path.parent.name
        if dataset_key not in class_map:
            skipped_missing_prompts += 1
            continue
        modality = npz_path.parents[1].name if len(npz_path.parents) > 1 else "unknown"
        records.append(VolumeRecord(str(npz_path), dataset_key, modality))
        if limit_volumes and len(records) >= limit_volumes:
            break
    if not records:
        raise RuntimeError(f"No usable NPZ files found under {train_root}")
    print(
        f"Usable train volumes: {len(records)} "
        f"(skipped without prompt mapping: {skipped_missing_prompts})",
        flush=True,
    )
    return records


def resize_image_mask(image_2d: np.ndarray, mask_2d: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    if image_2d.shape[-2:] == (image_size, image_size) and mask_2d.shape[-2:] == (image_size, image_size):
        return image_2d, mask_2d
    channels = [
        cv2.resize(image_2d[c], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        for c in range(image_2d.shape[0])
    ]
    resized_image = np.stack(channels, axis=0)
    resized_mask = cv2.resize(mask_2d, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def augment_stack(image: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = image[:, :, ::-1]
        masks = masks[:, :, ::-1]
    if random.random() < 0.5:
        image = image[:, ::-1, :]
        masks = masks[:, ::-1, :]
    if random.random() < 0.3:
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            masks = np.rot90(masks, k, axes=(1, 2))
    if random.random() < 0.35:
        scale = np.random.uniform(0.9, 1.1)
        shift = np.random.uniform(-8.0, 8.0)
        image = np.clip(image.astype(np.float32) * scale + shift, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(masks)


class CvprNpzSliceDataset(Dataset):
    def __init__(
        self,
        records: list[VolumeRecord],
        class_map: dict,
        image_size: int = 512,
        foreground_prob: float = 0.85,
        num_prompts: int = 1,
        augment: bool = True,
    ):
        self.records = records
        self.class_map = class_map
        self.image_size = image_size
        self.foreground_prob = foreground_prob
        self.num_prompts = max(1, num_prompts)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        data = np.load(record.path, allow_pickle=True)
        image_3d = data["imgs"].astype(np.uint8, copy=False)
        gt_3d = data["gts"].astype(np.uint8, copy=False)

        prompts = self.class_map[record.dataset_key]
        is_instance = int(prompts.get("instance_label", 0))
        class_ids = sorted(int(k) for k in prompts.keys() if str(k).isdigit())
        if not class_ids:
            raise RuntimeError(f"No class ids for {record.dataset_key}")

        selected_ids = random.choices(class_ids, k=self.num_prompts)
        prompt_texts = []
        masks = []
        slice_candidates = None
        for class_id in selected_ids:
            prompt_texts.append(choose_prompt(prompts[str(class_id)]))
            target_3d = gt_3d > 0 if is_instance else gt_3d == class_id
            if slice_candidates is None:
                per_slice = target_3d.reshape(target_3d.shape[0], -1).sum(axis=1)
                if random.random() < self.foreground_prob and np.any(per_slice > 0):
                    slice_candidates = np.flatnonzero(per_slice > 0)
                else:
                    slice_candidates = np.arange(target_3d.shape[0])
            masks.append(target_3d)

        z = int(random.choice(slice_candidates))
        prev_z = max(0, z - 1)
        next_z = min(image_3d.shape[0] - 1, z + 1)
        image = np.stack([image_3d[z], image_3d[prev_z], image_3d[next_z]], axis=0)
        mask_stack = np.stack([m[z].astype(np.uint8) for m in masks], axis=0)

        resized_masks = []
        image, first_mask = resize_image_mask(image, mask_stack[0], self.image_size)
        resized_masks.append(first_mask)
        for i in range(1, mask_stack.shape[0]):
            _, resized_mask = resize_image_mask(image, mask_stack[i], self.image_size)
            resized_masks.append(resized_mask)
        mask_stack = np.stack(resized_masks, axis=0)

        if self.augment:
            image, mask_stack = augment_stack(image, mask_stack)

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "labels": torch.from_numpy(mask_stack.copy()).long(),
            "text": "[SEP]".join(prompt_texts),
        }


def build_model(checkpoint_path: Path, device: torch.device):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), job_name="cvpr_finetune", version_base=None):
        cfg = compose(config_name="seem_model_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = {k[6:] if isinstance(k, str) and k.startswith("model.") else k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model.to(device)


def set_trainable(model: torch.nn.Module, mode: str) -> int:
    for param in model.parameters():
        param.requires_grad = False

    def enable(module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True

    if mode == "all":
        enable(model)
    elif mode == "head":
        enable(model.sem_seg_head.pixel_decoder)
        enable(model.sem_seg_head.predictor)
        if hasattr(model.sem_seg_head.predictor, "language_encoder"):
            for param in model.sem_seg_head.predictor.language_encoder.parameters():
                param.requires_grad = False
    elif mode == "decoder":
        enable(model.sem_seg_head.predictor)
        if hasattr(model.sem_seg_head.predictor, "language_encoder"):
            for param in model.sem_seg_head.predictor.language_encoder.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_loss(predictions: dict, labels: torch.Tensor, pos_weight: float = 3.0) -> torch.Tensor:
    batch_size, num_masks, height, width = labels.shape
    target_masks = (labels.reshape(batch_size * num_masks, 1, height, width) > 0).float()
    pred_masks = predictions["pred_gmasks"]
    if pred_masks.shape[-2:] != (height, width):
        pred_masks = F.interpolate(pred_masks, size=(height, width), mode="bicubic", align_corners=False, antialias=True)

    existence_target = (target_masks.flatten(1).sum(dim=1) > 0).float()
    mask_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks, reduction="none").mean(dim=(1, 2, 3))
    obj_logits = predictions["object_existence"].reshape(-1)
    obj_loss = F.binary_cross_entropy_with_logits(
        obj_logits,
        existence_target,
        pos_weight=torch.tensor(pos_weight, device=obj_logits.device),
        reduction="none",
    )
    return (mask_loss * existence_target + obj_loss).mean()


def save_checkpoint(model: torch.nn.Module, output_path: Path, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {f"model.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "metadata": metadata}, output_path)
    print(f"Saved checkpoint: {output_path}", flush=True)


def jsonable_args(args: argparse.Namespace) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=Path, required=True)
    parser.add_argument("--class_map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("model_weights/biomedparse_v2.ckpt"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/finetune_cvpr_npz_slices"))
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--foreground_prob", type=float, default=0.85)
    parser.add_argument("--trainable", choices=["head", "decoder", "all"], default="decoder")
    parser.add_argument("--amp", choices=["fp16", "bf16", "off"], default="fp16")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--limit_volumes", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Fine-tuning this model requires CUDA")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"Using device: {torch.cuda.get_device_name(device)}", flush=True)

    class_map = json.loads(args.class_map.read_text())
    records = build_volume_records(args.train_root, class_map, args.limit_volumes)
    dataset = CvprNpzSliceDataset(
        records=records,
        class_map=class_map,
        image_size=args.image_size,
        foreground_prob=args.foreground_prob,
        num_prompts=args.num_prompts,
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    model = build_model(args.checkpoint, device)
    trainable_params = set_trainable(model, args.trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable mode={args.trainable} trainable={trainable_params:,} / total={total_params:,}", flush=True)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16")
    autocast_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    use_autocast = args.amp != "off"

    metadata = {
        "script": str(Path(__file__).resolve()),
        "base_checkpoint": str(args.checkpoint),
        "train_root": str(args.train_root),
        "class_map": str(args.class_map),
        "args": jsonable_args(args),
        "records": len(records),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    step = 0
    running_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = {"image": images, "text": list(batch["text"])}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                outputs = model(inputs, mode="train")
                loss = compute_loss(outputs["predictions"], labels) / args.accum_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.accum_steps

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"step={step}/{args.max_steps} loss={running_loss / args.log_every:.5f} "
                    f"elapsed={elapsed:.1f}s peak_alloc={peak_gb:.2f}GiB",
                    flush=True,
                )
                running_loss = 0.0
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(model, args.output_dir / f"checkpoint_step{step:06d}.ckpt", {**metadata, "step": step})

    save_checkpoint(model, args.output_dir / "last.ckpt", {**metadata, "step": step})
    print("Fine-tuning finished.", flush=True)


if __name__ == "__main__":
    main()
