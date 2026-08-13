"""
Fine-tune the 2D segmentation model on real client-labeled drone imagery.

Workflow:
  1. Collect a small set of client RGB frames and hand-label them (or use
     SAM-assisted labeling for the easy vegetation class -- see docs/sota.md
     Part A on SAM's thin-wire limitation, so still hand-check the powerline
     class) in the unified taxonomy from classes.json.
  2. Lay out `data/client/train/{images,masks}/` and `data/client/val/{images,masks}/`
  3. Run this script:

     python finetune2d.py \
         --weights checkpoints/encroachnet_2d_best.ckpt \
         --data data/client \
         --output_weights checkpoints/encroachnet_2d_finetuned.ckpt

Even a small labeled set is expected to help substantially after public-data
pretraining -- see docs/datasets.md Data Strategy and Softgrove's TreeLite3D
precedent (sparse-label fine-tuning after self-supervised pretraining).
By default only the classification head is fine-tuned (backbone frozen);
pass --full to fine-tune the entire backbone as well.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.dataset2d import SegmentationDataset
from core.losses import CombinedSegmentationLoss
from core.model2d import build_model
from core import CLASS_NAMES


def _freeze_backbone(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name:
            param.requires_grad = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="Root containing train/{images,masks} and val/{images,masks}")
    p.add_argument("--output_weights", default="checkpoints/encroachnet_2d_finetuned.ckpt")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--full", action="store_true", help="Fine-tune full backbone (not just the head)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device(args.device)

    model = build_model(cfg["model2d"])
    state = torch.load(args.weights, map_location=device)
    sd = state.get("state_dict", state)
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)

    if not args.full:
        _freeze_backbone(model)
        print("Fine-tuning classification head only (pass --full for backbone)")
    else:
        print("Fine-tuning full model")

    image_size = tuple(cfg["data"]["image_size"])
    train_ds = SegmentationDataset(f"{args.data}/train", source="client", image_size=image_size, augment=True)
    val_ds = SegmentationDataset(f"{args.data}/val", source="client", image_size=image_size, augment=False)
    print(f"Loaded {len(train_ds)} train / {len(val_ds)} val client-labeled frames")

    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                           num_workers=cfg["data"]["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                         num_workers=cfg["data"]["num_workers"])

    criterion = CombinedSegmentationLoss(
        class_names=CLASS_NAMES,
        focal_alpha=cfg["loss"]["focal_alpha"],
        focal_gamma=cfg["loss"]["focal_gamma"],
        dice_weight=cfg["loss"]["dice_weight"],
        topo_weight=cfg["loss"]["topo_weight"],
        topo_class=cfg["loss"]["topo_class"],
        num_classes=cfg["model2d"]["num_classes"],
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            logits = model(images)
            losses = criterion(logits, masks)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses["loss"].item()
        train_loss /= max(len(train_dl), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                logits = model(images)
                val_loss += criterion(logits, masks)["loss"].item()
        val_loss /= max(len(val_dl), 1)

        print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    Path(args.output_weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_weights)
    print(f"Saved fine-tuned weights -> {args.output_weights}")


if __name__ == "__main__":
    main()
