"""
EncroachNet 2D segmentation training entry point.

Usage:
    python train2d.py --config configs/default.json
    python train2d.py --config configs/default.json --batch_size 4 --max_epochs 100
    python train2d.py --config configs/default.json --resume
    python train2d.py --config configs/default.json --sources ttpla,uavid
"""

import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

from core.dataset2d import SegmentationDataset
from core.losses import CombinedSegmentationLoss
from core.model2d import build_model
from core import CLASS_NAMES


class EncroachNet2DModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = build_model(cfg["model2d"])
        self.criterion = CombinedSegmentationLoss(
            class_names=CLASS_NAMES,
            focal_alpha=cfg["loss"]["focal_alpha"],
            focal_gamma=cfg["loss"]["focal_gamma"],
            dice_weight=cfg["loss"]["dice_weight"],
            topo_weight=cfg["loss"]["topo_weight"],
            topo_class=cfg["loss"]["topo_class"],
            num_classes=cfg["model2d"]["num_classes"],
        )

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self.model(batch["image"])
        losses = self.criterion(logits, batch["mask"])
        for k, v in losses.items():
            self.log(f"{stage}/{k}", v, prog_bar=(k == "loss"), batch_size=batch["image"].shape[0])
        return losses["loss"]

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["training"]["learning_rate"],
            weight_decay=self.cfg["training"]["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg["training"]["max_epochs"]
        )
        return [opt], [sched]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sources", default="ttpla,uavid",
                    help="Comma-separated source datasets to merge (see core/dataset2d.py SOURCE_LABEL_MAPS)")
    # Inline overrides
    p.add_argument("--batch_size", type=int)
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--data_root", type=str)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    data_root = os.environ.get("DATA_ROOT", cfg["data"]["root"])
    sources = args.sources.split(",")
    image_size = tuple(cfg["data"]["image_size"])

    train_ds = ConcatDataset([
        SegmentationDataset(f"{data_root}/{s}/train", source=s, image_size=image_size, augment=True)
        for s in sources
    ])
    val_ds = ConcatDataset([
        SegmentationDataset(f"{data_root}/{s}/val", source=s, image_size=image_size, augment=False)
        for s in sources
    ])

    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                           num_workers=cfg["data"]["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                         num_workers=cfg["data"]["num_workers"])

    module = EncroachNet2DModule(cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="encroachnet_2d_{epoch:03d}",
        monitor=cfg["training"]["checkpoint_metric"],
        mode=cfg["training"]["checkpoint_mode"],
        save_last=True,
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger("runs/", name="encroachnet_2d")

    precision = "16-mixed" if cfg["training"]["mixed_precision"] else "32"
    ckpt_path = "checkpoints/last.ckpt" if args.resume else None

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        precision=precision,
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_dl, val_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
