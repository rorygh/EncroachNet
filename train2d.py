"""
EncroachNet 2D segmentation training entry point.

Usage:
    python train2d.py --config configs/default.json
    python train2d.py --config configs/default.json --batch_size 4 --max_epochs 100
    python train2d.py --config configs/default.json --resume
    python train2d.py --config configs/default.json --sources ddos,vepl
"""

import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from core.dataset2d import SegmentationDataset
from core.inference3d import predict_tiled
from core.losses import CombinedSegmentationLoss
from core.model2d import build_model
from core import CLASS_NAMES, CLASS_COLORS, IGNORE_INDEX


def _colorize_mask(mask: torch.Tensor) -> torch.Tensor:
    """(H, W) class-index mask -> (3, H, W) float RGB in [0, 1], using classes.json's class_colors."""
    palette = torch.tensor(
        [[int(hexcolor[i : i + 2], 16) for i in (1, 3, 5)] for hexcolor in CLASS_COLORS],
        dtype=torch.float32,
    ) / 255.0
    rgb = palette[mask.clamp(min=0)]
    rgb[mask == IGNORE_INDEX] = 0.0
    return rgb.permute(2, 0, 1)


class EncroachNet2DModule(pl.LightningModule):
    def __init__(self, cfg: dict, full_res_ref: SegmentationDataset | None = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.full_res_ref = full_res_ref  # one fixed full-resolution val image, for tiled-inference visualization
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
        # average=None -> per-class IoU; mIoU is their mean, logged separately below so a
        # class absent from a given run's ground truth (e.g. vegetation when --sources ttpla)
        # is visible on its own rather than silently blended into one number.
        self.val_iou = MulticlassJaccardIndex(
            num_classes=cfg["model2d"]["num_classes"], ignore_index=IGNORE_INDEX, average=None
        )

    def _step(self, batch: dict, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(batch["image"])
        losses = self.criterion(logits, batch["mask"])
        for k, v in losses.items():
            self.log(f"{stage}/{k}", v, prog_bar=(k == "loss"), batch_size=batch["image"].shape[0])
        return losses["loss"], logits

    def training_step(self, batch, _):
        loss, _ = self._step(batch, "train")
        if torch.cuda.is_available():
            self.log("train/gpu_util_pct", float(torch.cuda.utilization()))
            self.log("train/gpu_mem_gb", torch.cuda.memory_allocated() / 1e9)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self._step(batch, "val")
        self.val_iou.update(logits.argmax(dim=1), batch["mask"])
        if batch_idx == 0:
            self._log_val_images(batch, logits)
        return loss

    def on_validation_epoch_end(self) -> None:
        per_class_iou = self.val_iou.compute()
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            self.log(f"val/iou_{name}", iou)
        self.log("val/miou", per_class_iou.mean())
        self.val_iou.reset()

        if self.full_res_ref is not None:
            self._log_full_image_prediction()

    def _log_full_image_prediction(self, max_width: int = 1536) -> None:
        """Runs core/inference3d.py's tiled inference over one fixed full-resolution
        val image (not the training-scale crops validation_step otherwise sees) --
        this is the closest thing to a real deployment preview available during
        training, since infer3d.py will run on full drone frames the same way.
        """
        image, mask = self.full_res_ref.get_full_resolution(0)
        tile = self.cfg["data"]["image_size"][0]
        probs = predict_tiled(self.model, image, str(self.device), tile=tile, overlap=0.25)
        pred = probs.argmax(-1)

        image_t = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        combined = torch.cat([
            image_t,
            _colorize_mask(torch.from_numpy(mask)),
            _colorize_mask(torch.from_numpy(pred)),
        ], dim=2)  # RGB | ground truth | prediction, side by side

        if combined.shape[-1] > max_width:  # full drone-frame resolution is overkill for a TB thumbnail
            scale = max_width / combined.shape[-1]
            combined = F.interpolate(combined.unsqueeze(0), scale_factor=scale, mode="bilinear",
                                      align_corners=False, recompute_scale_factor=False)[0]

        self.logger.experiment.add_image("val/full_image_prediction", combined, self.current_epoch)

    def _log_val_images(self, batch: dict, logits: torch.Tensor, n: int = 4) -> None:
        preds = logits.argmax(dim=1)
        n = min(n, batch["image"].shape[0])
        rows = [
            torch.cat([
                batch["image"][i].cpu(),
                _colorize_mask(batch["mask"][i].cpu()),
                _colorize_mask(preds[i].cpu()),
            ], dim=2)  # RGB | ground truth | prediction, side by side
            for i in range(n)
        ]
        grid = vutils.make_grid(rows, nrow=1)
        self.logger.experiment.add_image("val/predictions", grid, self.current_epoch)

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
    p.add_argument("--sources", default="ddos,vepl",
                    help="Comma-separated source datasets to merge (see core/dataset2d.py SOURCE_LABEL_MAPS)")
    # Inline overrides
    p.add_argument("--batch_size", type=int)
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--data_root", type=str)
    return p.parse_args()


def main():
    torch.set_float32_matmul_precision("high")  # enable tensor cores (RTX Ada / Ampere+)

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

    module = EncroachNet2DModule(cfg, full_res_ref=val_ds.datasets[0])

    # Name runs by backbone + source datasets so different experiments log to separate
    # TensorBoard curves / checkpoint dirs but with identical metric tags -- directly comparable.
    run_name = f"encroachnet_2d_{cfg['model2d']['backbone']}_{'-'.join(sources)}"

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}/",
        filename="{epoch:03d}",
        monitor=cfg["training"]["checkpoint_metric"],
        mode=cfg["training"]["checkpoint_mode"],
        save_last=True,
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger("runs/", name=run_name)

    precision = "16-mixed" if cfg["training"]["mixed_precision"] else "32"
    ckpt_path = f"checkpoints/{run_name}/last.ckpt" if args.resume else None

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
