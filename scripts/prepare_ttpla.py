"""Rasterize TTPLA's Supervisely-format polygon annotations into the flat
{source}/{split}/{images,masks} PNG-pair layout core/dataset2d.py expects.

TTPLA ships polygon JSON (cable / tower_lattice / tower_wooden / tower_tucohy /
void), not raster masks -- this is a one-time conversion, not something the
dataset loader should do on the fly. Raw mask ids below are remapped to the
unified taxonomy by SOURCE_LABEL_MAPS["ttpla"] in core/dataset2d.py, same as
every other source.

Usage:
    python scripts/prepare_ttpla.py --raw /workspace/data/TTPLA --out /workspace/data/encroachnet/ttpla
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Raw TTPLA mask ids (paint order: later entries drawn on top of earlier ones,
# so "void" wins any overlap and gets excluded from the loss via SOURCE_LABEL_MAPS).
RAW_CLASS_IDS = {
    "cable": 1,
    "tower_lattice": 2,
    "tower_wooden": 3,
    "tower_tucohy": 4,
    "void": 5,
}


def rasterize(ann: dict) -> np.ndarray:
    h, w = ann["size"]["height"], ann["size"]["width"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in ann["objects"]:
        class_id = RAW_CLASS_IDS.get(obj["classTitle"])
        if class_id is None:
            continue
        exterior = np.array(obj["points"]["exterior"], dtype=np.int32)
        cv2.fillPoly(mask, [exterior], class_id)
        for hole in obj["points"].get("interior", []):
            cv2.fillPoly(mask, [np.array(hole, dtype=np.int32)], 0)
    return mask


def prepare_split(raw_root: Path, out_root: Path, split: str) -> None:
    img_dir, ann_dir = raw_root / split / "img", raw_root / split / "ann"
    out_images, out_masks = out_root / split / "images", out_root / split / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    ann_paths = sorted(ann_dir.glob("*.json"))
    for ann_path in ann_paths:
        stem = ann_path.stem.removesuffix(".jpg")
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue

        mask = rasterize(json.loads(ann_path.read_text()))
        cv2.imwrite(str(out_masks / f"{stem}.png"), mask)

        link = out_images / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(img_path.resolve())

    print(f"{split}: {len(ann_paths)} pairs -> {out_root / split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/data/TTPLA", help="Extracted TTPLA Supervisely-format root (train/val/test with img/ann subfolders)")
    ap.add_argument("--out", default="/workspace/data/encroachnet/ttpla")
    args = ap.parse_args()

    raw_root, out_root = Path(args.raw), Path(args.out)
    for split in ("train", "val"):
        prepare_split(raw_root, out_root, split)


if __name__ == "__main__":
    main()
