"""
EncroachNet end-to-end 3D inference: RGB images + camera poses -> labeled 3D
point cloud + catenary fits + clearance/risk report.

Usage:
    python infer3d.py \
        --weights checkpoints/encroachnet_2d_best.ckpt \
        --images path/to/flight/images/ \
        --poses path/to/flight/poses.json \
        --points path/to/flight/points.laz \
        --output output/
        [--config configs/default.json]
        [--device cuda]

`poses.json` format: a list of {"image": "<filename>", "K": [[..]], "R": [[..]],
"t": [..], "depth_map": "<path.npy>"} entries -- one per image, aligned by
filename. See docs/architecture.md Stage 2 for how poses are sourced (LiDAR
direct georeferencing, or COLMAP for RGB-only flights via LogMotion).

Output: <output>/labeled_points.npz (points + per-point class labels) and
<output>/risk_report.txt (clearance/encroachment summary).
"""

import argparse
import json
from pathlib import Path

import cv2
import laspy
import numpy as np
import torch

from core.inference3d import run_pipeline, write_report
from core.model2d import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--points", required=True, help="LAZ/LAS point cloud to label (LiDAR path) "
                                                     "or COLMAP dense reconstruction export")
    p.add_argument("--output", default="output/")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_state_dict(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    state = torch.load(weights_path, map_location=device)
    sd = state.get("state_dict", state)
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    model.load_state_dict(sd)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device(args.device)

    model = build_model(cfg["model2d"])
    _load_state_dict(model, args.weights, device)
    model.to(device).eval()

    with open(args.poses) as f:
        pose_entries = json.load(f)

    image_dir = Path(args.images)
    images, cameras_raw = [], []
    for entry in pose_entries:
        img = cv2.cvtColor(cv2.imread(str(image_dir / entry["image"])), cv2.COLOR_BGR2RGB)
        images.append(img)
        cameras_raw.append({
            "K": np.array(entry["K"], dtype=np.float64),
            "R": np.array(entry["R"], dtype=np.float64),
            "t": np.array(entry["t"], dtype=np.float64),
            "depth_map": np.load(entry["depth_map"]),
        })
    print(f"Loaded {len(images)} frames with poses from {args.poses}")

    las = laspy.read(args.points)
    points = np.vstack([las.x, las.y, las.z]).T
    print(f"Loaded {len(points):,} 3D points from {args.points}")

    result = run_pipeline(model, images, cameras_raw, points, cfg, device=str(device))

    n_labeled = int((result.labels >= 0).sum())
    print(f"Labeled {n_labeled:,} / {len(points):,} points from multi-view backprojection")
    print(f"Fit {len(result.catenaries)} conductor catenaries")
    if result.risk_mask is not None:
        print(f"Flagged {int(result.risk_mask.sum())} vegetation points below MVCD "
              f"({cfg['clearance']['mvcd_meters']} m)")

    write_report(result, args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
