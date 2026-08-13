"""End-to-end pipeline orchestration: RGB images + camera poses -> labeled 3D
point cloud + catenary fits + clearance/risk report. See docs/architecture.md
for the full stage-by-stage description; this module wires the stages together.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core import CLASS_NAMES
from core.backproject import CameraPose, MultiViewBackprojector
from core.catenary import fit_all_conductors, flag_risk_points


@dataclass
class PipelineResult:
    points: np.ndarray                # (N, 3)
    labels: np.ndarray                # (N,) class index per point, -1 = unlabeled
    catenaries: list = field(default_factory=list)
    clearance_distances: np.ndarray | None = None
    risk_mask: np.ndarray | None = None


def _tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Sliding-window start coordinates along one axis, covering [0, size) with the
    last tile flush against the far edge even when stride doesn't evenly divide size."""
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


@torch.no_grad()
def _predict_tiled(model: torch.nn.Module, image: np.ndarray, device: str,
                    tile: int, overlap: float) -> np.ndarray:
    """Full-resolution drone frames (~4000-8000px) are both too large to fit in GPU
    memory at once and too large-scale relative to what the model was trained on
    (512x512 crops) -- feeding a whole frame in directly OOMs and, even if it fit,
    would show the model wires at the wrong apparent scale. Standard fix: run the
    model over overlapping tiles at training resolution and average the overlap
    regions back into one full-size probability map, avoiding seams at tile borders.
    """
    h, w = image.shape[:2]
    if h <= tile and w <= tile:
        tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
        return F.softmax(model(tensor), dim=1)[0].permute(1, 2, 0).cpu().numpy()

    stride = max(1, int(tile * (1 - overlap)))
    prob_sum, weight = None, np.zeros((h, w), dtype=np.float32)

    for y in _tile_origins(h, tile, stride):
        for x in _tile_origins(w, tile, stride):
            crop = image[y:y + tile, x:x + tile]
            tensor = torch.from_numpy(crop / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
            probs = F.softmax(model(tensor), dim=1)[0].permute(1, 2, 0).cpu().numpy()
            if prob_sum is None:
                prob_sum = np.zeros((h, w, probs.shape[-1]), dtype=np.float32)
            prob_sum[y:y + crop.shape[0], x:x + crop.shape[1]] += probs
            weight[y:y + crop.shape[0], x:x + crop.shape[1]] += 1.0

    return prob_sum / weight[..., None]


def run_2d_segmentation(model: torch.nn.Module, images: list[np.ndarray], device: str = "cuda",
                         tile: int = 512, overlap: float = 0.25) -> list[np.ndarray]:
    """Runs the trained 2D model over a batch of RGB frames, returns per-frame
    softmax class-probability maps (H, W, num_classes) for backprojection.
    `tile` should match the resolution the model was trained at (cfg["data"]["image_size"]).
    """
    model.eval()
    return [_predict_tiled(model, image, device, tile, overlap) for image in images]


def run_pipeline(model: torch.nn.Module, images: list[np.ndarray], cameras_raw: list[dict],
                  points: np.ndarray, cfg: dict, device: str = "cuda") -> PipelineResult:
    """cameras_raw: list of {"K": (3,3), "R": (3,3), "t": (3,), "depth_map": (H,W)}
    aligned 1:1 with `images` (see docs/architecture.md Stage 2 for how these are
    sourced -- LiDAR-flight direct georeferencing, or COLMAP for RGB-only flights).
    """
    class_probs = run_2d_segmentation(model, images, device, tile=cfg["data"]["image_size"][0])

    cameras = [
        CameraPose(K=c["K"], R=c["R"], t=c["t"], depth_map=c["depth_map"], class_probs=probs)
        for c, probs in zip(cameras_raw, class_probs)
    ]

    backprojector = MultiViewBackprojector(**cfg["backproject"])
    labels, _scores = backprojector.backproject(points, cameras)

    powerline_idx = CLASS_NAMES.index("powerline")
    vegetation_idx = CLASS_NAMES.index("vegetation")

    powerline_points = points[labels == powerline_idx]
    vegetation_points = points[labels == vegetation_idx]

    catenaries = fit_all_conductors(
        powerline_points,
        eps=cfg["catenary"].get("cluster_eps", 1.0),
        min_samples=cfg["catenary"]["min_points_per_span"],
    )

    distances, risk_mask = flag_risk_points(
        vegetation_points, catenaries, cfg["clearance"]["mvcd_meters"]
    )

    return PipelineResult(
        points=points, labels=labels, catenaries=catenaries,
        clearance_distances=distances, risk_mask=risk_mask,
    )


def write_report(result: PipelineResult, output_dir: str) -> None:
    """Writes a labeled point cloud (npz) and a plain-text clearance/risk summary.
    Swap the point-cloud writer for a LAZ export (laspy, matching Softgrove's
    infer.py convention) once the client's preferred GIS format is confirmed.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez(
        out / "labeled_points.npz",
        points=result.points, labels=result.labels,
    )

    n_risk = int(result.risk_mask.sum()) if result.risk_mask is not None else 0
    with open(out / "risk_report.txt", "w") as f:
        f.write(f"Conductors fit: {len(result.catenaries)}\n")
        f.write(f"Vegetation points flagged below MVCD: {n_risk}\n")
        if result.clearance_distances is not None and len(result.clearance_distances) > 0:
            f.write(f"Min clearance observed: {result.clearance_distances.min():.2f} m\n")
