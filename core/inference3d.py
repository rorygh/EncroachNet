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


@torch.no_grad()
def run_2d_segmentation(model: torch.nn.Module, images: list[np.ndarray],
                         device: str = "cuda") -> list[np.ndarray]:
    """Runs the trained 2D model over a batch of RGB frames, returns per-frame
    softmax class-probability maps (H, W, num_classes) for backprojection.
    """
    model.eval()
    outputs = []
    for image in images:
        tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].permute(1, 2, 0).cpu().numpy()
        outputs.append(probs)
    return outputs


def run_pipeline(model: torch.nn.Module, images: list[np.ndarray], cameras_raw: list[dict],
                  points: np.ndarray, cfg: dict, device: str = "cuda") -> PipelineResult:
    """cameras_raw: list of {"K": (3,3), "R": (3,3), "t": (3,), "depth_map": (H,W)}
    aligned 1:1 with `images` (see docs/architecture.md Stage 2 for how these are
    sourced -- LiDAR-flight direct georeferencing, or COLMAP for RGB-only flights).
    """
    class_probs = run_2d_segmentation(model, images, device)

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
