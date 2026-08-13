"""Multi-view 2D -> 3D label backprojection and fusion.

Implements docs/architecture.md Stage 3 (2D3DNet-style): for each 3D point,
project into every camera, keep only cameras where the point is actually visible
(depth-buffer check against that camera's depth map), then vote per-class
probability across all visible views.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraPose:
    """A single camera view: intrinsics K, world-to-camera rotation R and
    translation t, plus a depth map (from LiDAR range or MVS) used for the
    visibility check, and the per-pixel class-probability map from the 2D model.
    """
    K: np.ndarray            # (3, 3)
    R: np.ndarray            # (3, 3) world -> camera rotation
    t: np.ndarray            # (3,)   world -> camera translation
    depth_map: np.ndarray    # (H, W) camera-space depth
    class_probs: np.ndarray  # (H, W, num_classes) softmax output of the 2D model


class MultiViewBackprojector:
    """See docs/architecture.md Stage 3 for the visibility-check + voting math."""

    def __init__(self, visibility_eps: float = 0.05, min_views: int = 2,
                 confidence_threshold: float = 0.3):
        self.visibility_eps = visibility_eps
        self.min_views = min_views
        self.confidence_threshold = confidence_threshold

    def _project(self, points: np.ndarray, cam: CameraPose) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pixel_coords (N,2), camera_space_depth (N,))."""
        cam_pts = (cam.R @ points.T).T + cam.t  # (N, 3)
        depth = cam_pts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = (cam.K @ cam_pts.T).T
            pixels = proj[:, :2] / proj[:, 2:3]
        return pixels, depth

    def _visible_mask(self, pixels: np.ndarray, depth: np.ndarray, cam: CameraPose) -> np.ndarray:
        h, w = cam.depth_map.shape
        u, v = pixels[:, 0], pixels[:, 1]
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depth > 0)

        visible = np.zeros(len(pixels), dtype=bool)
        idx = np.where(in_bounds)[0]
        if len(idx) == 0:
            return visible

        uu = u[idx].astype(np.int32)
        vv = v[idx].astype(np.int32)
        map_depth = cam.depth_map[vv, uu]
        consistent = np.abs(map_depth - depth[idx]) < self.visibility_eps
        visible[idx[consistent]] = True
        return visible

    def backproject(self, points: np.ndarray, cameras: list[CameraPose]) -> tuple[np.ndarray, np.ndarray]:
        """points: (N, 3) world-space 3D points.

        Returns:
            labels: (N,) argmax class per point, -1 for points with insufficient
                     camera coverage (fewer than min_views visible views).
            scores: (N, num_classes) accumulated vote scores (docs/architecture.md
                     Stage 3: s_i(c) = sum_k visible(i,k) * p_k,c(u_i,k)).
        """
        num_classes = cameras[0].class_probs.shape[-1]
        n = points.shape[0]
        scores = np.zeros((n, num_classes), dtype=np.float32)
        view_count = np.zeros(n, dtype=np.int32)

        for cam in cameras:
            pixels, depth = self._project(points, cam)
            visible = self._visible_mask(pixels, depth, cam)
            idx = np.where(visible)[0]
            if len(idx) == 0:
                continue

            uu = pixels[idx, 0].astype(np.int32)
            vv = pixels[idx, 1].astype(np.int32)
            probs = cam.class_probs[vv, uu]  # (M, num_classes)

            confident = probs.max(axis=1) >= self.confidence_threshold
            idx = idx[confident]
            probs = probs[confident]

            scores[idx] += probs
            view_count[idx] += 1

        labels = np.full(n, -1, dtype=np.int64)
        covered = view_count >= self.min_views
        labels[covered] = scores[covered].argmax(axis=1)
        return labels, scores
