# EncroachNet Architecture

Full pipeline description: 2D segmentation, multi-view backprojection, optional 3D refinement, and catenary-based encroachment-risk computation. See `docs/sota.md` for literature and `docs/datasets.md` for training data.

---

## Overview

```
RGB images {I_k}           (drone frames, known intrinsics K_k)
      │
      ▼
2D Semantic Segmentation    per-pixel class probs p_k(u,v) ∈ Δ³  {bg, vegetation, powerline}
      │
      ▼
Camera poses {K_k, R_k, t_k}   (from LiDAR-flight direct georeferencing, or COLMAP SfM)
      │
      ▼
Multi-view Backprojection    per 3D point x_i: fuse p_k(π(x_i)) over all visible k → label ŷ_i
      │
      ▼
[Optional] Sparse 3D Refinement    spconv U-Net cleans ŷ_i using local geometry
      │
      ▼
Catenary Fit + Clearance     conductor centerlines C_j(t); d(x_i, C_j) per vegetation point
      │
      ▼
Labeled 3D point cloud + georeferenced risk report
```

---

## Stage 1: 2D Semantic Segmentation

Backbone: SegFormer or HRNetV2 (compare both — see `docs/sota.md` Part A). Output per-pixel class probabilities over `{background, vegetation, powerline}` (optionally `tower` as a 4th class).

### Class-imbalance loss (Focal Phi Loss family)

Power-line pixels are ~1–5% of any given image. Standard focal loss:

```
L_focal = -Σ_{u,v} α_c (1 - p_c(u,v))^γ · log(p_c(u,v))
```

where `c` is the ground-truth class at pixel `(u,v)`, `α_c` is a per-class weight (small for background, large for powerline), `γ` (default 2.0) down-weights easy/already-confident pixels. This is the same imbalance-handling family used in Softgrove for the tree-top heatmap head, applied here to the powerline class instead of tree apexes — both are cases of a tiny, spatially-sparse foreground class against a large uniform background.

The Focal Phi variant generalizes the weighting term from the Matthews Correlation Coefficient rather than a fixed `α`, empirically shown to converge faster and score higher precision/Dice on power-line data specifically (`docs/sota.md`).

### Topology-preserving term (thin-wire connectivity)

Per-pixel loss alone permits high IoU with broken wire connectivity (small gaps score a negligible IoU penalty but are fatal to the downstream catenary fit, which needs a roughly continuous point set per span). Add a skeleton-based term on the powerline class:

```
L_topo = 1 - clDice(P̂, Y) = 1 - 2 · (T_prec · T_sens) / (T_prec + T_sens)

T_prec = |S(P̂) ∩ Y| / |S(P̂)|      (fraction of predicted skeleton inside GT mask)
T_sens = |S(Y) ∩ P̂| / |S(Y)|       (fraction of GT skeleton inside predicted mask)
```

where `S(·)` is the morphological skeleton. Use the cheaper Skeleton Recall Loss formulation in practice (see `docs/sota.md`) — same intent, lower compute overhead than full soft-clDice.

### Total 2D loss

```
L_2D = λ_focal · L_focal + λ_dice · L_dice + λ_topo · L_topo(powerline only)
```

Default weights: λ_focal = 1.0, λ_dice = 1.0, λ_topo = 0.5 (tune per validation connectivity metric, not just mIoU — see Verification in the plan).

---

## Stage 2: Camera Poses / 3D Geometry

Two supported paths, selected per-flight based on what's available:

**Path A — LiDAR-coregistered RGB** (Wingtra+Hesai style, matching Softgrove's existing data): camera poses come from onboard PPK/RTK + IMU direct georeferencing (no SfM needed); 3D points are the existing LiDAR point cloud. Preferred when available — LiDAR is denser and geometrically more accurate on non-wire surfaces than photogrammetric MVS, and wire points (however sparse) are still real 3D returns rather than MVS-hallucinated surface fill.

**Path B — RGB-only**: run [OpenDroneMap (ODM)](https://github.com/OpenDroneMap/ODM) rather than hand-chaining raw COLMAP. ODM is purpose-built for exactly this kind of multi-image drone-corridor capture — its pipeline is OpenSfM (poses) → OpenMVS (dense reconstruction/depth maps) → orthophoto/DSM — whereas the COLMAP pipeline already on disk at `C:\rory\scripts\LogMotion` (`01_extract_frames.py` → `02_colmap_sparse.py` → `03_dense.py`) has only been exercised on video/ground-level scenes and would need adaptation for aerial multi-view geometry. ODM exposes both artifacts Stage 3 needs directly: `shots.geojson` (per-image position + orientation → `R, t`) and per-image OpenMVS depth maps (→ `depth_map` in `core/backproject.py`'s `CameraPose`). It also produces an orthomosaic/DSM as a byproduct, which folds the earlier "pre-built orthomosaic + DSM" option into the same tool rather than a separate path. Expect MVS (via either tool) to poorly reconstruct or entirely omit wire points (thin, low-texture, often sub-pixel) — this is exactly why Stage 5's catenary fit does not trust raw point density and instead fits a parametric model through whatever sparse/noisy wire points do exist.

---

> **⚠ NEEDS REVIEW — Stage 3 onward.** Everything from here down (multi-view backprojection, optional 3D refinement, catenary fitting, clearance computation) is candidate design, not yet confirmed. Flagging per Rory's request before picking this back up.

---

## Stage 3: Multi-View Backprojection  ⚠ NEEDS REVIEW

For each 3D point `x_i` and each camera `k` with pose `(K_k, R_k, t_k)`:

**Projection:**
```
u_i,k = K_k (R_k x_i + t_k)     (homogeneous image-plane coordinates)
```

**Visibility check** (depth-buffer test): point `x_i` is counted as visible in camera `k` only if its camera-space depth matches the camera's depth map (from LiDAR range, or MVS depth) within tolerance `ε`:

```
visible(i, k) = 1  iff  |depth(x_i, k) - D_k(u_i,k)| < ε
```

This excludes points occluded by nearer geometry (e.g. a vegetation point behind the actual visible foliage surface, or a wire point behind a tower member) from contributing false votes.

**Multi-view label fusion** (2D3DNet-style voting): accumulate per-class probability across all visible views, weighted by 2D-model confidence:

```
s_i(c) = Σ_k visible(i, k) · p_k,c(u_i,k)

ŷ_i = argmax_c s_i(c)
```

Points with no visible camera (`Σ_k visible(i,k) = 0`) are left unlabeled and excluded from downstream steps.

---

## Stage 4 (Optional): Sparse 3D Refinement  ⚠ NEEDS REVIEW

The backprojected labels `ŷ_i` are noisy at object boundaries (projection/calibration error, imperfect visibility checks) and sparse where camera coverage is poor. A lightweight sparse-convolution U-Net (same architecture pattern as Softgrove's and SimpleUNet's backbones — voxelize, encode, decode, per-point classification head) can be trained against these pseudo-labels to smooth predictions using local 3D geometric consistency, following 2D3DNet's third stage. This does not change the label taxonomy, only cleans it.

---

## Stage 5: Catenary Fitting  ⚠ NEEDS REVIEW

Conductors sag as a catenary between towers:

```
z(s) = a · cosh(s / a) + c
```

where `s` is horizontal distance along the span's principal direction, `a` is the catenary parameter (related to conductor tension/weight), `c` is a vertical offset.

**Pipeline:**
1. **Cluster** labeled powerline points into per-span, per-conductor groups (connected-component or RANSAC line-grouping in the plane perpendicular to the corridor's principal direction — multiple parallel conductors per span must be separated before fitting).
2. **Rotate/project** each cluster into a local 2D coordinate frame aligned with the span direction: `(s, z)`.
3. **Fit** `a, c` (and the span's horizontal offset) by nonlinear least-squares:
   ```
   (â, ĉ) = argmin_{a,c} Σ_i ( z_i - a·cosh(s_i/a) - c )²
   ```
   solved iteratively (Newton-Raphson on the residual gradient, since `cosh` is nonlinear in `a`). Literature-reported fitting residuals are sub-10 cm even from sparse, noisy input points.
4. The fitted curve `C_j(s) = (s, a_j cosh(s/a_j) + c_j)` — not the raw points — becomes conductor `j`'s 3D centerline for clearance computation.

---

## Stage 6: Clearance / Encroachment-Risk Computation  ⚠ NEEDS REVIEW

For each labeled vegetation point `x_i` (or per-tree-crown cluster centroid, if a LiDAR channel with usable crown structure is present — reuse Softgrove's tree-top/instance clustering rather than reimplementing):

```
d_i = min_j min_s || x_i - C_j(s) ||₂        (nearest distance to any conductor centerline)

risk_i = 1  iff  d_i < MVCD
```

where MVCD is the utility's regulatory Minimum Vegetation Clearance Distance (industry-standard framing — see `docs/sota.md` Part D). Output: georeferenced point cloud with per-point class + clearance distance, and a derived risk-zone layer (contiguous regions/points below MVCD) for reporting.

---

## Key Design Choices

| Choice | Rationale |
|---|---|
| Focal/Dice + skeleton-topology combined loss | Powerline pixels are a tiny, spatially sparse class (same imbalance shape as Softgrove's tree-top heatmap); topology term specifically protects wire connectivity, which per-pixel IoU does not reward |
| HRNetV2 as a baseline candidate | Preserves full-resolution features throughout the network — literature-credited advantage for thin/tiny structures without extra post-processing |
| Visibility-checked multi-view voting (not naive projection) | Prevents occluded points from receiving false votes from cameras that see something else at the same image location |
| LiDAR-first when available (Path A) | Photogrammetric MVS is known to poorly reconstruct thin wires; LiDAR returns, however sparse, are still real measurements rather than interpolated/hallucinated surface fill |
| Catenary fit over raw point trust | Both MVS and LiDAR undersample wires; fitting the known physical curve model recovers an accurate centerline from sparse, noisy input |
| Optional sparse-3D refinement stage | Matches 2D3DNet's validated third stage; reuses existing spconv U-Net infrastructure from Softgrove/SimpleUNet rather than introducing new 3D-model code |
