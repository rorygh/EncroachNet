# EncroachNet

RGB drone imagery → 2D semantic segmentation of power lines and vegetation → backprojected to 3D → catenary fitting + vegetation-clearance encroachment risk.

## Project Overview

EncroachNet trains a 2D semantic segmentation model on drone RGB imagery to classify pixels as `background` / `vegetation` / `powerline` / `tower`, then backprojects those per-pixel predictions into 3D using known camera poses (either from LiDAR-flight direct georeferencing or from COLMAP structure-from-motion), fuses labels across multi-view observations of each 3D point, fits catenary curves to the resulting conductor point clusters, and computes vegetation-to-conductor clearance distances against a regulatory Minimum Vegetation Clearance Distance (MVCD) threshold.

This is a sibling project to Softgrove (native 3D LiDAR tree instance segmentation) but architecturally different: the learned model here is 2D, and 3D only enters at the backprojection/post-processing stage.

## Architecture

See [docs/architecture.md](docs/architecture.md) for full mathematical derivations.

### Pipeline Summary

1. **2D Semantic Segmentation** (`core/model2d.py`): SegFormer or HRNetV2 backbone, trained with a class-weighted focal+Dice loss plus a skeleton-based topology-preserving term on the powerline class (thin-wire connectivity matters for the downstream catenary fit even when per-pixel IoU is already reasonable).
2. **Camera Pose Resolution**: two paths — reuse existing LiDAR-flight direct georeferencing (PPK/RTK+IMU) when a co-registered LiDAR channel is present, or run [OpenDroneMap (ODM)](https://github.com/OpenDroneMap/ODM) for RGB-only flights (purpose-built for multi-image drone-corridor capture, unlike the COLMAP pipeline at `C:\rory\scripts\LogMotion`, which has only been exercised on video/ground-level scenes).
3. **Multi-View Backprojection** (`core/backproject.py`): for each 3D point, depth-buffer visibility check against every camera, then vote/fuse per-class probabilities across all visible views.
4. **Optional Sparse 3D Refinement**: lightweight spconv U-Net (same pattern as Softgrove/SimpleUNet) cleans noisy backprojected pseudo-labels using local 3D geometry.
5. **Catenary Fitting** (`core/catenary.py`): cluster wire points per conductor span, fit a catenary curve by nonlinear least-squares — recovers a clean centerline even from the sparse, noisy wire points both MVS and LiDAR tend to produce.
6. **Clearance / Risk Computation**: nearest-point-to-curve distance per vegetation point, thresholded against MVCD.

## Directory Structure

```
EncroachNet/
├── train2d.py             # 2D segmentation training entry point
├── infer3d.py              # Full pipeline: images + poses -> labeled 3D + risk report
├── finetune2d.py           # Fine-tuning on client imagery
├── setup-env.sh            # uv-based environment bootstrap (run once)
├── Dockerfile.runpod       # RunPod deployment image
├── requirements.txt        # Python dependencies
├── classes.json            # Semantic class registry
├── configs/
│   └── default.json        # Default hyperparameters
├── core/
│   ├── __init__.py          # Shared constants (loads classes.json)
│   ├── model2d.py            # SegFormer / HRNetV2 wrapper
│   ├── losses.py             # Focal/Dice + skeleton-topology thin-wire loss
│   ├── dataset2d.py           # 2D image dataset loader
│   ├── backproject.py         # Multi-view 2D->3D label fusion
│   ├── catenary.py            # Conductor curve fitting + clearance computation
│   └── inference3d.py         # End-to-end pipeline orchestration
├── scripts/
│   ├── download_datasets.sh  # Dataset download (run on RunPod, not locally)
│   └── prepare_ttpla.py      # Rasterizes TTPLA's polygon annotations to mask PNGs
├── docs/
│   ├── architecture.md       # Full mathematical architecture description
│   ├── sota.md                # State-of-the-art model survey
│   └── datasets.md            # Available training dataset survey
├── checkpoints/               # Saved weights (auto-created, gitignored)
├── runs/                      # TensorBoard logs (auto-created, gitignored)
└── output/                    # Exported labeled point clouds + risk reports (auto-created, gitignored)
```

## Quick Start

### Environment Setup

```bash
bash setup-env.sh
```

Installs into system Python via `uv` — no conda on the RunPod base image (see `setup-env.sh`).

### 2D Segmentation Training

```bash
python train2d.py --config configs/default.json
```

### End-to-End 3D Inference

```bash
python infer3d.py \
    --weights checkpoints/encroachnet_2d_best.ckpt \
    --images path/to/flight/images/ \
    --poses path/to/flight/poses.json \
    --output output/
```

## RunPod Deployment

1. Build and push:
   ```bash
   docker build --platform linux/amd64 -f Dockerfile.runpod -t rorygh/encroachnet:latest .
   docker push rorygh/encroachnet:latest
   ```
2. Launch pod with RTX 3090 / A6000 or similar; CUDA 12.4 base image.
3. Mount a network volume at `/workspace/`; data at `/workspace/data/`, checkpoints at `/workspace/checkpoints/`.
4. Environment variable `DATA_ROOT` overrides the default data path.

## Git

- Remote: https://github.com/rorygh/EncroachNet (to be created)
- Commit email: rory@mcclenagan.net

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` / `timm` | SegFormer / HRNetV2 backbones |
| `pytorch-lightning` | Training loop + checkpointing |
| `pycolmap` | Reading COLMAP camera poses (RGB-only path) |
| `spconv-cu124` | Optional sparse 3D refinement stage |
| `laspy[lazrs]` | LiDAR point cloud I/O (LiDAR-coregistered path) |
| `scikit-image` | Skeletonization for topology-preserving loss |
| `open3d` | Point cloud visualization / IO |

## Research Context

- **Core problem asymmetry**: vegetation segmentation is a solved problem (any competent aerial segmentation backbone handles it); power-line segmentation is the research risk — wires are 1–5% of pixels, low-texture, easily confused with clutter.
- **Combined-taxonomy data exists, at limited scale**: [VEPL](https://zenodo.org/records/7800234) (real, public, ~2.4 km corridor, exact target taxonomy + DSMs) and [DDOS](https://huggingface.co/datasets/benediktkol/DDOS) (public, synthetic AirSim flights with trees + power lines + GT depth) both cover power lines and vegetation together. Older powerline-only (TTPLA, InsPLAD) and vegetation-only (UAVid, Semantic Drone, VDD) datasets remain useful as a scale/diversity supplement — see [docs/datasets.md](docs/datasets.md).
- **No in-house synthetic data generation** — DDOS already fills that role (public, pre-built, includes GT depth), so building a bespoke Blender/SynthBlend-style corridor renderer is out of scope.
- **Closest published system**: [Advanced YOLO-based Real-time Power Line Detection for Vegetation Management](https://arxiv.org/abs/2503.00044) (2025) — bbox-level detection + encroachment metric; EncroachNet aims for pixel + 3D precision on the same problem.
- **RGB-only pose/depth path**: use [OpenDroneMap (ODM)](https://github.com/OpenDroneMap/ODM), not the COLMAP pipeline at `C:\rory\scripts\LogMotion` (that one has only been exercised on video/ground-level scenes, not drone-corridor multi-view geometry).
- **Local reusable tooling**: `C:\rory\scripts\search3d\...\semantic_sam` (vendored Mask2Former/Semantic-SAM codebase), Softgrove/SimpleUNet's spconv sparse U-Net pattern for the optional 3D refinement stage.
