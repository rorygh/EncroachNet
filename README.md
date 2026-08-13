# EncroachNet

RGB drone imagery → 2D semantic segmentation (power lines + vegetation) → backprojected to 3D → catenary fitting + vegetation-clearance encroachment risk.

**Repo:** https://github.com/rorygh/EncroachNet (to be created)

---

## What it does

EncroachNet takes RGB drone imagery of a power-line corridor and outputs a 3D point cloud labeled by class (`background` / `vegetation` / `powerline` / `tower`), plus a georeferenced report of any points where vegetation is closer to a conductor than the regulatory Minimum Vegetation Clearance Distance (MVCD).

```
input:  RGB frames + camera poses (LiDAR-coregistered, or solved via COLMAP)
output: labeled 3D point cloud + per-conductor catenary fit + clearance/risk report
```

Unlike Softgrove (native 3D LiDAR instance segmentation), the learned model here is 2D — a semantic segmentation network trained on drone images — with 3D entering only at the backprojection and post-processing stages.

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for full derivations.

```
RGB images
      │
      ▼
┌─────────────────────┐
│ 2D Semantic Seg      │  SegFormer / HRNetV2, class-weighted focal+Dice loss
│                      │  + skeleton-topology term on the powerline class
└──────────┬───────────┘
           │        Camera poses: LiDAR-flight direct georeferencing,
           │        or ODM SfM/MVS (OpenDroneMap)
           ▼
┌─────────────────────┐
│ Multi-view           │  Visibility-checked backprojection + label voting
│ Backprojection       │  across all cameras that see each 3D point
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ [Optional] Sparse 3D │  spconv U-Net cleans pseudo-labels using local
│ Refinement           │  geometry (same pattern as Softgrove/SimpleUNet)
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Catenary Fit +        │  Per-conductor curve fit (nonlinear least squares)
│ Clearance Computation │  + nearest vegetation-to-conductor distance
└──────────┬───────────┘
           ▼
   Labeled 3D point cloud + encroachment-risk report
```

---

## Quick Start

### Environment

```bash
bash setup-env.sh
```

Installs into system Python via `uv` (no conda on the RunPod base image). Requires CUDA 12.4.

### RunPod / Docker

```bash
docker build --platform linux/amd64 -f Dockerfile.runpod -t rorygh/encroachnet:latest .
docker push rorygh/encroachnet:latest
```

Data at `/workspace/data/`, checkpoints at `/workspace/checkpoints/`. `DATA_ROOT` env var overrides the data path.

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

### Fine-Tuning on Client Imagery

```bash
python finetune2d.py \
    --weights checkpoints/encroachnet_2d_best.ckpt \
    --data path/to/client_labels/ \
    --output_weights checkpoints/encroachnet_2d_finetuned.ckpt
```

---

## Training Data

Two datasets close most of the "no one labels both classes together" gap — see [docs/datasets.md](docs/datasets.md) for the full survey and multi-stage strategy:

| Stage | Purpose | Datasets |
|-------|---------|----------|
| 1 | Synthetic pretraining + pipeline validation | [DDOS](https://huggingface.co/datasets/benediktkol/DDOS) (public, AirSim-simulated flights with trees + power lines + GT depth — also used to validate the ODM/backprojection path) |
| 2 | Real combined-taxonomy fine-tuning | [VEPL](https://zenodo.org/records/7800234) (public, real drone corridor imagery, already in the exact `{vegetation, powerline, background}` taxonomy, ships with DSMs) |
| 3 (optional) | Scale/diversity supplement | [TTPLA](https://arxiv.org/abs/2010.10032) (powerline/tower) + [UAVid](https://arxiv.org/pdf/2109.08937) / Semantic Drone Dataset / VDD (vegetation) |
| 4 | Domain adaptation | Fine-tune on real client drone imagery once available |

## Comparison to State of the Art

Full survey: [docs/sota.md](docs/sota.md). Key references:

| Method | Role | Notes |
|--------|------|-------|
| [Advanced YOLO-based Power Line Detection for Vegetation Management](https://arxiv.org/abs/2503.00044) | Closest published end-to-end system | Bbox-level detection + encroachment metric; EncroachNet targets pixel + 3D precision |
| [2D3DNet](https://arxiv.org/pdf/2110.11325) | Backprojection blueprint | 2D predict → multi-view backproject/vote → sparse 3D refine |
| [PLGAN](https://arxiv.org/pdf/2204.07243) / [Focal Phi Loss](https://www.mdpi.com/1424-8220/21/8/2803) | Thin-wire segmentation losses | Hough-space loss / MCC-generalized focal loss for the 1–5%-of-pixels powerline class |

---

## Project Structure

```
EncroachNet/
├── train2d.py             2D segmentation training entry point
├── infer3d.py              Full pipeline: images + poses → labeled 3D + risk report
├── finetune2d.py           Fine-tune on client imagery
├── setup-env.sh            uv-based environment bootstrap
├── Dockerfile.runpod       RunPod deployment
├── configs/
│   └── default.json        Hyperparameters
├── core/
│   ├── model2d.py           SegFormer / HRNetV2 wrapper
│   ├── losses.py            Focal/Dice + skeleton-topology thin-wire loss
│   ├── dataset2d.py          2D image dataset loader
│   ├── backproject.py        Multi-view 2D→3D label fusion
│   ├── catenary.py           Conductor curve fitting + clearance computation
│   └── inference3d.py        End-to-end pipeline orchestration
├── scripts/
│   ├── download_datasets.sh  Dataset download (run on RunPod, not locally)
│   └── prepare_ttpla.py      Rasterizes TTPLA's polygon annotations to mask PNGs
└── docs/
    ├── architecture.md       Full pipeline math
    ├── sota.md                Model survey
    └── datasets.md            Dataset survey
```

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
