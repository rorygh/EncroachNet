# Datasets: RGB Powerline + Vegetation Segmentation

Survey of publicly available datasets, organized by role in the training pipeline. See `docs/sota.md` for the corresponding model survey.

Last updated: August 2026

---

## The Core Gap

**No public dataset labels power lines and vegetation together in the same aerial scene.** Powerline-specific datasets (TTPLA, InsPLAD, CPLID) do not label vegetation at all; vegetation/land-cover aerial datasets (UAVid, Semantic Drone, VDD) do not label power lines. Every existing published powerline-detection-for-vegetation-management system (e.g. the YOLO-OBB paper in `docs/sota.md`) works around this by running a powerline detector and a generic "clutter near the line" heuristic rather than a true joint-taxonomy segmentation model.

This means EncroachNet's data strategy has to be constructed, not simply downloaded — see **Data Strategy** below.

---

## Stage 1: Powerline-Specific 2D Datasets

### TTPLA — Primary Powerline/Tower Source ★
**Paper:** Abdelfattah et al., *"TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines"* ([arXiv 2010.10032](https://arxiv.org/abs/2010.10032))
**Code/data:** https://github.com/R3ab/ttpla_dataset

| Property | Value |
|---|---|
| Images | 1,100 @ 3,840×2,160 |
| Instances | 8,987 manually labeled (towers + power lines) |
| Labels | Supports detection, semantic segmentation, and instance segmentation |
| Classes | Transmission towers, power lines (cables) |

The only major public dataset with pixel-level power-line labels at reasonable scale. No vegetation class — will need to be merged with a vegetation-labeled dataset under a unified taxonomy (see Data Strategy).

### InsPLAD — Component/Defect Inspection
**Paper:** *"InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images"* ([arXiv 2311.01619](https://arxiv.org/abs/2311.01619))
**Code:** https://github.com/andreluizbvs/InsPLAD

| Property | Value |
|---|---|
| Images | 10,607 UAV color images |
| Instances | 28,933 across 17 unique power-line asset classes (insulators, spacers, dampers, etc.) |
| Defects | 6 types (4 corrosion, 1 broken component, 1 bird's nest) |
| Tasks | Object detection (AP), defect classification (balanced accuracy), anomaly detection (AUROC) |

Component/defect-level, not full-scene wire segmentation — not directly useful for the encroachment task, but a natural extension once asset-inspection features are wanted later.

### CPLID + Weather-Degraded Variants
Chinese Power Line Insulator Dataset — insulator detection/segmentation. Synthesized weather-degraded variants exist (HazeCPLID/HazeTTPLA/HazeInsPLAD, RainCPLID/RainTTPLA/RainInsPLAD, SnowCPLID/SnowInsPLAD) for image restoration under fog/rain/snow prior to segmentation. Secondary priority — useful once robustness to weather becomes a requirement, not for the initial model.

---

## Stage 2: Vegetation / General Aerial 2D Datasets

### UAVid
**Paper:** [arXiv 2109.08937](https://arxiv.org/pdf/2109.08937) (UNetFormer paper benchmarks on it)

| Property | Value |
|---|---|
| Images | 420 @ 3,840×2,160 (200 train / 70 val / 150 test) |
| Classes | 8: building, road, **tree**, **low vegetation**, static car, moving car, human, background |
| Scene | Urban street-level UAV scenes |

Standard benchmark for aerial transformer segmentation (SegFormer, Mask2Former, BANet, UNetFormer all reported — BANet: 64.6% mIoU). Source of the vegetation class and of backbone benchmarking numbers for `docs/sota.md`.

### Semantic Drone Dataset (Graz)
Institute of Computer Graphics and Vision, TU Graz.

| Property | Value |
|---|---|
| Images | 400 train / 200 test @ 6,000×4,000 |
| Altitude | 5–30 m AGL (bird's-eye) |
| Classes | 22, incl. **tree, vegetation, grass**, paved area, roof, fence, person, car, obstacle |

Altitude range (5–30 m) is notably closer to typical corridor-inspection flying height than mapping-altitude datasets, which matters for apparent wire thickness and vegetation-crown resolution.

### VDD (Varied Drone Dataset)
**Paper:** [arXiv 2305.13608](https://arxiv.org/html/2305.13608v3)

Newer/broader drone semantic segmentation dataset; additional vegetation-class diversity beyond UAVid/Semantic Drone. Worth including in the merged-taxonomy pretraining mix for scene diversity.

---

## Stage 3: 3D / LiDAR Powerline Data (if a LiDAR channel is available)

### DALES (cross-referenced from Softgrove's dataset survey)
**Website:** https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales

| Property | Value |
|---|---|
| Points | 500M+ (hand-labeled) |
| Classes | 8, including explicit **"power lines"** and **"poles"** |
| Sensor | Real airborne LiDAR |

Already known to Rory via Softgrove's semantic pre-training stage; directly relevant here too as a 3D-side validation/pretraining source for the powerline class when a LiDAR point cloud is part of the pipeline (see `docs/sota.md` Part C on the direct 3D-native alternative).

### Powerline-corridor point-cloud segmentation literature
No single standard public benchmark dataset was identified for powerline-corridor LiDAR segmentation specifically (the relevant 2025 papers — elevation-aware multi-resolution network, aerial-slender-target segmentation — largely use proprietary utility-corridor scans). Treat this as a gap to fill with the client's own co-registered LiDAR data if that path is used.

---

## Synthetic Data Precedent

No public dataset combines powerline + vegetation at corridor scale, but synthetic generation is well-precedented for the powerline class specifically: synthetic high-voltage insulator image datasets have been built by importing public CAD models of towers/insulators into game engines, varying camera pose/lighting/background, with **automatic ground-truth mask generation** via color-keyed rendering. This validates that thin metallic wire/tower structures render and label cleanly in a synthetic pipeline — the same logic already proven for Rory's own work:

- **Boreal3D** (used in Softgrove) — synthetic forest LiDAR pretraining, pretrain-then-finetune-on-20%-real matches full-real-data performance.
- **SynthBlend** (`C:\rory\scripts\aerosynth\SimpleUNet`) — Rory's own existing synthetic RGB/point-cloud render pipeline (Blender/GTA-V based), already used for 3D semantic segmentation pretraining.

EncroachNet can plausibly follow the same recipe: procedurally place towers, catenary-curved wires, and tree/vegetation models in a 3D scene (Blender), render RGB + depth + per-pixel semantic masks from realistic UAV corridor-flight camera trajectories, and use this as Stage 0 pretraining data before merged-real-dataset pretraining and client fine-tuning.

---

## Data Strategy (Recommended Multi-Stage Pipeline)

```
Stage 0 — Synthetic corridor pretraining (optional but recommended)
  → Procedurally rendered towers + catenary wires + trees (Blender), following
    the SynthBlend/Boreal3D precedent already validated for Rory's other projects

Stage 1 — Merged-taxonomy real-data pretraining
  → TTPLA (powerline/tower classes) + UAVid / Semantic Drone / VDD (vegetation
    classes), remapped to a shared label space: {background, vegetation, powerline}
    (+ optionally tower, as a 4th class, since TTPLA already labels it)

Stage 2 — Domain adaptation to client corridor imagery
  → Fine-tune on real client drone RGB once available, following the same
    sparse-label fine-tuning precedent already validated for Softgrove
    (finetune.py; TreeLite3D's "1 annotated tree" result — see Softgrove's
    docs/sota.md, Part A, TreeLite3D section)
```

### Label-space merging notes

| Unified class | TTPLA | UAVid | Semantic Drone | VDD |
|---|---|---|---|---|
| `background` | (implicit) | building, road, car, human, background | paved area, roof, wall, etc. | scene-dependent |
| `vegetation` | — | tree, low vegetation | tree, vegetation, grass | tree/vegetation classes |
| `powerline` | power lines | — | — | — |
| `tower` (optional 4th class) | transmission towers | — | — | — |

### Format notes

- Client RGB imagery will most likely arrive as individual geotagged frames (Wingtra-style PPK/RTK direct georeferencing) rather than a pre-built orthomosaic — preserves multi-view redundancy needed for the backprojection stage (see `docs/sota.md` Part B). Avoid segmenting a pre-stitched orthomosaic where avoidable; orthomosaic stitching tends to blur or drop thin wires entirely.
- If a LiDAR channel is present (Wingtra+Hesai style, per Softgrove), expect the same `.laz` format and `core/dataset.py`-style loader conventions already established there — reuse rather than reinvent.
