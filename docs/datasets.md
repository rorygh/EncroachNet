# Datasets: RGB Powerline + Vegetation Segmentation

Survey of publicly available datasets, organized by role in the training pipeline. See `docs/sota.md` for the corresponding model survey.

Last updated: August 2026

---

## Correction: The Combined Dataset Gap Is Smaller Than Initially Assessed

The first pass of this survey concluded that no dataset labels power lines and vegetation together. That was wrong — a second, more targeted search turned up two datasets that directly close this gap, at different scales and for different purposes:

- **[VEPL](https://zenodo.org/records/7800234)** — a small, real, public, purpose-built dataset: drone orthomosaics of an actual power-line corridor in Colombia, labeled exactly `{vegetation, powerline, background}`, **and it ships DSMs alongside the orthomosaics** — i.e. real labeled 2D data with real paired 3D geometry, no photogrammetry run required to test the orthomosaic+DSM lift path.
- **[DDOS](https://huggingface.co/datasets/benediktkol/DDOS)** — a large synthetic (AirSim) dataset whose "small suburban town" environment explicitly includes "dense trees and numerous power lines" in the same scenes, with full RGB + depth + per-pixel segmentation ground truth for every frame of 300 training / 20 val / 20 test **10 Hz flight sequences**.

Both are detailed below. The older powerline-only (TTPLA, InsPLAD) and vegetation-only (UAVid, Semantic Drone, VDD) datasets are still useful for scale and diversity once VEPL's ~2.4 km / single-geography footprint is outgrown, but they're no longer the only option — see **Data Strategy** below for how all of these fit together.

---

## ★ Combined Powerline + Vegetation Datasets

### VEPL — Primary Real-World Target-Taxonomy Dataset ★
**Papers:** dataset paper ([ResearchGate](https://www.researchgate.net/publication/372931199_VEPL_Dataset_A_Vegetation_Encroachment_in_Power_Line_Corridors_Dataset_for_Semantic_Segmentation_of_Drone_Aerial_Orthomosaics)); companion model paper, [VEPL-Net](https://www.mdpi.com/2220-9964/12/11/454) (*ISPRS Int. J. Geo-Information*, 2023)
**Data:** https://doi.org/10.5281/zenodo.7800234 — CC BY 4.0, publicly downloadable

| Property | Value |
|---|---|
| Classes | Exactly EncroachNet's target taxonomy minus `tower`: `vegetation`, `powerline`, `background` |
| Coverage | ~2.4 km of drone flights along a secondary road, Envigado, Colombia |
| Contents | 4 full-size orthomosaics + masks **+ DSMs** (digital surface models), plus tessellated chunk sets: 532 base image/mask pairs, 3,724 with geometric augmentation, 3,192 with spectral augmentation |

This is the dataset EncroachNet's whole taxonomy was designed around, and it exists. Two important caveats: (1) it's built from **stitched orthomosaics, not raw source frames** — no per-image camera poses are provided, so it can validate the orthomosaic+DSM 2D→3D lift path directly (`docs/architecture.md` Stage 2) but not the multi-view backprojection path, since there's no multi-view redundancy left after stitching; (2) single geography / ~2.4 km — real signal, but not enough scale or diversity on its own to be the whole training set. Use it as the primary fine-tuning/validation set in the already-correct taxonomy, supplemented by the larger single-class datasets below for pretraining scale.

### DDOS — Synthetic Multi-View Flight Sequences with Depth ★
**Paper:** Kolbeinsson, *"DDOS: The Drone Depth and Obstacle Segmentation Dataset"* ([arXiv 2312.12494](https://arxiv.org/abs/2312.12494), CVPRW 2024)
**Data:** https://huggingface.co/datasets/benediktkol/DDOS — publicly downloadable

| Property | Value |
|---|---|
| Generation | Synthetic, AirSim (Unreal Engine) |
| Environments | "Small suburban town" (**dense trees and numerous power lines**, residential) and a park setting (football field, floodlights, dense trees, office buildings) |
| Sequences | 300 train / 20 val / 20 test **flights**, each a continuous 10 Hz, 10-second trajectory (~100 frames/flight) |
| Per-frame data | RGB, depth map, pixel-wise semantic segmentation, optical flow, surface normals |
| Classes | 10: Animals, Vehicles, Buildings, **Trees**, Large Mesh, Small Mesh, **Thin Structures**, **Ultra-thin**, Other, Background |

This directly answers "is there a simulated drone dataset we could run ODM on": yes — DDOS's flights are genuine sequential multi-view trajectories (not single stills), so its RGB frames can be fed straight into ODM to reconstruct camera poses + geometry, **and then checked against DDOS's own ground-truth depth/pose** as a correctness test for the whole Stage 2→3 pipeline before ever touching real data. It also directly supplies the `{vegetation≈Trees, powerline≈Thin Structures/Ultra-thin, background}` remap needed for 2D pretraining, at far larger scale than VEPL (30k+ frames vs. ~500 base VEPL chunks), and unlike a from-scratch synthetic renderer, it's already built and public.

**Why this replaces building a from-scratch synthetic corridor renderer:** DDOS already covers the case a bespoke Blender/SynthBlend-style corridor generator would have been built for (towers/wires + trees in a controllable synthetic scene, with GT labels) — no need to build and maintain that ourselves.

---

## RGB-D / Depth-Equipped Powerline Datasets (Real)

Useful specifically for validating `core/backproject.py`'s visibility-check math against real sensor depth, without running photogrammetry at all.

### TL-RGBD — Component/Defect Focus, Real Depth Sensor
Real data captured by **China Southern Power Grid** UAVs with synchronized RGB + depth sensors.

| Property | Value |
|---|---|
| Image pairs | 10,000 paired RGB + depth |
| Instances | 73,448 annotated, across 9 component/defect states (insulators, tie wires, poles, flashover, bird's nests, etc.) |
| Object size | 94.5% of instances are small objects (<32² px) |

Component/defect-level (like InsPLAD), not corridor/vegetation segmentation — relevant here mainly as a real RGB-D validation source, not primary training data. Access/download details not confirmed publicly; check the citing paper ([arXiv 2602.01696](https://arxiv.org/html/2602.01696)) for a data-availability statement.

### APSD (AirSim Power System Dataset) — Synthetic RGB-D, Access Unconfirmed
Chao et al., Fujian Institute of Research on the Structure of Matter (CAS), published in *Pattern Recognition*, June 2026. Referenced by Rory: [CAS news writeup](https://english.cas.cn/newsroom/research-news/202606/t20260604_1161124.shtml).

| Property | Value |
|---|---|
| Generation | Synthetic, AirSim |
| Image pairs | 4,000+ RGB-D |
| Classes | Power lines, power poles, street lights, traffic lights |
| Scenes | Multiple simulated urban and industrial environments |

Potentially useful (RGB-D + explicit powerline class), but two open questions before relying on it: (1) the class mix (poles/streetlights/traffic lights alongside power lines) suggests this may be a lower-altitude urban-infrastructure simulation rather than a high-voltage transmission-corridor flight — worth confirming the camera platform/altitude before assuming it transfers to Wingtra-style corridor imagery; (2) no public download link found — the paper states the dataset was built for their M3WaveGNet RGB-D segmentation framework, so access likely requires contacting the lead author (jchao@fjirsm.ac.cn) or checking the *Pattern Recognition* paper's data-availability statement directly.

---

## Stage 1: Powerline-Specific 2D Datasets (Scale/Diversity Supplement)

### TTPLA — Primary Powerline/Tower Source ★
**Paper:** Abdelfattah et al., *"TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines"* ([arXiv 2010.10032](https://arxiv.org/abs/2010.10032))
**Code/data:** https://github.com/R3ab/ttpla_dataset

| Property | Value |
|---|---|
| Images | 1,234 @ 3,840×2,160 (905/109/220 train/val/test) |
| Instances | 8,987 manually labeled (towers + power lines) |
| Labels | Supports detection, semantic segmentation, and instance segmentation |
| Classes | 5 raw: `cable`, `tower_lattice`, `tower_wooden`, `tower_tucohy` (tubular/concrete/hybrid poles), `void` (ambiguous, excluded from eval/loss) — no subdivision of cable into conductor vs. static wire |
| Format | Polygon annotations (LabelMe-style JSON / Supervisely format), **not raster masks** — rasterize before use (`scripts/prepare_ttpla.py`) |
| License | Apache 2.0 — permissive, no noncommercial restriction (unlike DDOS) |
| Depth/3D | **None.** Single 2D images with polygon (LabelMe) instance masks only — no depth maps, no multi-view overlap, no camera poses. Not usable for photogrammetry/ODM; not usable to validate the backprojection stage. |

The largest public dataset with pixel-level power-line labels at reasonable scale, but 2D-only. No vegetation class — merge with a vegetation-labeled dataset (or use VEPL/DDOS directly, above) under a unified taxonomy (see Data Strategy). Official Google Drive download frequently hits Google's per-file quota ("too many users have viewed or downloaded this file"); `scripts/download_datasets.sh` uses the [Dataset Ninja Supervisely-format mirror](https://datasetninja.com/ttpla) instead, which isn't gated.

**Benchmark results on TTPLA** (cable-only binary segmentation is the de facto standard eval — most follow-on papers drop the tower classes and instance-segmentation task entirely; see `docs/sota.md` Part A for the full comparison table and per-paper architecture notes). Notably, DUFormer's comparison table reports **SegFormer-B2 at 71.59% IoU** on TTPLA cable segmentation; `configs/default.json` initially trained that exact backbone, then switched to HRNetV2 (`hrnet_w32`) specifically to compare against it — see the HRNet-OCR row in the same table, and `docs/sota.md`'s takeaway on why.

**Native resolution matters for training too, not just inference**: TTPLA ships full 3,840×2,160 frames, not pre-tessellated tiles like VEPL. `core/dataset2d.py`'s `SegmentationDataset` used to resize the whole frame down to `image_size` (512×512) for every source — for TTPLA that meant squashing a 16:9 frame into a 1:1 square while shrinking already-thin wires ~7.5× before the model ever saw them. Fixed to crop natively (random crop while training, deterministic center crop for validation) whenever the source image is at least as large as `image_size`, only falling back to resize-up for sources smaller than the target (VEPL's 256×256 tessellated tiles). This also keeps training scale consistent with `core/inference3d.py`'s tiled inference, which runs the model over native-scale windows.

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

## Data Strategy (Recommended Multi-Stage Pipeline)

No from-scratch synthetic data generation (no in-house Blender/SynthBlend-style corridor renderer) — **DDOS already covers that role** with public, pre-built synthetic flights containing both target classes plus ground-truth depth. Real work goes into the stages below instead:

```
Stage 1 — Synthetic pretraining + pipeline validation
  → DDOS (public, AirSim): {Trees -> vegetation, Thin Structures/Ultra-thin -> powerline,
    everything else -> background}. Large scale (30k+ frames), sequential multi-view
    flights -- also the dataset to validate core/backproject.py and the ODM path against,
    since GT depth/segmentation is known for every frame.

Stage 2 — Real combined-taxonomy fine-tuning
  → VEPL (public, real): already in the exact target taxonomy. Small (~2.4 km,
    single geography) but zero label-remapping needed, and the DSM lets the
    orthomosaic+DSM lift path (docs/architecture.md Stage 2) be tested on real data.

Stage 3 — Merged-taxonomy scale/diversity supplement (optional, if Stage 1+2 underperform)
  → TTPLA (powerline/tower classes) + UAVid / Semantic Drone / VDD (vegetation
    classes), remapped to the shared label space (see table below)

Stage 4 — Domain adaptation to client corridor imagery
  → Fine-tune on real client drone RGB once available, following the same
    sparse-label fine-tuning precedent already validated for Softgrove
    (finetune.py; TreeLite3D's "1 annotated tree" result — see Softgrove's
    docs/sota.md, Part A, TreeLite3D section)
```

### Label-space merging notes

| Unified class | DDOS | VEPL | TTPLA | UAVid | Semantic Drone | VDD |
|---|---|---|---|---|---|---|
| `background` | Animals, Vehicles, Buildings, Large/Small Mesh, Other, Background | background | (implicit) | building, road, car, human, background | paved area, roof, wall, etc. | scene-dependent |
| `vegetation` | Trees | vegetation | — | tree, low vegetation | tree, vegetation, grass | tree/vegetation classes |
| `powerline` | Thin Structures, Ultra-thin | powerline | power lines | — | — | — |
| `tower` (optional 4th class) | — | — | transmission towers | — | — | — |

### Format notes

- Client RGB imagery will most likely arrive as individual geotagged frames (Wingtra-style PPK/RTK direct georeferencing) rather than a pre-built orthomosaic — preserves multi-view redundancy needed for the backprojection stage (see `docs/sota.md` Part B). Avoid segmenting a pre-stitched orthomosaic where avoidable; orthomosaic stitching tends to blur or drop thin wires entirely.
- If a LiDAR channel is present (Wingtra+Hesai style, per Softgrove), expect the same `.laz` format and `core/dataset.py`-style loader conventions already established there — reuse rather than reinvent.
