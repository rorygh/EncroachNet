# State of the Art: RGB Powerline + Vegetation Segmentation, Backprojected to 3D

Survey of the relevant literature for EncroachNet, in three parts: (A) 2D segmentation models for the powerline/vegetation classes, (B) 2D→3D backprojection methods, (C) 3D post-processing for encroachment risk.

Last updated: August 2026

---

## Part A: 2D Semantic Segmentation

The vegetation class is not the hard part of this problem — any competent aerial segmentation backbone handles trees/low-vegetation well, and mature datasets (UAVid, Semantic Drone) exist for it. **The powerline class is the hard part**: wires are 1–5 px wide, occupy roughly 1–5% of image pixels, have low texture, and are easily confused with background clutter (utility poles, fence lines, contrails, tree branches). Every architectural choice below is driven by that asymmetry.

### General aerial-segmentation backbones

| Model | Type | Notes |
|---|---|---|
| SegFormer | MiT transformer encoder + lightweight MLP decoder | Scales B0 (real-time) → B5 (accuracy); strong reported results on UAVid |
| Mask2Former | Swin backbone + masked-attention transformer decoder | Universal (semantic/instance/panoptic) segmentation; a working Mask2Former/Semantic-SAM codebase is already vendored locally at `C:\rory\scripts\search3d\...\semantic_sam` — usable as a starting point instead of building from scratch |
| UNetFormer / BANet | CNN-transformer hybrids purpose-built for remote-sensing scenes | BANet reports 64.6% mIoU on UAVid |
| **HRNet / HRNetV2** | Maintains a full-resolution branch in parallel with downsampled branches through the *entire* network, rather than downsample-then-upsample | Literature specifically credits this design with better edge/tiny-object preservation **without any post-processing** — the standout candidate for the powerline class given its structural resemblance to other thin-object problems (pose estimation keypoints, pipe-corrosion segmentation) |

### Domain-specific powerline detectors and losses

- **[TTPLA](https://arxiv.org/abs/2010.10032)** (Abdelfattah et al., 2020) — the founding dataset paper; establishes transmission-tower + power-line instance segmentation as a benchmark task; baseline models evaluated for detection, semantic, and instance segmentation.
- **[A Novel Focal Phi Loss for Power Line Segmentation with Auxiliary Classifier U-Net](https://www.mdpi.com/1424-8220/21/8/2803)** (Jaffari et al., *Sensors* 2021) — power lines are 1–5% of pixels vs. 95–99% background; proposes a focal loss generalized from the Matthews Correlation Coefficient (Phi coefficient), paired with an auxiliary-classifier U-Net (ACU-Net) head for faster convergence. Reports **+16% Dice, +19% precision** over plain balanced binary cross-entropy. Directly analogous to the focal-loss-for-rare-class solution already used in Softgrove for tree-top heatmap detection (`docs/architecture.md` there) — same imbalance problem, same loss family, different domain.
- **[PLGAN: Generative Adversarial Networks for Power-Line Segmentation in Aerial Images](https://arxiv.org/pdf/2204.07243)** (IEEE TIP 2023) — GAN-based feature embedding plus a **novel loss defined in Hough-transform parameter space**, exploiting the fact that power lines are locally near-straight and therefore concentrate into compact clusters in Hough space; claims SOTA on power-line segmentation and line detection over prior methods.
- **[Axial-UNet++ Power Line Detection Network Based on Gated Axial Attention Mechanism](https://doi.org/10.3390/rs16234585)** (*Remote Sensing* 2024) — addresses class imbalance, small sample counts, and long-range dependency (wires span the full image, need a receptive field that reaches across it) via gated axial attention in a UNet++ backbone.
- **[Advanced YOLO-based Real-time Power Line Detection for Vegetation Management](https://arxiv.org/abs/2503.00044)** (Rong et al., 2025, submitted to IEEE Trans. Power Delivery) — **the closest published system to EncroachNet's actual end goal.** YOLOv8 + directional filters (extracts directional texture/features of wires) generating Oriented Bounding Boxes, followed by a post-processing algorithm that computes a **quantitative vegetation-encroachment metric**. Detection/bbox-level rather than pixel-precise segmentation — EncroachNet's pixel-level 2D→3D approach is a natural precision upgrade on this exact pipeline shape.
- **[Deep Learning in Automated Power Line Inspection: A Review](https://arxiv.org/abs/2502.07826)** (Faisal et al., 2025) — broad survey of component detection + fault diagnosis methods; useful as an ongoing reference as the project's scope grows toward asset/defect inspection (InsPLAD-style tasks) beyond pure encroachment.

### Topology-preserving losses (transferred from tubular/curvilinear-structure segmentation)

Power lines, like blood vessels or roads, are curvilinear structures where per-pixel IoU/Dice is a poor training signal — a segmentation can score well on IoU while having broken connectivity (gaps in the wire), which is exactly the failure mode that matters for downstream 3D catenary fitting.

- **clDice / soft-clDice** (Shit et al.) — loss computed on the intersection between predicted masks and their morphological *skeleton*, explicitly rewarding connectivity rather than raw overlap. Established as state-of-the-art for tubular-structure segmentation across both 2D and 3D, architecture-independent (validated on both U-Net and FCN). Known limitation: significant computational overhead, especially for multi-class training.
- **[Skeleton Recall Loss](https://arxiv.org/html/2404.03010v1)** (2024) — a newer, cheaper alternative reported to match or exceed clDice's topological accuracy at much lower compute cost. The more practical default for EncroachNet given the added cost of also running a full 3D backprojection pipeline downstream.

### Foundation-model caveat

Segment Anything (SAM) is a powerful zero-shot segmenter (SA-1B, 1.1B+ masks) but is **documented to miss thin wire/cable structures** in zero-shot use. Not usable as a zero-shot base for the powerline class without fine-tuning; still potentially useful for the (easy) vegetation class, or as a human-assisted labeling aid when building the client training set.

### Recommendation

Baseline two backbones — **SegFormer** (fast, proven on UAVid) and **HRNetV2** (thin-structure-preserving by design) — trained with a combined loss: class-weighted focal/Dice term (Focal Phi Loss family) for the pixel-count imbalance, plus a Skeleton Recall or soft-clDice term specifically on the powerline class for connectivity. Compare wire recall *and* connectivity metrics, not just mIoU, before picking a winner — see `docs/datasets.md` for why mIoU alone is misleading here.

---

## Part B: 2D → 3D Backprojection

- **[2D3DNet](https://arxiv.org/pdf/2110.11325)** ("Learning 3D Semantic Segmentation with only 2D Image Supervision") — the closest published blueprint for EncroachNet's overall pipeline shape. Three stages: (1) a 2D model pretrained on labeled images segments every frame of an unlabeled RGB+LiDAR sequence, (2) **multi-view fusion**: for each 3D point, backproject into every camera view, keep only views where the point is actually visible (depth/geometry consistency check), and vote/aggregate labels across all visible views to assign a best-guess 3D label, (3) the resulting dense-but-noisy pseudo-labeled point cloud is used as training input/target for a **sparse 3D convolutional network**, which produces the final per-point prediction. Stage 3 is architecturally identical to the spconv sparse U-Net already implemented in Softgrove and SimpleUNet — directly reusable rather than novel.
- **3DMV** (Dai & Nießner, CVPR 2018 lineage) — projects and voxel-max-pools 2D CNN *features* (not just final labels) before fusing with 3D geometric features. Richer signal than label-only voting, more compute; worth an ablation once the label-voting baseline works.
- **[Virtual Multi-view Fusion for 3D Semantic Segmentation](https://arxiv.org/pdf/2007.13138)** (Google, ECCV 2020) — renders synthetic camera views of the *reconstructed* mesh/point cloud itself rather than relying solely on original capture poses, producing denser and better-conditioned viewing angles before 2D segmentation and backprojection. Relevant if raw UAV image coverage of a wire span from favorable angles is sparse (e.g. nadir-only mapping flights rarely see wires side-on).
- Softgrove's own SOTA survey (`C:\rory\scripts\Softgrove\docs\sota.md`, Part B4 "2D-Guided 3D") already covers the general paradigm this project extends: Open3DIS, SAMPro3D, SegDINO3D all use the same visibility-check-then-fuse recipe for lifting 2D foundation-model outputs into 3D. This confirms the backprojection step is a mature, low-risk technique for EncroachNet rather than something that needs inventing.
- **Direct 3D-native alternative** (when a co-registered LiDAR channel is available): recent work extracts power lines straight from point clouds without any 2D step — KPConv-based classification, and an **[elevation-aware multi-resolution network](https://doi.org/10.3390/rs17193318)** (*Remote Sensing* 2025) purpose-built for powerline-corridor point-cloud segmentation. Notably, a 2025 paper (["a segmentation method for LiDAR point clouds of aerial slender targets"](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1548786/full)) explicitly validates the RGB-guides-LiDAR idea for this exact target class: thin aerial wires have such a small reflective cross-section that LiDAR alone struggles, and using 2D image semantics to guide point-cloud segmentation measurably helps. This is direct literature support for EncroachNet's core premise (2D segmentation informing 3D wire/vegetation labeling) even in the LiDAR-available case.
- **Tooling correction (RGB-only path)**: the original candidate solution proposed reusing the raw-COLMAP pipeline at `C:\rory\scripts\LogMotion`, but that pipeline has only been exercised on video/ground-level scenes. **[OpenDroneMap (ODM)](https://github.com/OpenDroneMap/ODM)** is purpose-built for multi-image drone-corridor capture instead (OpenSfM poses → OpenMVS dense reconstruction/depth maps → orthophoto/DSM) and directly exposes both artifacts the backprojection stage needs (`shots.geojson` poses, per-image depth maps) — see `docs/architecture.md` Stage 2.

### Recommendation

Implement the 2D3DNet-style three-stage pattern: per-point visibility-checked multi-view label voting, with an optional lightweight spconv refinement pass reusing the existing Softgrove/SimpleUNet backbone pattern. Where LiDAR is available, prefer backprojecting onto the LiDAR point cloud directly (denser, more accurate geometry than photogrammetric MVS, which struggles badly on thin wires — see Part C).

---

## Part C: 3D Post-Processing for Encroachment Risk

### Catenary curve fitting

Conductors sag between towers along a catenary (hyperbolic cosine) curve: `y = a·cosh(x/a) + c`. Both photogrammetric MVS and LiDAR undersample thin wires badly (small cross-section, motion blur, low reflectivity for LiDAR at glancing incidence), so the raw backprojected/LiDAR wire points are sparse and noisy relative to vegetation or ground points. The established fix, well precedented in the transmission-line-monitoring literature:

1. Cluster wire points per span / per individual conductor (e.g. connected-component or RANSAC line-grouping in the plane perpendicular to the corridor direction).
2. Fit the catenary model per conductor by nonlinear least-squares (iterative Newton-Raphson on the catenary parameters).
3. Use the fitted curve — not the raw points — as the conductor's 3D centerline for downstream clearance computation. Reported fitting residuals in the literature are **sub-10 cm**, i.e. this recovers a clean centerline even from a sparse, patchy point sample.

### Clearance / risk computation

Given a fitted conductor centerline and labeled vegetation points (or per-tree-crown clusters, reusing Softgrove's own instance-clustering machinery if a LiDAR channel with tree structure is present), compute the nearest point-to-curve distance per vegetation point, and threshold against the utility's regulatory **Minimum Vegetation Clearance Distance (MVCD)**. This is standard industry practice for LiDAR-based vegetation management programs — annual (≤18-month) inspection cycles with MVCD compliance reporting are the norm across the sources surveyed (YellowScan, GIM International, Hornbill Technology industry write-ups).

### Recommendation

Treat catenary fitting as a required step, not optional polish — it is what makes the pipeline robust to the wire-undersampling problem inherent to both MVS and LiDAR. Output a georeferenced point cloud with per-point class + a separate flagged risk-zone layer (points/regions below MVCD), matching how existing LiDAR vegetation-management tools report results.

---

## Key Takeaways for EncroachNet

1. **Vegetation segmentation is solved; powerline segmentation is the research risk.** Budget accuracy/architecture effort accordingly — HRNetV2 + a topology-aware thin-wire loss (Focal Phi / Skeleton Recall) is the most literature-supported starting point.
2. **The 2D3DNet three-stage pattern (2D predict → multi-view backproject/vote → optional 3D refine) is a proven blueprint**, not something to design from scratch, and its refinement stage reuses infrastructure Rory already has (spconv sparse U-Net from Softgrove/SimpleUNet).
3. **Never trust raw backprojected wire points directly** — always fit a catenary model before computing clearance; this is the standard, literature-validated fix for wire undersampling in both MVS and LiDAR.
4. **[Advanced YOLO-based Real-time Power Line Detection for Vegetation Management](https://arxiv.org/abs/2503.00044)** is the single closest existing published system to EncroachNet's stated goal and is worth re-reading in full once implementation starts — it's operating at bbox precision where EncroachNet aims for pixel + 3D precision, so it's a good sanity baseline, not a ceiling.
