#!/usr/bin/env bash
# =============================================================================
# EncroachNet — dataset download script
#
# Run this ONLY on the RunPod pod after mounting the network volume at
# /workspace/. Datasets are large; do not run locally.
#
# Usage:
#   bash scripts/download_datasets.sh [--datasets <list>] [--data-root <path>]
#
# Examples:
#   bash scripts/download_datasets.sh                          # download all
#   bash scripts/download_datasets.sh --datasets powerline     # TTPLA/InsPLAD only
#   bash scripts/download_datasets.sh --datasets TTPLA,UAVid
#
# Dataset groups (see docs/datasets.md for the full survey):
#   combined   — VEPL, DDOS                     (primary: powerline + vegetation together)
#   powerline  — TTPLA, InsPLAD                 (powerline/tower classes, supplement)
#   vegetation — UAVid, SemanticDrone, VDD       (vegetation classes, supplement)
#   lidar      — DALES                           (3D-side powerline validation)
#   all        — all of the above (default)
#
# Individual dataset names (comma-separated):
#   VEPL, DDOS, TTPLA, InsPLAD, UAVid, SemanticDrone, VDD, DALES
#
# Environment variables:
#   DATA_ROOT   Override data directory (default: /workspace/data)
# =============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
DATASETS="all"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$DATA_ROOT"
echo "============================================================"
echo "  EncroachNet dataset download"
echo "  DATA_ROOT = $DATA_ROOT"
echo "  DATASETS  = $DATASETS"
echo "============================================================"
echo

# ── Helpers ──────────────────────────────────────────────────────────────────

want() {
    local name="$1"
    [[ "$DATASETS" == "all" ]] && return 0
    [[ "$DATASETS" == *"$name"* ]] && return 0
    return 1
}

zenodo_dl() {
    local record_id="$1"
    local dest="$2"
    echo "  → Zenodo record $record_id → $dest"
    mkdir -p "$dest"
    pip install -q zenodo_get 2>/dev/null || true
    zenodo_get "$record_id" -o "$dest"
}

# ── Combined powerline + vegetation datasets (primary) ───────────────────────

if want combined || want VEPL; then
    echo "[VEPL] Real drone corridor imagery, exact target taxonomy + DSMs — Zenodo 7800234"
    zenodo_dl 7800234 "$DATA_ROOT/VEPL"
    echo
fi

if want combined || want DDOS; then
    echo "[DDOS] Synthetic AirSim flights, trees + power lines + GT depth — HuggingFace"
    mkdir -p "$DATA_ROOT/DDOS"
    pip install -q "huggingface_hub[cli]" 2>/dev/null || true
    huggingface-cli download benediktkol/DDOS --repo-type dataset --local-dir "$DATA_ROOT/DDOS" || \
        echo "  ⚠  huggingface-cli failed — see https://huggingface.co/datasets/benediktkol/DDOS for manual download."
    echo
fi

# ── Powerline datasets (scale/diversity supplement) ───────────────────────────

if want powerline || want TTPLA; then
    echo "[TTPLA] 1,234 images (905/109/220 train/val/test), Apache-2.0 — cable + 3 tower types"
    echo "  Official source: https://github.com/R3ab/ttpla_dataset (Google Drive link in README --"
    echo "  frequently hits Google's per-file quota: 'too many users have viewed or downloaded this file')."
    echo "  Using the Dataset Ninja Supervisely-format mirror instead (same data, polygon annotations,"
    echo "  not gated): https://datasetninja.com/ttpla"
    mkdir -p "$DATA_ROOT/TTPLA"
    curl -sL "https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMTUyNl9UVFBMQS90dHBsYS1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJ1R3BKNWhoQWN2bUJta1RnM3Z0MlZ4Y3JlOUwvWEoyaFMvNVdCZkFaQjJVPSJ9?response-content-disposition=attachment%3B%20filename%3D%22ttpla-DatasetNinja.tar%22" \
        -o "$DATA_ROOT/TTPLA/ttpla-DatasetNinja.tar" \
        && tar -xf "$DATA_ROOT/TTPLA/ttpla-DatasetNinja.tar" -C "$DATA_ROOT/TTPLA" \
        || echo "  ⚠  Mirror fetch failed -- signed URL may have expired; get a fresh one from the" \
                "'Download' button at https://datasetninja.com/ttpla and re-run, or fall back to the" \
                "GitHub README's Google Drive link once its quota resets."
    echo "  TTPLA ships polygon annotations, not raster masks -- rasterize before training:"
    echo "    python scripts/prepare_ttpla.py --raw $DATA_ROOT/TTPLA --out $DATA_ROOT/encroachnet/ttpla"
    echo
fi

if want powerline || want InsPLAD; then
    echo "[InsPLAD] 10,607 UAV images, 17 asset classes + 6 defect types"
    echo "  Source: https://github.com/andreluizbvs/InsPLAD"
    mkdir -p "$DATA_ROOT/InsPLAD"
    echo "  ⚠  Manual step — see repo README for the Mendeley Data / release download link."
    echo "     Component/defect-level only, not primary training data — see docs/datasets.md."
    echo
fi

# ── Vegetation / general aerial datasets ──────────────────────────────────────

if want vegetation || want UAVid; then
    echo "[UAVid] 420 images, 8 classes incl. tree/low vegetation — urban UAV scenes"
    echo "  Source: https://uavid.nl/"
    mkdir -p "$DATA_ROOT/UAVid"
    echo "  ⚠  Requires registration at https://uavid.nl/ — download manually,"
    echo "     then place at: $DATA_ROOT/UAVid/"
    echo
fi

if want vegetation || want SemanticDrone; then
    echo "[SemanticDrone] 400 train / 200 test, 22 classes incl. tree/vegetation/grass"
    echo "  Source: https://www.tugraz.at/index.php?id=22387 (ICG, TU Graz)"
    mkdir -p "$DATA_ROOT/SemanticDrone"
    echo "  ⚠  Manual step — see project page for current download link."
    echo "     Place extracted images/masks at: $DATA_ROOT/SemanticDrone/"
    echo
fi

if want vegetation || want VDD; then
    echo "[VDD] Varied Drone Dataset for semantic segmentation"
    echo "  Paper: https://arxiv.org/abs/2305.13608"
    mkdir -p "$DATA_ROOT/VDD"
    echo "  ⚠  Manual step — see paper for current dataset release link."
    echo
fi

# ── LiDAR (3D-side powerline validation, only if a LiDAR channel is in scope) ──

if want lidar || want DALES; then
    echo "[DALES] ~500M pts aerial LiDAR — includes explicit power-lines/poles class"
    echo "  Zenodo mirror (unofficial): https://zenodo.org/records/4694695"
    zenodo_dl 4694695 "$DATA_ROOT/DALES"
    echo
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Download complete (or manual-step instructions printed above)."
echo "Data layout:"
ls -lh "$DATA_ROOT" 2>/dev/null || true
echo
echo "After manual downloads + label remapping (core/dataset2d.py SOURCE_LABEL_MAPS),"
echo "set DATA_ROOT in your training command:"
echo "  DATA_ROOT=$DATA_ROOT python train2d.py --config configs/default.json --sources ttpla,uavid"
echo "============================================================"
