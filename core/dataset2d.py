"""2D image/mask dataset loader with merged-taxonomy remapping.

VEPL and DDOS (docs/datasets.md) already label power lines and vegetation
together and are the primary sources; TTPLA/UAVid/etc. remain useful as a
scale/diversity supplement. Each source dataset ships its own label ids;
SOURCE_LABEL_MAPS remaps every source into the shared taxonomy in classes.json
({background, vegetation, powerline, tower}). Add an entry here per new source
dataset rather than special-casing loading code.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core import CLASS_NAMES, IGNORE_INDEX

_BG, _VEG, _PL, _TOWER = (CLASS_NAMES.index(n) for n in ["background", "vegetation", "powerline", "tower"])

# Per-source-dataset remap: {source_raw_id: unified_class_id}. IGNORE_INDEX drops a
# source class entirely (e.g. UAVid's "car"/"human" are irrelevant clutter here).
SOURCE_LABEL_MAPS: dict[str, dict[int, int]] = {
    "vepl": {
        # VEPL (Zenodo 7800234) ships {vegetation, powerline, background} masks --
        # exact raw pixel ids TBD from the downloaded mask files, verify before use.
        0: _BG,
        1: _VEG,
        2: _PL,
    },
    "ddos": {
        # DDOS (huggingface.co/datasets/benediktkol/DDOS) semantic classes, in the
        # order documented by the dataset card: Animals, Vehicles, Buildings, Trees,
        # Large Mesh, Small Mesh, Thin Structures, Ultra-thin, Other, Background.
        # Raw ids assumed 0-indexed in that order -- verify against the actual
        # downloaded label files before training (dataset card / loader script).
        0: IGNORE_INDEX,  # animals
        1: IGNORE_INDEX,  # vehicles
        2: _BG,           # buildings
        3: _VEG,          # trees
        4: _BG,           # large mesh (fences etc.)
        5: _BG,           # small mesh
        6: _PL,           # thin structures (wires, poles)
        7: _PL,           # ultra-thin (finest wires)
        8: IGNORE_INDEX,  # other
        9: _BG,           # background
    },
    "ttpla": {
        0: _BG,
        1: _TOWER,
        2: _PL,
    },
    "uavid": {
        0: _BG,       # building
        1: _BG,       # road
        2: _VEG,      # tree
        3: _VEG,      # low vegetation
        4: IGNORE_INDEX,  # static car
        5: IGNORE_INDEX,  # moving car
        6: IGNORE_INDEX,  # human
        7: _BG,       # background clutter
    },
    "semantic_drone": {
        # Semantic Drone Dataset (TU Graz) grayscale label ids, per the official
        # class_dict.csv row order -- an empty dict here previously would have
        # crashed on load (see SegmentationDataset.__init__); ids below are best
        # recollection and MUST be verified against the actual downloaded
        # class_dict.csv before training.
        0: _BG,   # unlabeled
        1: _BG,   # paved-area
        2: _BG,   # dirt
        3: _VEG,  # grass
        4: _BG,   # gravel
        5: _BG,   # water
        6: _BG,   # rocks
        7: _BG,   # pool
        8: _VEG,  # vegetation
        9: _BG,   # roof
        10: _BG,  # wall
        11: _BG,  # window
        12: _BG,  # door
        13: _BG,  # fence
        14: _BG,  # fence-pole
        15: IGNORE_INDEX,  # person
        16: IGNORE_INDEX,  # dog
        17: IGNORE_INDEX,  # car
        18: IGNORE_INDEX,  # bicycle
        19: _VEG,  # tree
        20: _VEG,  # bald-tree
        21: IGNORE_INDEX,  # ar-marker
        22: _BG,   # obstacle
        23: IGNORE_INDEX,  # conflicting
    },
    "client": {
        # Client-labeled masks are expected to already use the unified taxonomy
        # in classes.json (identity mapping) -- see finetune2d.py.
        _BG: _BG,
        _VEG: _VEG,
        _PL: _PL,
        _TOWER: _TOWER,
    },
}


class SegmentationDataset(Dataset):
    """Expects `root/images/*.png` (or .jpg) and `root/masks/*.png` (paletted/id masks),
    with matching filenames, plus a `source` string selecting the remap table above.
    """

    def __init__(self, root: str, source: str, image_size: tuple[int, int] = (1024, 1024),
                 augment: bool = False):
        self.root = Path(root)
        self.image_paths = sorted((self.root / "images").glob("*.*"))
        self.mask_dir = self.root / "masks"
        self.image_size = image_size
        self.augment = augment
        self.label_map = SOURCE_LABEL_MAPS.get(source, {})
        if not self.label_map:
            raise ValueError(f"No label remap registered for source '{source}' in SOURCE_LABEL_MAPS")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _remap_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        remapped = np.full_like(raw_mask, fill_value=IGNORE_INDEX, dtype=np.int64)
        for src_id, dst_id in self.label_map.items():
            remapped[raw_mask == src_id] = dst_id
        return remapped

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{img_path.stem}.png"

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
        raw_mask = cv2.resize(raw_mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)

        mask = self._remap_mask(raw_mask)

        if self.augment:
            image, mask = _augment(image, mask)

        image_t = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).long()
        return {"image": image_t, "mask": mask_t}


def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random flip + rotation. Deliberately conservative -- power lines are
    orientation-sensitive (near-straight, roughly horizontal in most nadir/oblique
    corridor shots), so avoid aggressive rotation/shear that would erase that prior.
    """
    if np.random.rand() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    if np.random.rand() < 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    return image, mask
