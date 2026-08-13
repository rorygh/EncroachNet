import json
from pathlib import Path

_CLASSES_PATH = Path(__file__).parent.parent / "classes.json"
with open(_CLASSES_PATH) as f:
    _REGISTRY = json.load(f)

IGNORE_INDEX: int = _REGISTRY["ignore_index"]
CLASS_NAMES: list[str] = _REGISTRY["semantic"]["class_names"]
CLASS_COLORS: list[str] = _REGISTRY["semantic"]["class_colors"]
NUM_CLASSES: int = _REGISTRY["semantic"]["num_classes"]
