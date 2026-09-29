from pathlib import Path
import yaml
from ultralytics import YOLO

# 2 Ebenen nach oben: registry.py -> cropai -> src -> crop-health-ai
ROOT = Path(__file__).resolve().parents[2]


class CropRegistry:
    """Liest configs/crops/*.yaml und lädt Modelle mit Caching."""

    def __init__(self, config_dir=None):
        self.config_dir = Path(config_dir) if config_dir else ROOT / "configs" / "crops"
        self.configs = {}
        self._models = {}
        for f in sorted(self.config_dir.glob("*.yaml")):
            cfg = yaml.safe_load(f.read_text(encoding="utf-8"))
            self.configs[cfg["id"]] = cfg

    def list_crops(self):
        return [{"id": c["id"], "name_vi": c["name_vi"]} for c in self.configs.values()]

    def get_config(self, crop_id):
        if crop_id not in self.configs:
            raise ValueError(f"Pflanze '{crop_id}' wird nicht unterstützt. Verfügbar: {list(self.configs)}")
        return self.configs[crop_id]

    def get_model(self, crop_id, model_id):
        key = (crop_id, model_id)
        if key not in self._models:
            cfg = self.get_config(crop_id)
            m = next(m for m in cfg["models"] if m["id"] == model_id)
            self._models[key] = YOLO(str(ROOT / m["weights"]))
        return self._models[key]