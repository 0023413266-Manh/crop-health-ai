from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]


class CropClassifier:
    """Tầng 1: nhìn ảnh, đoán đây là cây gì."""

    def __init__(self, config_path=None):
        path = Path(config_path) if config_path else ROOT / "configs" / "crop_classifier.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.model = YOLO(str(ROOT / self.cfg["weights"]))

    def predict(self, image, k=3):
        r = self.model.predict(image, imgsz=self.cfg["imgsz"], verbose=False)[0]
        raw = r.names[r.probs.top1]
        conf = float(r.probs.top1conf)
        crop_id = self.cfg["classes"].get(raw)

        # Lấy top-k từ toàn bộ xác suất, không chỉ top1
        probs = r.probs.data.tolist()
        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
        top_k = []
        for idx in ranked:
            raw_i = r.names[idx]
            top_k.append({
                "raw_class": raw_i,
                "crop_id": self.cfg["classes"].get(raw_i),
                "conf": round(probs[idx], 3),
            })

        return {
            "crop_id": crop_id,
            "raw_class": raw,
            "conf": round(conf, 3),
            "confident": conf >= self.cfg["min_conf"],
            "top_k": top_k,
        }