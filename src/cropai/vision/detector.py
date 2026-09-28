from collections import defaultdict
from .registry import CropRegistry


class CropDetector:
    def __init__(self, registry=None):
        self.registry = registry or CropRegistry()

    def predict(self, image, crop_id):
        cfg = self.registry.get_config(crop_id)
        detections = []

        for m in cfg["models"]:
            model = self.registry.get_model(crop_id, m["id"])
            res = model.predict(image, imgsz=m["imgsz"], conf=m["conf"], verbose=False)[0]
            for box in res.boxes:
                raw = res.names[int(box.cls)]
                info = m["classes"].get(raw)
                if info is None:
                    continue
                detections.append({
                    "model_type": m["type"],          # stage hoặc disease
                    "raw_class": raw,
                    "key": info["key"],
                    "name_vi": info["name_vi"],
                    "conf": round(float(box.conf), 3),
                    "box": [round(v, 1) for v in box.xyxy[0].tolist()],
                })

        return {
            "crop": crop_id,
            "crop_name_vi": cfg["name_vi"],
            "stage": self._dominant(detections, "stage"),
            "diseases": self._summary(detections, "disease"),
            "detections": detections,
        }

    @staticmethod
    def _dominant(dets, mtype):
        """Giai đoạn chiếm ưu thế = key có tổng độ tin cậy cao nhất."""
        score, name = defaultdict(float), {}
        for d in dets:
            if d["model_type"] == mtype:
                score[d["key"]] += d["conf"]
                name[d["key"]] = d["name_vi"]
        if not score:
            return None
        k = max(score, key=score.get)
        return {"key": k, "name_vi": name[k]}

    @staticmethod
    def _summary(dets, mtype):
        agg = {}
        for d in dets:
            if d["model_type"] != mtype:
                continue
            a = agg.setdefault(d["key"], {"key": d["key"], "name_vi": d["name_vi"], "count": 0, "max_conf": 0})
            a["count"] += 1
            a["max_conf"] = max(a["max_conf"], d["conf"])
        return list(agg.values())