from collections import defaultdict
 
try:
    from ..registry import CropRegistry
except (ImportError, ValueError):
    from cropai.registry import CropRegistry
 
 
class CropDetector:
    def __init__(self, registry=None):
        self.registry = registry or CropRegistry()
 
    def predict(self, image, crop_id, only_types=None):
        """
        only_types: set các model_type được phép chạy, vd {"stage"} để CHỈ
        chạy model giai đoạn, bỏ hẳn model bệnh — không tốn thời gian chạy
        model bệnh rồi lọc bỏ sau như trước, và không có rủi ro lọt sót vì
        so khớp từ khóa không khớp hết nhãn thật.
 
        None (mặc định) = chạy TẤT CẢ model đã đăng ký cho cây này, giữ
        nguyên hành vi cũ.
        """
        cfg = self.registry.get_config(crop_id)
        detections = []
 
        for m in cfg["models"]:
            if only_types is not None and m["type"] not in only_types:
                continue
 
            model = self.registry.get_model(crop_id, m["id"])
            res = model.predict(image, imgsz=m["imgsz"], conf=m["conf"], verbose=False)[0]
 
            for box in res.boxes:
                cls_id = int(box.cls[0]) if box.cls.ndim > 0 else int(box.cls.item())
                raw = res.names[cls_id]
 
                info = m["classes"].get(raw)
                if info is None:
                    continue
 
                conf_val = float(box.conf[0]) if box.conf.ndim > 0 else float(box.conf.item())
 
                detections.append({
                    "model_type": m["type"],
                    "raw_class": raw,
                    "key": info["key"],
                    "name_vi": info["name_vi"],
                    "conf": round(conf_val, 3),
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
 