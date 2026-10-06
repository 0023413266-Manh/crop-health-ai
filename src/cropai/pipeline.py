from .vision.classifier import CropClassifier
from .vision.detector import CropDetector
from .vision.image_quality import check_image_quality


class CropPipeline:
    def __init__(self):
        self.classifier = CropClassifier()
        self.detector = CropDetector()

    def run(self, image, crop_id=None, skip_quality_check=False):
        if not skip_quality_check:
            quality = check_image_quality(image)
            if not quality["ok"]:
                return {
                    "status": "poor_quality",
                    "quality": quality,
                    "message": "Ảnh chưa đạt chất lượng tốt để nhận diện chính xác: "
                               + " ".join(quality["warnings"]),
                }

        info = None
        if crop_id is None:
            info = self.classifier.predict(image)
            if not info["confident"]:
                # gắn thêm tên tiếng Việt cho từng lựa chọn trong top_k
                for item in info["top_k"]:
                    cfg = self.detector.registry.configs.get(item["crop_id"])
                    item["name_vi"] = cfg["name_vi"] if cfg else item["crop_id"]
                return {"status": "need_manual_choice", "classifier": info,
                        "message": "Chưa chắc đây là cây gì, hãy chọn đúng loại cây."}
            crop_id = info["crop_id"]

        if crop_id not in self.detector.registry.configs:
            return {"status": "unsupported", "classifier": info, "crop_id": crop_id,
                    "message": f"Chưa hỗ trợ giai đoạn của cây '{crop_id}'."}

        result = self.detector.predict(image, crop_id)
        return {"status": "ok", "classifier": info, "result": result}