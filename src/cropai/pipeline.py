from .vision.classifier import CropClassifier
from .vision.detector import CropDetector


class CropPipeline:
    def __init__(self):
        self.classifier = CropClassifier()
        self.detector = CropDetector()

    def run(self, image, crop_id=None):
        """crop_id=None: tự nhận diện cây. crop_id='orange': người dùng chọn tay."""
        info = None
        if crop_id is None:
            info = self.classifier.predict(image)
            if not info["confident"]:
                return {"status": "need_manual_choice", "classifier": info,
                        "message": "Chưa chắc đây là cây gì, hãy chọn loại cây."}
            crop_id = info["crop_id"]

        if crop_id not in self.detector.registry.configs:
            return {"status": "unsupported", "classifier": info, "crop_id": crop_id,
                    "message": f"Chưa hỗ trợ giai đoạn của cây '{crop_id}'."}

        result = self.detector.predict(image, crop_id)
        return {"status": "ok", "classifier": info, "result": result}