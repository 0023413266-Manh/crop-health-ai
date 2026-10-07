from .vision.classifier import CropClassifier
from .vision.detector import CropDetector
from .vision.image_quality import check_image_quality


class CropPipeline:
    def __init__(self):
        self.classifier = CropClassifier()
        self.detector = CropDetector()

    def run(self, image, crop_id=None, skip_quality_check=False):
        # 1. Kiểm tra chất lượng ảnh
        if not skip_quality_check:
            quality = check_image_quality(image)

            if not quality["ok"]:
                warnings = quality.get("warnings", [])

                if isinstance(warnings, list):
                    warning_text = " ".join(str(w) for w in warnings)
                else:
                    warning_text = str(warnings)

                return {
                    "status": "poor_quality",
                    "quality": quality,
                    "message": (
                        "Ảnh chưa đạt chất lượng tốt để nhận diện chính xác: "
                        + warning_text
                    ),
                }

        info = None

        # 2. Nếu chưa có crop_id thì tự động nhận diện cây bằng Tier 1
        if crop_id is None:
            info = self.classifier.predict(image)

            # Lấy Top 1 của Tier 1
            top1 = info.get("top_k", [None])[0]

            if top1 is None:
                return {
                    "status": "classification_failed",
                    "classifier": info,
                    "message": "Không thể xác định loại cây trong ảnh.",
                }

            # Lấy crop_id của Top 1
            crop_id = top1.get("crop_id")

            if not crop_id:
                return {
                    "status": "classification_failed",
                    "classifier": info,
                    "message": "Không xác định được loại cây từ kết quả nhận diện.",
                }

            # Bổ sung tên tiếng Việt cho kết quả Top 1
            cfg = self.detector.registry.configs.get(crop_id)

            if cfg:
                top1["name_vi"] = cfg["name_vi"]

        # 3. Kiểm tra loại cây có model Tier 2 hay không
        if crop_id not in self.detector.registry.configs:
            return {
                "status": "unsupported",
                "classifier": info,
                "crop_id": crop_id,
                "message": f"Chưa hỗ trợ nhận diện cho loài cây '{crop_id}'.",
            }

        # 4. Chạy Tier 2 cho loại cây đã được xác định
        result = self.detector.predict(image, crop_id)

        return {
            "status": "ok",
            "classifier": info,
            "crop_id": crop_id,
            "result": result,
        }