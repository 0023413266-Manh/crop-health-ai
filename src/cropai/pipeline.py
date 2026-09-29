from .vision.classifier import CropClassifier
from .vision.detector import CropDetector
 
 
class CropPipeline:
    def __init__(self):
        self.classifier = CropClassifier()
        self.detector = CropDetector()
 
    def run(self, image, crop_id=None):
        """
        Phiên bản CHỈ NHẬN DẠNG GIAI ĐOẠN CÂY.
 
        Chặn model bệnh NGAY TỪ NGUỒN (detector.py không chạy model type
        "disease" nữa) thay vì chạy xong rồi lọc theo từ khóa — cách cũ dễ
        bỏ sót nếu tên nhãn không khớp đúng từ khóa trong STAGE_KEYWORDS.
 
        Muốn bật lại bệnh sau này: đổi only_types={"stage"} thành None
        (hoặc {"stage", "disease"}) trong dòng gọi self.detector.predict bên dưới.
        """
        info = None
        if crop_id is None:
            info = self.classifier.predict(image)
            if not info.get("confident", False):
                return {
                    "status": "need_manual_choice",
                    "classifier": info,
                    "message": "Chưa chắc đây là cây gì, hãy chọn loại cây."
                }
            crop_id = info["crop_id"]
 
        if crop_id not in self.detector.registry.configs:
            return {
                "status": "unsupported",
                "classifier": info,
                "crop_id": crop_id,
                "message": f"Chưa hỗ trợ giai đoạn của cây '{crop_id}'."
            }
 
        result = self.detector.predict(image, crop_id, only_types={"stage"})
 
        return {"status": "ok", "classifier": info, "result": result}
