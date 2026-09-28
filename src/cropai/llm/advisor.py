from pathlib import Path
from .client import GeminiClient

ROOT = Path(__file__).resolve().parents[3]


class CropAdvisor:
    def __init__(self, client=None):
        self.client = client or GeminiClient()

    def advise(self, result, knowledge_file=None):
        """result: kết quả của CropDetector.predict()"""
        knowledge = ""
        if knowledge_file:
            p = ROOT / knowledge_file
            if p.exists():
                knowledge = p.read_text(encoding="utf-8")

        stage = result["stage"]["name_vi"] if result["stage"] else "không xác định được"
        diseases = ", ".join(
            f"{d['name_vi']} (độ tin cậy {d['max_conf']})" for d in result["diseases"]
        ) or "không phát hiện"

        prompt = f"""Bạn là chuyên gia nông nghiệp tư vấn cho nông dân Việt Nam.
Hệ thống nhận diện ảnh đã cho kết quả sau (đây là sự thật, không được thay đổi):
- Loại cây: {result['crop_name_vi']}
- Giai đoạn sinh trưởng: {stage}
- Bệnh phát hiện: {diseases}

Tài liệu tham khảo (ưu tiên dùng, không bịa thêm số liệu):
{knowledge or '(không có tài liệu)'}

Hãy viết lời tư vấn ngắn gọn bằng tiếng Việt, gồm:
1. Nhận định tình trạng cây hiện tại
2. Cách chăm sóc trong giai đoạn này (nước, phân bón)
3. Sâu bệnh cần theo dõi và cách phòng
4. Việc cần chuẩn bị cho giai đoạn kế tiếp
Nếu giai đoạn "không xác định được", hãy nói rõ và khuyên người dùng chụp lại ảnh rõ hơn."""
        return self.client.generate(prompt)