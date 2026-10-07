from pathlib import Path
from .client import GeminiClient

ROOT = Path(__file__).resolve().parents[3]


def extract_stage_section(knowledge_text: str, stage_name_vi: str) -> str:
    """Chỉ lấy phần markdown ứng với đúng giai đoạn (dòng bắt đầu bằng '## ')."""
    if not knowledge_text or not stage_name_vi:
        return knowledge_text
    lines = knowledge_text.splitlines()
    section, capturing = [], False
    for line in lines:
        if line.strip().startswith("## "):
            capturing = stage_name_vi.lower() in line.lower()
            if capturing:
                section.append(line)
            continue
        if capturing:
            section.append(line)
    return "\n".join(section).strip() or knowledge_text


class CropAdvisor:
    def __init__(self, client=None):
        self.client = client or GeminiClient()

    def _build_context(self, result, knowledge_file=None):
        knowledge = ""
        if knowledge_file:
            p = ROOT / knowledge_file
            if p.exists():
                full_text = p.read_text(encoding="utf-8")
                stage_name = result["stage"]["name_vi"] if result["stage"] else None
                knowledge = extract_stage_section(full_text, stage_name)

        stage = result["stage"]["name_vi"] if result["stage"] else "không xác định được"
        diseases = ", ".join(
            f"{d['name_vi']} (độ tin cậy {d['max_conf']})" for d in result["diseases"]
        ) or "không phát hiện"

        return f"""Bạn là chuyên gia nông nghiệp tư vấn cho nông dân Việt Nam.
Hệ thống nhận diện ảnh đã cho kết quả sau (đây là sự thật, không được thay đổi):
- Loại cây: {result['crop_name_vi']}
- Giai đoạn sinh trưởng: {stage}
- Bệnh phát hiện: {diseases}

Tài liệu tham khảo cho ĐÚNG giai đoạn này (ưu tiên dùng, không bịa thêm số liệu):
{knowledge or '(không có tài liệu riêng cho giai đoạn này)'}

Từ giờ, người dùng có thể hỏi thêm. Luôn trả lời dựa trên thông tin trên và tài liệu tham khảo, bằng tiếng Việt, ngắn gọn, dễ hiểu cho nông dân."""

    def advise(self, result, knowledge_file=None):
        """Tư vấn 1 lần, không giữ hội thoại."""
        context = self._build_context(result, knowledge_file)
        prompt = context + "\n\nHãy viết lời tư vấn ban đầu, gồm: (1) nhận định tình trạng, (2) cách chăm sóc giai đoạn này, (3) sâu bệnh cần theo dõi, (4) chuẩn bị cho giai đoạn kế tiếp."
        return self.client.generate(prompt)

    def start_session(self, result, knowledge_file=None):
        """Tạo 'phiên hỏi đáp' tự quản lý lịch sử, có fallback Ollama nếu Gemini lỗi."""
        context = self._build_context(result, knowledge_file)
        first_question = ("Hãy viết lời tư vấn, với yêu cầu: KHÔNG lặp lại thông tin hiển nhiên mà người trồng tự nhìn ảnh đã biết (ví dụ không cần mô tả lại 'đây là quả ổi xanh')."
                        "Thay vào đó, tập trung vào:"
                        "(1) Đây có phải THỜI ĐIỂM QUAN TRỌNG cần hành động ngay không (ví dụ: sắp tới hạn bón phân, đây là giai đoạn dễ bị sâu bệnh tấn công nhất trong cả vòng đời, sắp đến lúc phải ngừng phun thuốc để đảm bảo an toàn thu hoạch)."
                        "(2) Rủi ro CỤ THỂ dễ bị bỏ qua ở giai đoạn này mà người mới trồng hay mắc phải."
                        "(3) Một việc CẦN LÀM TRONG TUẦN TỚI, không phải lời khuyên chung chung theo kiểu sách giáo khoa.")

        session = ChatSession(self.client, context)
        first = session.ask(first_question)
        return session, first

    def resume_session(self, context: str, history: list):
        """
        Dựng lại ChatSession từ dữ liệu đã lưu trong DB.
        Không gọi lại Gemini, chỉ khôi phục context và history.
        history: list of (role, text) tuples như đã lưu trong history_json.
        Trả về ChatSession đã có sẵn lịch sử.
        """
        session = ChatSession(self.client, context)
        # Khôi phục lịch sử dạng list of [role, text] hoặc (role, text)
        session.history = [tuple(item) for item in history]
        return session


class ChatSession:
    """Giữ lịch sử hội thoại thủ công, mỗi câu hỏi gọi qua GeminiClient.generate()
    (tự có fallback Ollama sẵn trong generate())."""

    def __init__(self, client, context):
        self.client = client
        self.context = context
        self.history = []   # list of (role, text)

    def ask(self, question):
        history_text = "\n".join(f"{r}: {t}" for r, t in self.history)
        prompt = f"{self.context}\n\nLịch sử hội thoại:\n{history_text}\n\nCâu hỏi mới: {question}\nTrả lời:"
        answer = self.client.generate(prompt)
        self.history.append(("Người dùng", question))
        self.history.append(("Trợ lý", answer))
        return answer