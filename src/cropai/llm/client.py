import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import ollama
 
ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")  # no-op trên Streamlit Cloud (không có file .env), vô hại
 
 
class GeminiClient:
    def __init__(self, gemini_model_name: str = None):
        # gemini-2.5-flash ĐÃ BỊ GOOGLE NGỪNG HỖ TRỢ (retired giữa 2026).
        # Dùng gemini-3.5-flash-lite làm mặc định — 500 lượt/ngày (so với
        # 20 lượt/ngày của bản flash thường), phù hợp hơn cho demo liên tục.
        gemini_model_name = gemini_model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(gemini_model_name)
        else:
            self.gemini_model = None
 
    def generate(self, prompt: str) -> str:
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except ResourceExhausted:
                print("⚠️ Gemini hết quota (Lỗi 429). Đang chuyển sang tư vấn dự phòng...")
            except Exception as e:
                print(f"⚠️ Lỗi kết nối Cloud ({str(e)[:80]}). Đang chuyển sang tư vấn dự phòng...")
 
        # Fallback: thử Ollama CHỈ KHI chạy local (có biến môi trường đánh
        # dấu rõ ràng). Trên Streamlit Cloud, Ollama không bao giờ chạy
        # được (local server, không tồn tại trên server Cloud) — cố gọi
        # chỉ tổ tốn thời gian chờ rồi vẫn lỗi, nên bỏ qua thẳng sang
        # thông báo dự phòng thay vì để app "treo" rồi báo lỗi khó hiểu.
        if os.getenv("RUN_ENV", "cloud") == "local":
            try:
                res = ollama.chat(
                    model='qwen2.5:1.5b',
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return res['message']['content']
            except Exception as local_err:
                print(f"⚠️ Ollama local cũng lỗi: {local_err}")
 
        return (
            "Không thể kết nối dịch vụ tư vấn AI lúc này (hết quota hoặc "
            "lỗi kết nối). Đây là khuyến nghị chung: theo dõi ruộng định kỳ, "
            "tham khảo cán bộ khuyến nông địa phương nếu phát hiện dấu hiệu "
            "bất thường."
        )
 
    def start_chat(self, system_context: str = ""):
        if not self.gemini_model:
            raise RuntimeError("Chưa cấu hình Gemini (thiếu GEMINI_API_KEY), không dùng được chat.")
        history = []
        if system_context:
            history = [
                {"role": "user", "parts": [system_context]},
                {"role": "model", "parts": ["Đã hiểu, tôi sẵn sàng trả lời thêm."]},
            ]
        return self.gemini_model.start_chat(history=history)
 
 
CropAIClient = GeminiClient
