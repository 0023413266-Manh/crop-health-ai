import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import ollama

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


class GeminiClient:
    def __init__(self, gemini_model_name: str = None):
        gemini_model_name = gemini_model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(gemini_model_name)
        else:
            self.gemini_model = None

        self._last_error = None

    def generate(self, prompt: str) -> str:
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except ResourceExhausted as e:
                self._last_error = f"Hết quota (429): {e}"
                print(f"⚠️ Gemini hết quota (Lỗi 429). {e}")
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                print(f"⚠️ Lỗi kết nối Cloud ({str(e)[:200]}).")

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
            "Không thể kết nối dịch vụ tư vấn AI lúc này. "
            f"[DEBUG: {self._last_error or 'không rõ'}] "
            "Khuyến nghị chung: theo dõi ruộng định kỳ, tham khảo cán bộ khuyến nông..."
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