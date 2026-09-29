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
        gemini_model_name = gemini_model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(gemini_model_name)
        else:
            self.gemini_model = None

    def generate(self, prompt: str) -> str:
        # 1. Ưu tiên gọi Gemini Cloud
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except ResourceExhausted:
                print("⚠️ Gemini hết quota (Lỗi 429). Đang chuyển sang AI Local (Ollama)...")
            except Exception as e:
                print(f"⚠️ Lỗi kết nối Cloud ({str(e)[:40]}). Đang chuyển sang AI Local...")

        # 2. Fallback tự động sang Ollama Local trên ổ D
        try:
            res = ollama.chat(
                model='qwen2.5:1.5b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            return res['message']['content']
        except Exception as local_err:
            raise RuntimeError(f"Lỗi AI Local Ollama: {local_err}")
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

# Định nghĩa thêm alias để tránh vỡ code ở các module khác
CropAIClient = GeminiClient