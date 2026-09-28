import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


class GeminiClient:
    def __init__(self, model=None):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Chưa có GEMINI_API_KEY trong file .env")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def generate(self, prompt, retries=4):
        for i in range(retries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt)
                return resp.text
            except Exception as e:
                msg = str(e)
                if ("503" in msg or "429" in msg) and i < retries - 1:
                    time.sleep(3 * (i + 1))   # chờ 3s, 6s, 9s rồi thử lại
                    continue
                raise