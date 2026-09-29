import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.llm.advisor import extract_stage_section

text = (ROOT / "knowledge/orange.md").read_text(encoding="utf-8")
ket_qua = extract_stage_section(text, "Phát triển quả")
print("--- KẾT QUẢ CẮT ---")
print(ket_qua)
print("--- ĐỘ DÀI:", len(ket_qua), "so với file gốc:", len(text))