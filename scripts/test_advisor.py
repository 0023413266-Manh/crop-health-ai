import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.pipeline import CropPipeline
from cropai.llm.advisor import CropAdvisor

pipe = CropPipeline()
out = pipe.run(sys.argv[1])
print("Trạng thái:", out["status"])

if out["status"] == "ok":
    res = out["result"]
    print("Cây:", res["crop_name_vi"], "| Giai đoạn:", res["stage"])
    kfile = pipe.detector.registry.get_config(res["crop"]).get("knowledge_file")
    print("\n--- TƯ VẤN ---")
    try:
        print(CropAdvisor().advise(res, kfile))
    except Exception as e:
        print("LỖI GEMINI:", type(e).__name__)
        print(str(e)[:600])
else:
    print(out["message"])