import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.pipeline import CropPipeline

out = CropPipeline().run(sys.argv[1])
out.get("result", {}).pop("detections", None)   # ẩn bớt cho dễ đọc
print(json.dumps(out, ensure_ascii=False, indent=2))