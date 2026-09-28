import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.vision.detector import CropDetector
from cropai.vision.visualize import draw_boxes

ap = argparse.ArgumentParser()
ap.add_argument("--crop", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="result.jpg")
a = ap.parse_args()

result = CropDetector().predict(a.image, a.crop)
draw_boxes(a.image, result["detections"], a.out)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("Đã lưu ảnh kết quả:", a.out)