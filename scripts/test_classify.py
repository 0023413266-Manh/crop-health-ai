import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.vision.classifier import CropClassifier

print(json.dumps(CropClassifier().predict(sys.argv[1]), ensure_ascii=False, indent=2))