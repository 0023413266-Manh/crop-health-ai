import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cropai.pipeline import CropPipeline

out = CropPipeline().run(sys.argv[1], skip_quality_check=True)
print("status:", out["status"])
print("message:", out.get("message", "(không có)"))

if out["status"] == "need_manual_choice":
    for c in out["candidates"]:
        print(f"  {c['name_vi']:12} | Tầng1_conf={c['classifier_conf']:.2f} "
              f"| quét_thật_conf={c['scan_best_conf']:.2f} | số khung={c['scan_num_boxes']}")
elif out["status"] == "ok":
    res = out["result"]
    print("Loài:", res["crop_name_vi"])
    print("Giai đoạn:", res["stage"])
elif out["status"] == "poor_quality":
    print("Chất lượng ảnh:", out["quality"])