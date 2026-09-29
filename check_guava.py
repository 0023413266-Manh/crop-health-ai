from pathlib import Path
import yaml
from ultralytics import YOLO

# 1. Đọc config guava.yaml
yaml_path = Path("configs/crops/guava.yaml")
with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

weights_path = cfg["models"][0]["weights"]
yaml_classes = list(cfg["models"][0]["classes"].keys())

print("==================================================")
print(f"📁 Đường dẫn weights trong YAML : {weights_path}")
print(f"📋 Danh sách Class trong YAML    : {yaml_classes}")
print("--------------------------------------------------")

# 2. Kiểm tra file weights thực tế
w_file = Path(weights_path)
if not w_file.exists():
    print(f"❌ KHÔNG TÌM THẤY FILE WEIGHTS TẠI: {weights_path}")
else:
    size_mb = w_file.stat().st_size / (1024 * 1024)
    print(f"📦 Dung lượng file weights      : {size_mb:.2f} MB")
    
    # 3. Load thử model và đọc class names thực tế từ file .pt
    model = YOLO(weights_path)
    pt_names = list(model.names.values())
    print(f"🏷️ Class thực tế trong file .pt : {pt_names}")
    print("--------------------------------------------------")

    # 4. Đối soát khớp class
    has_error = False
    for name in pt_names:
        if name in yaml_classes:
            print(f"   ✅ Class '{name}': KHỚP!")
        else:
            print(f"   ❌ Class '{name}': KHÔNG TÌM THẤY TRONG YAML!")
            has_error = True

    if has_error:
        print("\n=> NGUYÊN NHÂN: Tên class trong file weights .pt khác với guava.yaml nên detector tự động xóa hết kết quả!")
    else:
        print("\n=> Cấu hình Class hoàn toàn trùng khớp 100%!")
print("==================================================")