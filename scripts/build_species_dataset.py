"""
build_species_dataset.py — tải toàn bộ project Roboflow của bạn (10 project,
7 giống cây) và gộp thành 1 dataset DUY NHẤT cho TẦNG 1 (nhận diện giống
cây trồng) — mỗi cây 1 class, bất kể project gốc có bao nhiêu class con
(giai đoạn) bên trong.

Cách dùng: chạy trong Kaggle/Colab, đã cài sẵn `pip install roboflow`.

    python build_species_dataset.py

Kết quả: thư mục species_dataset/ với cấu trúc chuẩn YOLO (train/valid/test
+ data.yaml, nc=7), sẵn sàng train model tầng 1.
"""
import shutil
from pathlib import Path
import os
KEY_A = os.getenv("ROBOFLOW_KEY_A")   # workspace new-workspace-kf0gd
KEY_B = os.getenv("ROBOFLOW_KEY_B")   # workspace 9-nhom
KEY_C = os.getenv("ROBOFLOW_KEY_C")   # workspace phm-phc-mnh-s-workspace
RAW_DIR = Path("raw_species_source")
OUT_DIR = Path("species_dataset")

SPECIES_SOURCES = {
    "lua": {
        "class_id": 0,
        "projects": [
            ("KEY_C", "phm-phc-mnh-s-workspace", "luatro", 2),
            ("KEY_C", "phm-phc-mnh-s-workspace", "lua-sinh-truong", 1),
            ("KEY_C", "phm-phc-mnh-s-workspace", "lua-chin", 1),
        ],
    },
    "xoai": {
        "class_id": 1,
        "projects": [
            ("KEY_C", "phm-phc-mnh-s-workspace", "xoai-sinh-truong", 2),
            ("KEY_C", "phm-phc-mnh-s-workspace", "xoai-ra-bong", 2),
            ("KEY_C", "phm-phc-mnh-s-workspace", "xoai-phat-trien-trai", 3),
        ],
    },
    "cam": {
        "class_id": 2,
        "projects": [
            ("KEY_A", "new-workspace-kf0gd", "cay-cam-new", 2),
        ],
    },
    "ot": {
        "class_id": 3,
        "projects": [
            ("KEY_B", "9-nhom", "cay-ot-akufo", 10),
        ],
    },
    "sau_rieng": {
        "class_id": 4,
        "projects": [
            ("KEY_B", "9-nhom", "sau-rieng-g8goj", 4),
        ],
    },
    "mit": {
        "class_id": 5,
        "projects": [
            ("KEY_B", "9-nhom", "cay-mit", 2),
        ],
    },
}

SPLITS = ["train", "valid", "test"]


def download_all():
    from roboflow import Roboflow

    _rf_clients: dict[str, Roboflow] = {}

    for species, info in SPECIES_SOURCES.items():
        for i, (api_key, workspace, project_id, version_n) in enumerate(info["projects"]):
            if api_key not in _rf_clients:
                _rf_clients[api_key] = Roboflow(api_key=api_key)
            rf = _rf_clients[api_key]

            dest = RAW_DIR / f"{species}__{i}__{project_id}"
            if dest.exists():
                print(f"[bỏ qua] Đã tải trước đó: {dest}")
                continue

            print(f"Đang tải {workspace}/{project_id}/v{version_n} -> {dest} ...")
            project = rf.workspace(workspace).project(project_id)
            project.version(version_n).download("yolov11", location=str(dest))


def read_yolo_label(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            boxes.append(tuple(values))
        else:
            xs, ys = values[0::2], values[1::2]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            boxes.append(((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin))
    return boxes


def build_merged_dataset():
    counters = {"images": 0, "boxes": 0}

    for species, info in SPECIES_SOURCES.items():
        class_id = info["class_id"]
        for i in range(len(info["projects"])):
            src_dir = next(RAW_DIR.glob(f"{species}__{i}__*"), None)
            if src_dir is None:
                print(f"[CẢNH BÁO] Không tìm thấy dữ liệu đã tải cho {species} (nguồn thứ {i})")
                continue

            for split in SPLITS:
                img_dir = src_dir / split / "images"
                lbl_dir = src_dir / split / "labels"
                if not img_dir.exists():
                    continue

                out_img_dir = OUT_DIR / split / "images"
                out_lbl_dir = OUT_DIR / split / "labels"
                out_img_dir.mkdir(parents=True, exist_ok=True)
                out_lbl_dir.mkdir(parents=True, exist_ok=True)

                for img_path in img_dir.glob("*.*"):
                    new_stem = f"{species}_{i}__{img_path.stem}"
                    shutil.copy2(img_path, out_img_dir / f"{new_stem}{img_path.suffix}")
                    counters["images"] += 1

                    boxes = read_yolo_label(lbl_dir / f"{img_path.stem}.txt")
                    lines = [f"{class_id} {cx} {cy} {w} {h}" for cx, cy, w, h in boxes]
                    (out_lbl_dir / f"{new_stem}.txt").write_text("\n".join(lines))
                    counters["boxes"] += len(boxes)

    names = [None] * len(SPECIES_SOURCES)
    for species, info in SPECIES_SOURCES.items():
        names[info["class_id"]] = species

    data_yaml = f"""train: {OUT_DIR.resolve()}/train/images
val: {OUT_DIR.resolve()}/valid/images
test: {OUT_DIR.resolve()}/test/images

nc: {len(names)}
names: {names}
"""
    (OUT_DIR / "data.yaml").write_text(data_yaml)

    print(f"\n=== Hoàn tất ===")
    print(f"Tổng ảnh: {counters['images']}")
    print(f"Tổng box: {counters['boxes']}")
    print(f"Class order: {names}")
    print(f"data.yaml: {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    download_all()
    build_merged_dataset()
