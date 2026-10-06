from PIL import Image, ImageDraw, ImageFont


def _load_font(size=18):
    """PIL mặc định không vẽ được dấu tiếng Việt, cần nạp font Unicode thật."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",        # Windows
        "C:/Windows/Fonts/seguiemj.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux / Streamlit Cloud
        "/System/Library/Fonts/Supplemental/Arial.ttf",     # macOS
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes(image_path, detections, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(18)

    for d in detections:
        is_disease = d.get("model_type") == "disease"
        color = "#e53935" if is_disease else "#43a047"   # đỏ = bệnh, xanh = giai đoạn
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{d.get('name_vi', d.get('key', ''))} {d['conf']:.0%}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # nền mờ phía sau chữ để dễ đọc trên mọi màu ảnh
        pad = 4
        draw.rectangle(
            [x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1],
            fill=color,
        )
        draw.text((x1 + pad, y1 - th - pad), label, fill="white", font=font)

    img.save(out_path)