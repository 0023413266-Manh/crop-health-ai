from PIL import Image, ImageDraw


def draw_boxes(image_path, detections, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        color = "red" if d["model_type"] == "disease" else "lime"
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 4, y1 + 4), f"{d['key']} {d['conf']:.2f}", fill=color)
    img.save(out_path)