import cv2
import numpy as np


def check_image_quality(image_path, blur_threshold=100.0, dark_threshold=40, bright_threshold=220):
    """Kiểm tra ảnh có đủ chất lượng để đưa vào model không.

    Trả về dict: {"ok": bool, "warnings": [...]}
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"ok": False, "warnings": ["Không đọc được file ảnh."]}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warnings = []

    # 1. Kiểm tra độ mờ (variance of Laplacian)
    # Ảnh càng sắc nét, phương sai càng cao. Ảnh mờ thường dưới 100.
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_threshold:
        warnings.append(f"Ảnh có vẻ bị mờ (độ nét: {blur_score:.0f}, khuyến nghị >= {blur_threshold:.0f}).")

    # 2. Kiểm tra độ sáng trung bình
    brightness = gray.mean()
    if brightness < dark_threshold:
        warnings.append(f"Ảnh khá tối (độ sáng trung bình: {brightness:.0f}/255).")
    elif brightness > bright_threshold:
        warnings.append(f"Ảnh bị cháy sáng / lóa (độ sáng trung bình: {brightness:.0f}/255).")

    return {
        "ok": len(warnings) == 0,
        "warnings": warnings,
        "blur_score": round(blur_score, 1),
        "brightness": round(float(brightness), 1),
    }