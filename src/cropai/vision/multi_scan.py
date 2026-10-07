"""
Quét nhiều loài khi Tầng 1 (classify) không chắc chắn.
Tận dụng lại các model Tầng 2 đã có sẵn để xác nhận loài nào thực sự
xuất hiện trong ảnh, thay vì chỉ dựa vào top-3 xác suất của Tầng 1.
"""


def scan_candidates(detector, image, top_k_candidates):
    """
    detector: instance của CropDetector (đã có sẵn, dùng lại registry + model Tầng 2)
    image: đường dẫn ảnh gốc
    top_k_candidates: list các dict từ top_k của Tầng 1, mỗi item có 'crop_id'

    Trả về: list các ứng viên, mỗi ứng viên gồm thông tin loài + kết quả quét thật
    bằng model Tầng 2 của chính loài đó trên ảnh gốc.
    """
    results = []

    for cand in top_k_candidates:
        crop_id = cand.get("crop_id")
        if not crop_id or crop_id not in detector.registry.configs:
            continue  # loài này chưa có model Tầng 2, bỏ qua

        try:
            scan_result = detector.predict(image, crop_id)
        except Exception as e:
            scan_result = None

        # Tính độ tin cậy cao nhất trong các khung tìm được (nếu có)
        best_conf = 0.0
        num_boxes = 0
        if scan_result and scan_result.get("detections"):
            confs = [d["conf"] for d in scan_result["detections"]]
            if confs:
                best_conf = max(confs)
                num_boxes = len(confs)

        results.append({
            "crop_id": crop_id,
            "name_vi": cand.get("name_vi", crop_id),
            "classifier_conf": cand.get("conf", 0.0),   # độ tin cậy từ Tầng 1
            "scan_best_conf": round(best_conf, 3),       # độ tin cậy cao nhất khi quét thật bằng Tầng 2
            "scan_num_boxes": num_boxes,                 # có bao nhiêu khung được tìm thấy
            "scan_result": scan_result,                  # kết quả đầy đủ, dùng để hiển thị nếu người dùng chọn
        })

    # Sắp theo độ tin cậy quét thật (scan_best_conf) giảm dần — đây là tín hiệu
    # đáng tin hơn so với conf của Tầng 1 thuần classify, vì nó dựa trên việc
    # THỰC SỰ tìm thấy vật thể đặc trưng của loài đó trong ảnh.
    results.sort(key=lambda r: r["scan_best_conf"], reverse=True)
    return results