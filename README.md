# Crop Health AI

Nhận diện loài cây và giai đoạn sinh trưởng bằng YOLO, tư vấn chăm sóc bằng Gemini.

## Luồng xử lý
Ảnh → YOLO-cls (loài cây) → YOLO-detect (giai đoạn / sâu bệnh) → Gemini + knowledge → tư vấn

## Cài đặt
pip install -r requirements.txt
copy .env.example .env   (rồi điền GEMINI_API_KEY)

## Weights
File .pt không nằm trong repo. Tải tại: <https://drive.google.com/drive/folders/1tpphSSILdvjSEiA2xEaWBOKDulDYb12h?usp=drive_link>
Đặt vào thư mục weights/ đúng cấu trúc trong configs/crops/*.yaml.

## Chạy
streamlit run ui/streamlit_app.py