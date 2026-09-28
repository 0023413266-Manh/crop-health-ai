import sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from PIL import Image
from cropai.pipeline import CropPipeline
from cropai.llm.advisor import CropAdvisor
from cropai.vision.visualize import draw_boxes

st.set_page_config(page_title="Tư vấn cây trồng", page_icon="🌱", layout="wide")


@st.cache_resource
def load_pipeline():
    return CropPipeline()


pipe = load_pipeline()
crops = pipe.detector.registry.list_crops()
names = {c["name_vi"]: c["id"] for c in crops}

st.title("🌱 Nhận diện và tư vấn sinh trưởng cây trồng")

with st.sidebar:
    st.header("Tùy chọn")
    mode = st.radio("Cách chọn loại cây", ["Tự động nhận diện", "Chọn thủ công"])
    manual_crop = None
    if mode == "Chọn thủ công":
        manual_crop = names[st.selectbox("Loại cây", list(names))]
    use_llm = st.checkbox("Tư vấn bằng Gemini", value=True)

file = st.file_uploader("Tải ảnh cây lên", type=["jpg", "jpeg", "png"])

if file:
    suffix = Path(file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        img_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(img_path), caption="Ảnh gốc", use_container_width=True)

    with st.spinner("Đang nhận diện..."):
        out = pipe.run(img_path, crop_id=manual_crop)

    if out["status"] == "need_manual_choice":
        st.warning(out["message"] + " Hãy chuyển sang 'Chọn thủ công' ở thanh bên.")
        st.json(out["classifier"])
    elif out["status"] == "unsupported":
        st.info(out["message"])
    else:
        res = out["result"]
        if out["classifier"]:
            c = out["classifier"]
            st.caption(f"Tự nhận diện: {res['crop_name_vi']} (độ tin cậy {c['conf']}). "
                       f"Sai? Chuyển sang 'Chọn thủ công'.")
        out_img = str(Path(tempfile.gettempdir()) / "cropai_result.jpg")
        draw_boxes(img_path, res["detections"], out_img)
        with col2:
            st.image(Image.open(out_img), caption="Kết quả nhận diện", use_container_width=True)

        st.subheader(f"{res['crop_name_vi']}")
        st.write("**Giai đoạn:**", res["stage"]["name_vi"] if res["stage"] else "Không xác định được")
        if res["diseases"]:
            st.write("**Bệnh phát hiện:**", ", ".join(d["name_vi"] for d in res["diseases"]))

        if use_llm:
            kfile = pipe.detector.registry.get_config(res["crop"]).get("knowledge_file")
            with st.spinner("Gemini đang soạn tư vấn..."):
                try:
                    st.markdown(CropAdvisor().advise(res, kfile))
                except Exception as e:
                    st.error(f"Không gọi được Gemini: {str(e)[:200]}")