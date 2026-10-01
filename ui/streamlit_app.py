import sys, tempfile, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
from PIL import Image
from cropai.pipeline import CropPipeline
from cropai.llm.advisor import CropAdvisor
from cropai.vision.visualize import draw_boxes

# 1. Cấu hình trang
st.set_page_config(page_title="Tư vấn cây trồng", page_icon="🌱", layout="wide")

# 2. CSS Tùy chỉnh: Phóng to chữ, làm đẹp nút bấm và khung Expander
st.markdown("""
    <style>
    /* Ẩn bớt công cụ mặc định của Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tăng kích thước font chữ toàn bộ App */
    html, body, [class*="css"] {
        font-size: 18px !important;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }

    /* Tiêu đề trang & Subheader */
    h1 {
        font-size: 2.3rem !important;
        color: #1b5e20 !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }
    h2, h3 {
        font-size: 1.5rem !important;
        color: #2e7d32 !important;
        font-weight: 600 !important;
    }

    /* Nút bấm, Radio, Checkbox, Selectbox to rõ hơn */
    div[data-testid="stMarkdownContainer"] p, 
    label[data-baseweb="radio"] span,
    div[data-baseweb="select"] span,
    .stCheckbox label p {
        font-size: 18px !important;
        font-weight: 500 !important;
    }

    .stButton>button {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        background-color: #2e7d32 !important;
        color: white !important;
        border: none !important;
    }

    /* Trang trí khung Expander (Hướng dẫn kỹ thuật) sạch đẹp, chuẩn Card */
    div[data-testid="stExpander"] {
        border: 2px solid #a5d6a7 !important;
        border-radius: 12px !important;
        background-color: #fcfdfc !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        overflow: hidden;
    }

    /* Tiêu đề của Expander */
    div[data-testid="stExpander"] details summary p {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #1b5e20 !important;
    }

    /* Khung tải ảnh to rõ */
    div[data-testid="stFileUploader"] section {
        padding: 20px !important;
        border: 2px dashed #2e7d32 !important;
        border-radius: 12px !important;
        background-color: #f9fbf9 !important;
    }
    </style>
""", unsafe_allow_html=True)

def get_stage_knowledge(crop_id: str, detected_stage: str) -> str:
    """
    Chỉ trích xuất phần cẩm nang thuộc đúng giai đoạn/bệnh phát hiện được.
    """
    md_path = f"knowledge/{crop_id}.md"
    
    if not os.path.exists(md_path):
        return "⚠️ Chưa có dữ liệu cẩm nang cho loại cây này."
        
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Cắt file markdown thành từng phần dựa vào tiêu đề '## '
    sections = content.split('\n## ')
    
    # 1. Tìm phần khớp với detected_stage
    for section in sections:
        lines = section.strip().split('\n')
        section_header = lines[0].strip().lower()
        
        # So sánh tên thẻ ## với stage phát hiện
        if detected_stage.lower() in section_header or section_header in detected_stage.lower():
            return "## " + section
            
    # 2. Nếu không tìm thấy chính xác giai đoạn đó -> Trả về phần Giới thiệu chung (Phần đầu file)
    main_intro = sections[0]
    return f"ℹ️ *Chưa có hướng dẫn riêng cho giai đoạn '{detected_stage}'. Dưới đây là thông tin chung:*\n\n{main_intro}"

def extract_stage_info(md_text, stage_name):
    """Trích xuất đúng đoạn tài liệu thuộc về giai đoạn, giữ nguyên định dạng gạch đầu dòng."""
    if not stage_name:
        return md_text
    
    lines = md_text.split("\n")
    filtered_lines = []
    recording = False
    
    for line in lines:
        # Nhận diện tiêu đề giai đoạn (VD: #, ##, ###)
        if line.strip().startswith("#"):
            if stage_name.lower() in line.lower():
                recording = True
                continue  # Bỏ qua chính dòng tiêu đề trùng lặp này
            elif recording:
                # Gặp tiêu đề giai đoạn tiếp theo thì dừng
                break
        if recording:
            filtered_lines.append(line)
            
    result = "\n".join(filtered_lines).strip()
    
    # Bỏ các đường kẻ phân cách '---' ở cuối nếu có
    result = re.sub(r'\n---\s*$', '', result).strip()
    
    return result if result else md_text

@st.cache_resource
def load_pipeline():
    return CropPipeline()

pipe = load_pipeline()
crops = pipe.detector.registry.list_crops()
names = {c["name_vi"]: c["id"] for c in crops}

st.title("🌱 Nhận diện và tư vấn sinh trưởng cây trồng")

if "advice_cache" not in st.session_state:
    st.session_state.advice_cache = {}

with st.sidebar:
    st.header("⚙️ Tùy chọn")
    mode = st.radio("Cách chọn loại cây", ["Tự động nhận diện", "Chọn thủ công"])
    manual_crop = None
    if mode == "Chọn thủ công":
        manual_crop = names[st.selectbox("Loại cây", list(names))]
    use_llm = st.checkbox("Tư vấn bằng AI (Gemini/Ollama)", value=True)

file = st.file_uploader("Tải ảnh cây trồng lên đây", type=["jpg", "jpeg", "png"])

if file:
    cache_key = f"{file.name}_{file.size}_{manual_crop}"

    suffix = Path(file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        img_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(img_path), caption="Ảnh gốc", use_container_width=True)

    with st.spinner("Đang phân tích hình ảnh..."):
        out = pipe.run(img_path, crop_id=manual_crop)

    if out["status"] == "need_manual_choice":
        st.warning(out["message"] + " Hãy chuyển sang 'Chọn thủ công' ở thanh bên.")
        st.json(out["classifier"])
    elif out["status"] == "unsupported":
        st.info(out["message"])
    else:
        res = out["result"]
        stage_vi = res["stage"]["name_vi"] if res["stage"] else None

        if out["classifier"]:
            c = out["classifier"]
            st.caption(f"Tự nhận diện: {res['crop_name_vi']} (độ tin cậy {c['conf']}). "
                       f"Sai? Chuyển sang 'Chọn thủ công'.")
        out_img = str(Path(tempfile.gettempdir()) / "cropai_result.jpg")
        draw_boxes(img_path, res["detections"], out_img)
        with col2:
            st.image(Image.open(out_img), caption="Kết quả nhận diện", use_container_width=True)

        st.subheader(f"🌾 Cây trồng: {res['crop_name_vi']}")
        st.write("**Giai đoạn phát hiện:**", f"`{stage_vi}`" if stage_vi else "Không xác định được")
        if res["diseases"]:
            st.write("**Bệnh phát hiện:**", ", ".join(f"`{d['name_vi']}`" for d in res["diseases"]))

        st.markdown("---")
        
        # 1. Đọc và LỌC file Markdown theo giai đoạn phát hiện được
       # 1. Đọc và LỌC file Markdown theo giai đoạn phát hiện được
        

        # 2. Gọi AI Tư vấn
        if use_llm:
            kfile = pipe.detector.registry.get_config(res["crop"]).get("knowledge_file")
            img_key = file.name  # dùng để biết ảnh có đổi không, tạo lại phiên chat mới

            if st.session_state.get("chat_img") != img_key:
                with st.spinner("Gemini đang soạn tư vấn..."):
                    try:
                        chat, first_msg = CropAdvisor().start_session(res, kfile)
                        st.session_state["chat"] = chat
                        st.session_state["chat_img"] = img_key
                        st.session_state["chat_history"] = [("assistant", first_msg)]
                    except Exception as e:
                        st.error(f"Không gọi được Gemini: {str(e)[:200]}")
                        st.session_state["chat"] = None

            for role, msg in st.session_state.get("chat_history", []):
                with st.chat_message(role):
                    st.markdown(msg)

            if st.session_state.get("chat"):
                question = st.chat_input("Hỏi thêm về cây này...")
                if question:
                    st.session_state["chat_history"].append(("user", question))
                    with st.spinner("Đang trả lời..."):
                        try:
                            answer = st.session_state["chat"].ask(question)
                        except Exception as e:
                            answer = f"Lỗi khi gọi Gemini: {str(e)[:200]}"
                    st.session_state["chat_history"].append(("assistant", answer))
                    st.rerun()