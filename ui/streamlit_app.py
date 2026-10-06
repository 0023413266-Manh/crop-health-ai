import random
import glob
import os
import sys
import tempfile
import time
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cropai.pipeline import CropPipeline
from cropai.llm.advisor import CropAdvisor
from cropai.vision.visualize import draw_boxes

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Health AI - Hệ Thống Chẩn Đoán Cây Trồng",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cao cấp
st.markdown("""
    <style>
    /* Ẩn bớt giao diện thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Font chữ & Màu sắc chủ đạo */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }

    /* Tiêu đề chính */
    .main-title {
        font-size: 2.2rem !important;
        color: #1b5e20 !important;
        font-weight: 800 !important;
        margin-bottom: 0px !important;
    }
    .sub-title {
        font-size: 1.05rem !important;
        color: #4b6b4e !important;
        margin-bottom: 25px !important;
    }

    /* Bảng chỉ số Metrics Card */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2e7d32 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #f4fbf4 !important;
        border: 1px solid #c8e6c9 !important;
        padding: 12px 18px !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    /* Khung Expander & Tabs */
    div[data-testid="stExpander"] {
        border: 1.5px solid #a5d6a7 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
    }
    
    /* Nút bấm mẫu & Upload */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }

        /* Khung upload ảnh */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed #4caf50 !important;
        border-radius: 14px !important;
        background-color: #f9fbf9 !important;
    }

    /* Chặn ảnh kết quả bị kéo giãn/vỡ bố cục, giữ đúng tỷ lệ ảnh gốc */
    div[data-testid="stImage"] img {
        height: 480px !important;
        width: 100% !important;
        object-fit: contain !important;
        border-radius: 14px !important;
        border: 1px solid #c8e6c9 !important;
        background-color: #f4fbf4 !important;
    }

    /* Banner trạng thái sức khỏe */
    .health-banner {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 18px;
    }
    .health-banner .icon { font-size: 2rem; }
    .health-banner .title { font-size: 1.15rem; font-weight: 700; margin: 0; }
    .health-banner .desc { font-size: 0.92rem; margin: 2px 0 0 0; opacity: 0.85; }

    /* Timeline giai đoạn */
    .stage-timeline { display: flex; align-items: flex-start; margin: 10px 0 20px 0; }
    .stage-step { flex: 1; text-align: center; position: relative; }
    .stage-step .dot {
        width: 18px; height: 18px; border-radius: 50%;
        background: #c8e6c9; margin: 0 auto 8px auto;
        border: 3px solid #c8e6c9; position: relative; z-index: 2;
    }
    .stage-step.done .dot { background: #66bb6a; border-color: #66bb6a; }
    .stage-step.current .dot {
        background: #2e7d32; border-color: #2e7d32;
        box-shadow: 0 0 0 5px rgba(46,125,50,0.2);
    }
    .stage-step .line {
        position: absolute; top: 9px; left: -50%; width: 100%;
        height: 3px; background: #c8e6c9; z-index: 1;
    }
    .stage-step.done .line, .stage-step.current .line { background: #66bb6a; }
    .stage-step:first-child .line { display: none; }
    .stage-step .label {
        font-size: 0.82rem; color: #4b6b4e; font-weight: 500;
    }
    .stage-step.current .label { color: #1b5e20; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & KNOWLEDGE BASE
# -----------------------------------------------------------------------------
def get_stage_knowledge(crop_id: str, detected_stage: str) -> str:
    """Trích xuất cẩm nang thuộc đúng giai đoạn phát hiện."""
    md_path = f"knowledge/{crop_id}.md"
    if not os.path.exists(md_path):
        return "⚠️ Chưa có dữ liệu cẩm nang kỹ thuật cho loại cây này."
        
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = content.split('\n## ')
    for section in sections:
        lines = section.strip().split('\n')
        section_header = lines[0].strip().lower()
        if detected_stage and (detected_stage.lower() in section_header or section_header in detected_stage.lower()):
            return "## " + section
            
    return f"ℹ️ *Chưa có hướng dẫn riêng cho giai đoạn '{detected_stage}'. Dưới đây là thông tin chung:*\n\n{sections[0]}"

@st.cache_resource
def load_pipeline():
    return CropPipeline()

pipe = load_pipeline()
crops = pipe.detector.registry.list_crops()
names = {c["name_vi"]: c["id"] for c in crops}


def get_stage_order(crop_id):
    """Lấy danh sách tên giai đoạn theo đúng thứ tự khai báo trong file YAML của cây đó."""
    cfg = pipe.detector.registry.get_config(crop_id)
    for m in cfg.get("models", []):
        if m.get("type") == "stage":
            return [v["name_vi"] for v in m["classes"].values()]
    return []


def render_stage_timeline(stage_order, current_name_vi):
    if not stage_order:
        return
    try:
        current_idx = stage_order.index(current_name_vi)
    except ValueError:
        current_idx = -1

    html = '<div class="stage-timeline">'
    for i, name in enumerate(stage_order):
        css_class = "stage-step"
        if i < current_idx:
            css_class += " done"
        elif i == current_idx:
            css_class += " current"
        html += f'''
        <div class="{css_class}">
            <div class="dot"><div class="line"></div></div>
            <div class="label">{name}</div>
        </div>'''
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def compute_health_status(res, classifier_conf):
    diseases = res.get("diseases") or []
    if diseases:
        return {"icon": "🔴", "title": "Cần chú ý — phát hiện dấu hiệu bất thường",
                "desc": "Hệ thống phát hiện sâu bệnh, nên kiểm tra và xử lý sớm.",
                "bg": "#fdecea", "fg": "#c62828"}

    stage_key = (res.get("stage") or {}).get("key", "")
    if "decline" in stage_key:
        return {"icon": "🔴", "title": "Cây đang suy yếu",
                "desc": "Phát hiện dấu hiệu suy thoái, nên kiểm tra rễ và dinh dưỡng ngay.",
                "bg": "#fdecea", "fg": "#c62828"}

    if res.get("stage") is None:
        return {"icon": "🟡", "title": "Chưa xác định rõ giai đoạn",
                "desc": "Nên chụp lại ảnh rõ nét hơn để có kết quả chính xác.",
                "bg": "#fff8e1", "fg": "#f9a825"}

    if classifier_conf is not None and classifier_conf < 0.6:
        return {"icon": "🟡", "title": "Cần xác nhận thêm",
                "desc": "Độ tin cậy nhận diện chưa cao, kết quả chỉ mang tính tham khảo.",
                "bg": "#fff8e1", "fg": "#f9a825"}

    return {"icon": "🟢", "title": "Cây đang phát triển bình thường",
            "desc": "Không phát hiện dấu hiệu bất thường trong ảnh này.",
            "bg": "#eafaf0", "fg": "#2e7d32"}
# -----------------------------------------------------------------------------
# 3. SIDEBAR - THAM SỐ CẤU HÌNH HỆ THỐNG
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628324.png", width=65)
    st.header("⚙️ Cấu hình Hệ thống")
    
    mode = st.radio("Chế độ chọn cây", ["🤖 Tự động nhận diện (Tầng 1)", "🖐️ Chọn thủ công"])
    manual_crop = None
    if mode == "🖐️ Chọn thủ công":
        manual_crop = names[st.selectbox("Loại cây trồng", list(names))]
    
    st.divider()
    
    # Nâng cấp: Cho chỉnh ngưỡng tin cậy YOLO
    conf_thresh = st.slider("Ngưỡng nhận diện (Confidence)", 0.1, 1.0, 0.35, step=0.05)
    
    st.divider()
    use_llm = st.checkbox("Bật AI Tư vấn Sinh trưởng", value=True)
    if use_llm:
        llm_model = st.selectbox("Mô hình LLM", ["Gemini 1.5-Flash", "Ollama (Local)"])

# -----------------------------------------------------------------------------
# 4. MAIN UI - HEADER & SAMPLE SELECTION
# -----------------------------------------------------------------------------
st.markdown("<p class='main-title'>🌱 Crop Health AI</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống AI 2 tầng: Nhận diện loài cây & Phân tích chuyên sâu giai đoạn sinh trưởng</p>", unsafe_allow_html=True)

# Khởi tạo bộ nhớ tạm để giữ ảnh mẫu không bị mất khi chỉnh slider
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

st.write("**🖼️ Thử nhanh với ảnh mẫu (Nhấn nhiều lần để đổi ảnh ngẫu nhiên):**")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

# Dùng glob để lấy danh sách toàn bộ ảnh khớp tiền tố, sau đó chọn ngẫu nhiên
if col_s1.button("🥥 Cây Dừa", use_container_width=True):
    dua_imgs = glob.glob(str(SAMPLES_DIR / "dua_test_*.jpg"))
    if dua_imgs:
        st.session_state.selected_sample = random.choice(dua_imgs)
        st.session_state.image_source = "sample"

if col_s2.button("🥭 Cây Đu Đủ", use_container_width=True):
    dudu_imgs = glob.glob(str(SAMPLES_DIR / "dudu_test_*.jpg"))
    if dudu_imgs:
        st.session_state.selected_sample = random.choice(dudu_imgs)
        st.session_state.image_source = "sample"

if col_s3.button("🍌 Cây Chuối", use_container_width=True):
    chuoi_imgs = glob.glob(str(SAMPLES_DIR / "chuoi_test_*.jpg"))
    if chuoi_imgs:
        st.session_state.selected_sample = random.choice(chuoi_imgs)
        st.session_state.image_source = "sample"

if col_s4.button("🥭 Cây Xoài", use_container_width=True):
    xoai_imgs = glob.glob(str(SAMPLES_DIR / "xoai_test_*.jpg"))
    if xoai_imgs:
        st.session_state.selected_sample = random.choice(xoai_imgs)
        st.session_state.image_source = "sample"

file = st.file_uploader("Hoặc tải ảnh cây trồng của bạn lên đây", type=["jpg", "jpeg", "png"])
st.info("💡 Để AI nhận diện chính xác nhất, hãy chụp cận cảnh 1 cây/quả, tránh để nhiều loại cây khác xen lẫn trong khung hình.")

# Nếu người dùng vừa thật sự chọn 1 file mới qua uploader, nguồn là "upload"
if file:
    st.session_state.image_source = "upload"

# Xử lý nguồn ảnh: ưu tiên theo HÀNH ĐỘNG GẦN NHẤT (cờ image_source), không chỉ theo việc còn tồn tại hay không
img_input_path = None
source = st.session_state.get("image_source")

if source == "upload" and file:
    suffix = Path(file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        img_input_path = tmp.name

elif source == "sample" and st.session_state.selected_sample:
    if os.path.exists(st.session_state.selected_sample):
        img_input_path = st.session_state.selected_sample
    else:
        st.error(f"❌ Không tìm thấy file: `{st.session_state.selected_sample}`.")

elif file:   # trường hợp khởi đầu: có file nhưng chưa từng bấm nút nào
    suffix = Path(file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        img_input_path = tmp.name
# -----------------------------------------------------------------------------
# 5. XỬ LÝ NHẬN DIỆN & HIỂN THỊ KẾT QUẢ DASHBOARD
# -----------------------------------------------------------------------------
if img_input_path:
    st.divider()
    
    # Nếu người dùng vừa bấm chọn 1 trong 3 gợi ý top-3, ưu tiên dùng lựa chọn đó
    crop_override = st.session_state.pop("manual_crop_override", None)
    force_run = st.session_state.pop("force_run_despite_quality", False)

    with st.spinner("🔍 AI đang phân tích hình ảnh (Tầng 1 Classifier & Tầng 2 Detector)..."):
        out = pipe.run(img_input_path, crop_id=manual_crop or crop_override,
                        skip_quality_check=force_run)

    
    if out["status"] == "poor_quality":
        st.warning("⚠️ " + out["message"])
        st.caption(f"Độ nét: {out['quality']['blur_score']} | Độ sáng: {out['quality']['brightness']}/255")
        st.error("🚫 Lưu ý: Kết quả nhận diện với ảnh chất lượng kém **có rủi ro sai lệch cao**. "
                 "Khuyến nghị chụp lại ảnh thay vì bỏ qua cảnh báo này.")
        if st.button("Tôi hiểu rủi ro, vẫn tiếp tục nhận diện", use_container_width=True):
            st.session_state["force_run_despite_quality"] = True
            st.rerun()
    elif out["status"] == "need_manual_choice":
        st.warning(out["message"])
        top_k = out["classifier"].get("top_k", [])
        if top_k:
            st.write("**Ảnh này giống với các loại cây sau, hãy chọn đúng loại:**")
            cols = st.columns(len(top_k))
            for col, item in zip(cols, top_k):
                with col:
                    label = f"{item.get('name_vi', item['crop_id'])}\n({item['conf']*100:.0f}%)"
                    if st.button(label, key=f"pick_{item['crop_id']}", use_container_width=True):
                        st.session_state["manual_crop_override"] = item["crop_id"]
                        st.rerun()
            st.caption("Không đúng loại nào trong 3 gợi ý? Chọn 'Chọn thủ công' ở thanh bên để xem đầy đủ danh sách.")
        else:
            st.json(out["classifier"])
    elif out["status"] == "unsupported":
        st.info(out["message"])
    else:
        res = out["result"]
        stage_vi = res["stage"]["name_vi"] if res["stage"] else "Chưa xác định"
        detections = res.get("detections", [])
        classifier_conf = out["classifier"]["conf"] if out.get("classifier") else None

        # --- BANNER TRẠNG THÁI SỨC KHỎE TỔNG QUAN ---
        health = compute_health_status(res, classifier_conf)
        st.markdown(f'''
        <div class="health-banner" style="background:{health['bg']}; color:{health['fg']};">
            <span class="icon">{health['icon']}</span>
            <div>
                <p class="title">{health['title']}</p>
                <p class="desc">{health['desc']}</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # --- TIMELINE GIAI ĐOẠN SINH TRƯỞNG ---
        stage_order = get_stage_order(res["crop"])
        if stage_order:
            render_stage_timeline(stage_order, stage_vi)

        # BỐ CỤC DASHBOARD 2 CỘT
        col_img, col_metrics = st.columns([3, 2], gap="large")
        
        # --- CỘT TRÁI: HÌNH ẢNH KẾT QUẢ ---
        with col_img:
            st.subheader("📸 Kết quả Phân tích Hình ảnh")
            out_img = str(Path(tempfile.gettempdir()) / "cropai_result.jpg")
            
            # Lọc detection theo conf_thresh từ slider
            filtered_dets = [d for d in detections if d.get("conf", 1.0) >= conf_thresh]
            draw_boxes(img_input_path, filtered_dets, out_img)
            
            st.image(Image.open(out_img), caption=f"Nhận diện vùng đối tượng (Conf >= {conf_thresh})", use_container_width=True)

        # --- CỘT PHẢI: METRICS & BẢNG THỐNG KÊ ---
        with col_metrics:
            st.subheader("📊 Thống kê Sinh trưởng")
            
            # Thẻ chỉ số chính (Metrics)
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Loài cây trồng", res['crop_name_vi'])
            with m2:
                conf_val = f"{out['classifier']['conf']*100:.2f}%" if out.get('classifier') else "Thủ công"
                st.metric("Độ tin cậy Tầng 1", conf_val)
            
            st.write("")
            st.metric("Giai đoạn / Trạng thái chính", stage_vi)
            
            # Bảng thống kê chi tiết các đối tượng phát hiện (Bounding boxes count)
            st.write("**Chi tiết số lượng phát hiện (Tầng 2):**")
            if filtered_dets:
                counts = {}
                for d in filtered_dets:
                    name = d.get("class_name_vi") or d.get("class_name") or "Đối tượng"
                    counts[name] = counts.get(name, 0) + 1
                
                df_counts = pd.DataFrame(list(counts.items()), columns=["Giai đoạn / Dấu hiệu", "Số lượng"])
                st.dataframe(df_counts, use_container_width=True, hide_index=True)
            else:
                st.info("Không phát hiện đối tượng chi tiết nào vượt ngưỡng tin cậy.")

            if res["diseases"]:
                st.error(f"⚠️ **Cảnh báo bệnh:** {', '.join(d['name_vi'] for d in res['diseases'])}")

        # -----------------------------------------------------------------------------
        # 6. KHU VỰC TƯ VẤN AI & CẨM NANG KỸ THUẬT (TABS)
        # -----------------------------------------------------------------------------
        st.divider()
        tab_ai, tab_guide = st.tabs(["🤖 AI Gemini Tư vấn Tương tác", "📚 Cẩm nang Kỹ thuật Sinh trưởng"])
        
        # --- TAB 1: CHAT AI GEMINI ---
        with tab_ai:
            if use_llm:
                kfile = pipe.detector.registry.get_config(res["crop"]).get("knowledge_file")
                img_key = file.name if file else st.session_state.selected_sample

                if st.session_state.get("chat_img") != img_key:
                    with st.spinner("🤖 Gemini đang đọc cẩm nang và khởi tạo tư vấn..."):
                        try:
                            chat, first_msg = CropAdvisor().start_session(res, kfile)
                            st.session_state["chat"] = chat
                            st.session_state["chat_img"] = img_key
                            st.session_state["chat_history"] = [("assistant", first_msg)]
                        except Exception as e:
                            st.error(f"Không kết nối được với mô hình AI: {str(e)[:200]}")
                            st.session_state["chat"] = None

                # Hiển thị lịch sử chat
                for role, msg in st.session_state.get("chat_history", []):
                    with st.chat_message(role):
                        st.markdown(msg)

                # Ô nhập câu hỏi tư vấn tiếp
                if st.session_state.get("chat"):
                    if question := st.chat_input("Hỏi AI thêm về chế độ phân bón, tưới nước hoặc bệnh hại..."):
                        st.session_state["chat_history"].append(("user", question))
                        with st.spinner("AI đang suy nghĩ câu trả lời..."):
                            try:
                                answer = st.session_state["chat"].ask(question)
                            except Exception as e:
                                answer = f"Lỗi khi phản hồi: {str(e)[:200]}"
                        st.session_state["chat_history"].append(("assistant", answer))
                        st.rerun()
            else:
                st.info("💡 Bạn đã tắt tính năng Tư vấn bằng AI. Hãy tích chọn 'Bật AI Tư vấn' ở thanh bên để sử dụng.")

        # --- TAB 2: CẨM NANG TRA CỨU ---
        with tab_guide:
            st.markdown("### 📖 Trích xuất Cẩm nang Kỹ thuật")
            knowledge_text = get_stage_knowledge(res["crop"], stage_vi)
            st.markdown(knowledge_text)