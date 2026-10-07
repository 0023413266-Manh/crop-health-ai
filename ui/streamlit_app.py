import glob
import html
import os
import random
from pathlib import Path
import sys
import tempfile
import time

from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from spotlight_selector import select_spotlight  # type: ignore
except ImportError:
    from ui.spotlight_selector import select_spotlight  # type: ignore

from cropai import history_store
from cropai.llm.advisor import CropAdvisor
from cropai.pipeline import CropPipeline
from cropai.vision.visualize import draw_boxes


# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Crop Health AI - Hệ Thống Chẩn Đoán Cây Trồng",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# 2. KHỞI TẠO SESSION STATE BẢO TOÀN KIẾN TRÚC CONVERSATION
# -----------------------------------------------------------------------------

defaults = {
    # Quản lý Conversation & Panel
    "current_conversation_id": None,
    "conversation_view_mode": "home",
    "app_session_initialized": False,
    "active_panel": None,           # Giữ tương thích session cũ; Search/Settings dùng dialog
    "selected_analysis_id": None,   # Phân tích đang xem trong conversation hiện tại
    "search_history_query": "",
    "search_dialog_open": False,
    "search_delete_popover_nonce": 0,
    # Cài đặt phân tích
    "setting_mode": "🤖 Tự động nhận diện (Tầng 1)",
    "setting_manual_crop": None,
    "setting_conf_thresh": 0.35,
    "setting_use_llm": True,
    "setting_llm_model": "Gemini 1.5-Flash",
    # Trạng thái ảnh & pipeline
    "selected_sample": None,
    "original_img_path": None,
    "uploaded_file_path": None,
    "cropped_img_path": None,
    "is_cropping": False,
    "last_file_id": None,
    "last_uploaded_file_id": None,
    "last_result_key": None,
    "last_saved_result_key": None,
    "last_pipeline_out": None,
    "spotlight_image_url": None,
    "spotlight_url_key": None,
    "image_source": None,
    "manual_crop_override": None,
    "spotlight_regions": [],
    "spotlight_active_index": -1,
    "spotlight_active": None,
    "chat": None,
    "chat_img": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & PIPELINE
# -----------------------------------------------------------------------------

def get_stage_knowledge(crop_id: str, detected_stage: str) -> str:
    if not crop_id:
        return "⚠️ Chưa có dữ liệu cẩm nang kỹ thuật cho loại cây này."

    if isinstance(detected_stage, dict):
        detected_stage = detected_stage.get("name_vi") or detected_stage.get("key") or ""
    detected_stage = str(detected_stage or "")

    md_path = ROOT / "knowledge" / f"{crop_id}.md"
    if not md_path.exists():
        return "⚠️ Chưa có dữ liệu cẩm nang kỹ thuật cho loại cây này."

    content = md_path.read_text(encoding="utf-8")
    sections = content.split("\n## ")

    for section in sections:
        lines = section.strip().split("\n")
        if not lines:
            continue
        section_header = lines[0].strip().lower()
        if detected_stage and (
            detected_stage.lower() in section_header
            or section_header in detected_stage.lower()
        ):
            return "## " + section

    return (
        f"ℹ️ *Chưa có hướng dẫn riêng cho giai đoạn '{detected_stage}'. "
        f"Dưới đây là thông tin chung:*\n\n" + sections[0]
    )


@st.cache_resource
def load_pipeline():
    return CropPipeline()


pipe = load_pipeline()
history_store.init_db()
history_store.init_history_storage()

crops = pipe.detector.registry.list_crops()
crop_names_map = {c["name_vi"]: c["id"] for c in crops}


def close_search_dialog():
    st.session_state["search_dialog_open"] = False


@st.dialog(" ", on_dismiss=close_search_dialog)
def show_search_dialog():
    """Tìm và mở các cuộc trò chuyện gần đây trong modal căn giữa."""
    search_val = st.text_input(
        "Tìm kiếm cuộc trò chuyện",
        value=st.session_state.get("search_history_query", ""),
        placeholder="🔍 Tìm theo tên, cây, tin nhắn...",
        label_visibility="collapsed",
        key="search_panel_input",
    )
    st.session_state["search_history_query"] = search_val.strip()

    conv_list = history_store.list_conversations(query=st.session_state["search_history_query"])
    if st.session_state["search_history_query"] and not conv_list:
        st.caption("ℹ️ Không tìm thấy cuộc trò chuyện phù hợp.")
        return
    if not conv_list:
        st.caption("Chưa có cuộc trò chuyện nào.")
        return

    grouped = history_store.group_conversations_by_time(conv_list)
    current_cid = st.session_state.get("current_conversation_id")
    for grp_label in ["HÔM NAY", "HÔM QUA", "7 NGÀY TRƯỚC", "CŨ HƠN"]:
        if grp_label not in grouped:
            continue
        st.markdown(f"<div class='history-group-title'>{grp_label}</div>", unsafe_allow_html=True)
        for conversation in grouped[grp_label]:
            cid = conversation["id"]
            title = conversation.get("title") or "Cuộc trò chuyện"
            rel_time = conversation.get("relative_time") or ""
            analyses_count = len(conversation.get("analyses", []))

            with st.container(key=f"search_dialog_item_{cid}"):
                col1, col2 = st.columns([0.88, 0.12], vertical_alignment="center")
                with col1:
                    if st.button(
                        f"💬  {title} · {rel_time} · {analyses_count} ảnh",
                        key=f"dialog_conv_{cid}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        st.session_state["current_conversation_id"] = cid
                        st.session_state["conversation_view_mode"] = "history"
                        st.session_state["selected_analysis_id"] = None
                        st.session_state["chat"] = None
                        st.session_state["last_file_id"] = None
                        st.session_state["active_panel"] = None
                        st.session_state["search_dialog_open"] = False
                        reset_cropped_state()
                        st.rerun()

                with col2:
                    with st.popover(
                        "🗑️",
                        key=f"btn_del_{cid}_{st.session_state['search_delete_popover_nonce']}",
                        help="Xóa đoạn chat này",
                    ):
                        st.markdown("Xóa cuộc trò chuyện này?")
                        cancel_col, confirm_col = st.columns(2, gap="small")
                        with cancel_col:
                            if st.button("Hủy", key=f"dialog_cancel_delete_{cid}", use_container_width=True):
                                st.session_state["search_delete_popover_nonce"] += 1
                                st.rerun(scope="app")
                        with confirm_col:
                            if st.button(
                                "Xóa",
                                key=f"dialog_delete_confirm_{cid}",
                                type="primary",
                                use_container_width=True,
                            ):
                                history_store.delete_conversation(cid)
                                if st.session_state.get("current_conversation_id") == cid:
                                    st.session_state["current_conversation_id"] = None
                                    st.session_state["conversation_view_mode"] = "home"
                                    st.session_state["selected_analysis_id"] = None
                                    st.session_state["chat"] = None
                                st.session_state["active_panel"] = None
                                st.toast("Đã xóa đoạn chat!", icon="🗑️")
                                st.rerun()


@st.dialog("⚙️ Cài đặt Phân tích")
def show_settings_dialog():
    """Cấu hình nhận diện cây và trợ lý AI trong modal."""
    st.markdown("**⚙️ Cấu hình nhận diện:**")
    mode = st.radio(
        "Chế độ chọn cây",
        ["🤖 Tự động nhận diện (Tầng 1)", "🖐️ Chọn thủ công"],
        index=0 if st.session_state["setting_mode"] == "🤖 Tự động nhận diện (Tầng 1)" else 1,
        key="radio_setting_mode",
    )
    st.session_state["setting_mode"] = mode

    if mode == "🖐️ Chọn thủ công":
        crop_options = list(crop_names_map)
        selected_crop = st.session_state.get("setting_manual_crop")
        default_index = next(
            (i for i, name in enumerate(crop_options) if crop_names_map[name] == selected_crop),
            0,
        )
        chosen_name = st.selectbox(
            "Loại cây trồng",
            crop_options,
            index=default_index,
            key="select_manual_crop",
        )
        st.session_state["setting_manual_crop"] = crop_names_map[chosen_name]
    else:
        st.session_state["setting_manual_crop"] = None

    st.session_state["setting_conf_thresh"] = st.slider(
        "Ngưỡng Confidence",
        0.10,
        1.00,
        value=st.session_state["setting_conf_thresh"],
        step=0.05,
        key="slider_conf_thresh",
    )

    st.divider()
    st.markdown("**🧠 Trợ lý AI & Mô hình LLM:**")
    st.session_state["setting_use_llm"] = st.checkbox(
        "Bật AI Tư vấn Sinh trưởng",
        value=st.session_state["setting_use_llm"],
        key="chk_use_llm",
    )
    if st.session_state["setting_use_llm"]:
        st.session_state["setting_llm_model"] = st.selectbox(
            "Mô hình LLM",
            ["Gemini 1.5-Flash", "Ollama (Local)"],
            index=0 if "Gemini" in st.session_state["setting_llm_model"] else 1,
            key="select_llm_model",
        )


def get_stage_order(crop_id):
    if not crop_id:
        return []
    cfg = pipe.detector.registry.get_config(crop_id)
    if not cfg:
        return []
    for model in cfg.get("models", []):
        if isinstance(model, dict) and model.get("type") == "stage":
            return [v.get("name_vi", str(v)) if isinstance(v, dict) else str(v) for v in model.get("classes", {}).values()]
    return []


def render_stage_timeline(stage_order, current_name_vi):
    if not stage_order:
        return

    if isinstance(current_name_vi, dict):
        current_name_vi = current_name_vi.get("name_vi") or current_name_vi.get("key") or ""
    current_name_vi = str(current_name_vi or "")

    current_idx = -1
    cur_clean = current_name_vi.strip().lower()

    for idx, name in enumerate(stage_order):
        if name.strip().lower() == cur_clean:
            current_idx = idx
            break

    if current_idx == -1 and cur_clean and cur_clean != "chưa xác định":
        for idx, name in enumerate(stage_order):
            name_clean = name.strip().lower()
            if cur_clean in name_clean or name_clean in cur_clean:
                current_idx = idx
                break

    steps = []
    for i, name in enumerate(stage_order):
        if current_idx >= 0 and i < current_idx:
            css_class = "stage-step done"
        elif i == current_idx:
            css_class = "stage-step current"
        else:
            css_class = "stage-step pending"

        steps.append(
            f'<div class="{css_class}">'
            f'<div class="dot"></div>'
            f'<div class="label">{html.escape(name)}</div>'
            f'</div>'
        )

    timeline_html = f'<div class="stage-timeline">{"".join(steps)}</div>'
    st.html(timeline_html)


def safe_get_stage_info(res_or_analysis):
    """
    Trích xuất an toàn (stage_name, stage_key, stage_obj)
    tương thích CẢ HAI dạng dữ liệu: dict lẫn string.
    """
    if not isinstance(res_or_analysis, dict):
        val_str = str(res_or_analysis or "")
        return val_str, val_str, {}

    stage_obj = res_or_analysis.get("stage_obj")
    stage_data = res_or_analysis.get("stage")
    stage_name_vi = res_or_analysis.get("stage_name_vi")

    if isinstance(stage_obj, dict) and stage_obj:
        stage_key = str(stage_obj.get("key") or "")
        stage_name = str(stage_obj.get("name_vi") or (stage_data if isinstance(stage_data, str) else "") or stage_name_vi or "")
        final_obj = stage_obj
    elif isinstance(stage_data, dict):
        stage_key = str(stage_data.get("key") or "")
        stage_name = str(stage_data.get("name_vi") or stage_name_vi or "")
        final_obj = stage_data
    elif isinstance(stage_data, str) and stage_data.strip():
        stage_key = stage_data.strip()
        stage_name = stage_data.strip()
        final_obj = {"key": stage_key, "name_vi": stage_name}
    elif isinstance(stage_name_vi, str) and stage_name_vi.strip():
        stage_key = stage_name_vi.strip()
        stage_name = stage_name_vi.strip()
        final_obj = {"key": stage_key, "name_vi": stage_name}
    else:
        stage_key = ""
        stage_name = ""
        final_obj = {}

    if not stage_name and isinstance(stage_name_vi, str):
        stage_name = stage_name_vi.strip()

    if not stage_name:
        stage_name = "Chưa xác định"

    return stage_name, stage_key, final_obj


def compute_health_status(res_or_analysis, classifier_conf=None):
    if not isinstance(res_or_analysis, dict):
        return {
            "icon": "🟢",
            "title": "Cây đang phát triển bình thường",
            "desc": "Không phát hiện dấu hiệu bất thường trong ảnh này.",
            "bg": "#eafaf0",
            "fg": "#2e7d32",
        }

    diseases = res_or_analysis.get("diseases") or []
    if isinstance(diseases, list) and len(diseases) > 0:
        return {
            "icon": "🔴",
            "title": "Cần chú ý — phát hiện dấu hiệu bất thường",
            "desc": "Hệ thống phát hiện sâu bệnh, nên kiểm tra và xử lý sớm.",
            "bg": "#fdecea",
            "fg": "#c62828",
        }

    stage_name, stage_key, _ = safe_get_stage_info(res_or_analysis)

    if isinstance(stage_key, str) and "decline" in stage_key.lower():
        return {
            "icon": "🔴",
            "title": "Cây đang suy yếu",
            "desc": "Phát hiện dấu hiệu suy thoái, nên kiểm tra rễ và dinh dưỡng ngay.",
            "bg": "#fdecea",
            "fg": "#c62828",
        }

    if not stage_name or stage_name == "Chưa xác định":
        return {
            "icon": "🟡",
            "title": "Chưa xác định rõ giai đoạn",
            "desc": "Nên chụp lại ảnh rõ nét hơn để có kết quả chính xác.",
            "bg": "#fff8e1",
            "fg": "#f9a825",
        }

    return {
        "icon": "🟢",
        "title": "Cây đang phát triển bình thường",
        "desc": "Không phát hiện dấu hiệu bất thường trong ảnh này.",
        "bg": "#eafaf0",
        "fg": "#2e7d32",
    }


def reset_cropped_state():
    st.session_state["cropped_img_path"] = None
    st.session_state["is_cropping"] = False
    st.session_state["spotlight_regions"] = []
    st.session_state["spotlight_active_index"] = -1
    st.session_state["spotlight_active"] = None
    st.session_state["last_result_key"] = None
    st.session_state["last_pipeline_out"] = None
    st.session_state["spotlight_image_url"] = None
    st.session_state["spotlight_url_key"] = None


def crop_image_from_region(image_path, region):
    if not region or "x" not in region:
        return Image.open(image_path).convert("RGB")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    x = float(region.get("x", 0))
    y = float(region.get("y", 0))
    w = float(region.get("w", 1))
    h = float(region.get("h", 1))

    left = int(x * width)
    top = int(y * height)
    right = int((x + w) * width)
    bottom = int((y + h) * height)

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    return image.crop((left, top, right, bottom))


def get_sample_thumbnail_path(analysis):
    """Ảnh đại diện đúng cây/giai đoạn cho bản history cũ chưa lưu ảnh."""
    crop_id = str(analysis.get("crop") or "").lower()
    stage_name, _, _ = safe_get_stage_info(analysis)
    stage_name = stage_name.lower()

    sample_by_stage = {
        ("banana", "buồng quả xanh"): "chuoi_test_1.jpg",
        ("coconut", "cây con"): "dua_test_1.jpg",
        ("coconut", "trái khô (dừa khô)"): "dua_test_4.jpg",
    }
    sample_name = sample_by_stage.get((crop_id, stage_name))
    if not sample_name:
        return None
    candidate = SAMPLES_DIR / sample_name
    return str(candidate) if candidate.exists() else None


@st.cache_data(ttl=300, show_spinner=False)
def get_legacy_analysis_image_lookup():
    """Tra ảnh từ conversation một ảnh cũ khi record đa ảnh bị thiếu image_path."""
    lookup = {}
    for conversation in history_store.list_conversations():
        conv_id = conversation.get("id")
        if not conv_id:
            continue

        analyses = conversation.get("analyses")
        if isinstance(analyses, list) and analyses:
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    continue
                stage_name, _, _ = safe_get_stage_info(analysis)
                crop_id = str(analysis.get("crop") or "").lower()
                image_path = analysis.get("image_path")
                if crop_id and image_path and os.path.exists(image_path):
                    lookup.setdefault((crop_id, stage_name.strip().lower()), image_path)
            continue

        crop_id = str(conversation.get("crop_id") or "").lower()
        stage_name = conversation.get("stage_name_vi") or ""
        stage_data = conversation.get("stage")
        if isinstance(stage_data, dict):
            stage_name = stage_data.get("name_vi") or stage_name
        elif isinstance(stage_data, str) and stage_data.strip():
            stage_name = stage_data
        image_file = conversation.get("image_file") or "image.jpg"
        image_path = history_store.HISTORY_DIR / conv_id / image_file
        key = (crop_id, str(stage_name).strip().lower())
        if crop_id and stage_name and image_path.exists():
            lookup.setdefault(key, str(image_path))
    return lookup


def get_legacy_analysis_image_path(analysis):
    stage_name, _, _ = safe_get_stage_info(analysis)
    key = (str(analysis.get("crop") or "").lower(), stage_name.strip().lower())
    return get_legacy_analysis_image_lookup().get(key)


def start_new_conversation():
    """Tạo một cuộc trò chuyện mới độc lập và chuyển ngay sang phiên đó."""
    new_cid = history_store.create_conversation("Cuộc trò chuyện mới")
    st.session_state["current_conversation_id"] = new_cid
    st.session_state["conversation_view_mode"] = "home"
    st.session_state["selected_analysis_id"] = None
    st.session_state["chat"] = None
    st.session_state["original_img_path"] = None
    st.session_state["uploaded_file_path"] = None
    st.session_state["selected_sample"] = None
    st.session_state.pop("file_uploader_crop", None)
    st.session_state["last_file_id"] = None
    st.session_state["last_uploaded_file_id"] = None
    st.session_state["image_source"] = None
    st.session_state["last_pipeline_out"] = None
    st.session_state["last_result_key"] = None
    st.session_state["last_saved_result_key"] = None
    st.session_state["active_panel"] = None
    reset_cropped_state()
    st.rerun()


def select_analysis(analysis_id):
    """Select a gallery result before Streamlit reruns the page."""
    st.session_state["selected_analysis_id"] = analysis_id


# Mỗi phiên ứng dụng bắt đầu ở Trang chủ với một cuộc trò chuyện mới.
# Không tự động mở cuộc trò chuyện cũ; người dùng chọn chúng từ Search khi cần.
if not st.session_state.get("app_session_initialized"):
    st.session_state["current_conversation_id"] = history_store.create_conversation("Cuộc trò chuyện mới")
    st.session_state["conversation_view_mode"] = "home"
    st.session_state["selected_analysis_id"] = None
    st.session_state["app_session_initialized"] = True

# Đảm bảo luôn có một conversation hiện hành (ví dụ cuộc trò chuyện vừa bị xóa).
if not st.session_state.get("current_conversation_id"):
    st.session_state["current_conversation_id"] = history_store.create_conversation("Cuộc trò chuyện mới")
    st.session_state["conversation_view_mode"] = "home"

active_conv = history_store.get_conversation(st.session_state["current_conversation_id"])
if not active_conv:
    st.session_state["current_conversation_id"] = history_store.create_conversation("Cuộc trò chuyện mới")
    st.session_state["conversation_view_mode"] = "home"
    active_conv = history_store.get_conversation(st.session_state["current_conversation_id"])


# -----------------------------------------------------------------------------
# 4. CSS TÙY CHỈNH: SIDEBAR ICON + DIALOG SEARCH/SETTINGS
# -----------------------------------------------------------------------------

sidebar_width_css = """
section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 76px !important;
    min-width: 76px !important;
    max-width: 76px !important;
    transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
"""

st.markdown(
    f"""
    <style>

    header, [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Streamlit Toolbar: Giữ hiển thị vì chứa stExpandSidebarButton */
    [data-testid="stToolbar"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }}

    /* Nút mở lại sidebar và nút thu gọn sidebar native */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        z-index: 1000000 !important;
    }}

    /* Thay ký tự chevron kép bằng icon sidebar dạng panel, giữ nút hoạt động native. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {{
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 12px !important;
    }}

    [data-testid="stExpandSidebarButton"] > button,
    [data-testid="stSidebarCollapseButton"] > button {{
        width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        min-height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 12px !important;
        background: transparent !important;
        box-shadow: none !important;
        transition: background-color 0.18s ease !important;
    }}

    [data-testid="stExpandSidebarButton"] > button:hover,
    [data-testid="stSidebarCollapseButton"] > button:hover {{
        background: #e8f5e9 !important;
    }}

    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {{
        width: 20px !important;
        height: 20px !important;
        overflow: hidden !important;
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
    }}

    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::before,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::before {{
        content: "◧";
        display: block;
        color: #52675a;
        font-family: Arial, sans-serif;
        font-size: 20px;
        line-height: 20px;
        text-align: center;
    }}

    /* Ẩn nút Deploy, Menu 3 chấm và footer */
    [data-testid="stToolbarActions"],
    [data-testid="stMainMenu"],
    #MainMenu,
    footer {{
        visibility: hidden !important;
        display: none !important;
    }}

    /* Sidebar: Nền và viền tối giản */
    section[data-testid="stSidebar"] {{
        background-color: #f7faf7 !important;
        border-right: 1px solid #e1ebe1 !important;
    }}

    /* Lớp phủ chiếm viewport; section bên trong mới là hộp thoại. */
    div[data-testid="stDialog"] {{
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        max-width: none !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 1rem !important;
        border-radius: 0 !important;
        background: rgba(15, 23, 42, 0.16) !important;
        box-shadow: none !important;
    }}

    section[role="dialog"] {{
        width: min(580px, calc(100vw - 2rem)) !important;
        max-width: 580px !important;
        border-radius: 16px !important;
        background: #ffffff !important;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.16) !important;
    }}

    section[role="dialog"] > div {{
        border-radius: 16px !important;
        background: #ffffff !important;
    }}

    /* Danh sách hội thoại trong Search dialog: phẳng, căn trái, hover dịu. */
    div[data-testid="stDialog"] [class*="st-key-search_dialog_item_"] [class*="st-key-dialog_conv_"] button,
    div[role="dialog"] [class*="st-key-search_dialog_item_"] [class*="st-key-dialog_conv_"] button {{
        width: 100% !important;
        min-height: 54px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        white-space: normal !important;
        border: none !important;
        border-radius: 10px !important;
        background: transparent !important;
        box-shadow: none !important;
        color: #263238 !important;
        transition: background-color 0.16s ease !important;
    }}

    div[data-testid="stDialog"] [class*="st-key-search_dialog_item_"] [class*="st-key-dialog_conv_"] button:hover,
    div[role="dialog"] [class*="st-key-search_dialog_item_"] [class*="st-key-dialog_conv_"] button:hover {{
        background: #f4f6f8 !important;
        border: none !important;
        color: #263238 !important;
    }}

    div[data-testid="stDialog"] [class*="st-key-search_dialog_item_"] [class*="st-key-dialog_conv_"] button > div,
    div[role="dialog"] [class*="st-key-search_dialog_item_"] [class*="st-key-dialog_conv_"] button > div {{
        width: 100% !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }}

    [class*="st-key-btn_del_"] button {{
        width: 40px !important;
        min-width: 40px !important;
        height: 40px !important;
        min-height: 40px !important;
        padding: 0 !important;
        justify-content: center !important;
        border-radius: 6px !important;
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        color: #a0aec0 !important;
        opacity: 1 !important;
    }}
    [class*="st-key-btn_del_"] button:hover {{
        background: #fff5f5 !important;
        color: #e53e3e !important;
    }}
    [class*="st-key-btn_del_"] button > div {{
        width: auto !important;
        justify-content: center !important;
    }}
    [class*="st-key-btn_del_"] button svg {{
        display: none !important;
    }}

    [class*="st-key-dialog_cancel_delete_"] button,
    [class*="st-key-dialog_delete_confirm_"] button {{
        min-height: 32px !important;
        padding: 4px 10px !important;
        border-radius: 7px !important;
        font-size: .85rem !important;
    }}
    [class*="st-key-dialog_cancel_delete_"] button {{
        background: #f7fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #4a5568 !important;
    }}
    [class*="st-key-dialog_delete_confirm_"] button {{
        background: #e53e3e !important;
        border: 1px solid #e53e3e !important;
        color: #ffffff !important;
    }}

    {sidebar_width_css}

    /* Padding trong sidebar & Gom khoảng cách nút lại gần nhau */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
        width: 100% !important;
        min-width: 0 !important;
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
        padding-top: 0.6rem !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {{
        width: 100% !important;
        min-width: 0 !important;
        align-items: stretch !important;
        gap: 0.25rem !important; /* Gom khoảng cách giữa các icon lại gần hơn */
    }}

    /* CSS nút icon Sidebar: Xóa nền trắng, xóa ô viền thô */
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] {{
        box-sizing: border-box !important;
        width: 100% !important;
        min-width: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        transform: none !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] [data-testid="stButton"],
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] [data-testid="stButton"] > div {{
        box-sizing: border-box !important;
        width: 100% !important;
        min-width: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        transform: none !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button {{
        box-sizing: border-box !important;
        width: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        height: 42px !important;
        flex: 0 0 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        border-radius: 10px !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        padding: 0 !important;
        margin: 2px auto !important; /* Giảm khoảng cách lề trên dưới */
        transform: none !important;
        background: transparent !important; /* Bỏ màu nền trắng thô */
        border: none !important; /* Bỏ khung viền xám */
        box-shadow: none !important; /* Bỏ bóng đổ */
        transition: background-color 0.18s ease, transform 0.18s ease !important;
    }}

    /* Giữ nguyên màu tươi sáng của Icon / Emoji */
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button img,
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button svg {{
        filter: none !important;
        opacity: 0.9 !important;
    }}

    /* Streamlit đặt nội dung emoji trong TooltipHoverTarget có flex-end mặc định */
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button > div,
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button [data-testid="stTooltipIcon"],
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button [data-testid="stTooltipHoverTarget"] {{
        box-sizing: border-box !important;
        width: 100% !important;
        min-width: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        transform: none !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button [data-testid="stMarkdownContainer"] {{
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* Hiệu ứng Hover: Nền xanh mầm cây nhẹ, bo góc mềm */
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] button:hover {{
        background: #e8f5e9 !important;
        border: none !important;
        transform: scale(1.06) !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"][class*="_p"] {{
        justify-content: flex-start !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"][class*="_p"] [data-testid="stButton"],
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"][class*="_p"] [data-testid="stButton"] > div {{
        justify-content: flex-start !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"][class*="_p"] button {{
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        height: 42px !important;
        flex: 1 1 auto !important;
        justify-content: flex-start !important;
        padding: 0 8px !important;
        margin: 3px 0 !important;
        font-size: .78rem !important;
        white-space: nowrap !important;
        box-shadow: none !important;
    }}

    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] [data-testid="stMarkdownContainer"] p {{
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }}

    /* Streamlit tạo thêm button bản sao làm vùng hover cho thuộc tính help */
    section[data-testid="stSidebar"] [class*="st-key-btn_icon_"] [data-testid="stTooltipHoverTarget"] button {{
        display: none !important;
    }}

    /* Tiêu đề nhóm lịch sử trong sidebar */
    .history-group-title {{
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #557957 !important;
        letter-spacing: 0.6px !important;
        margin-top: 14px !important;
        margin-bottom: 5px !important;
        text-transform: uppercase !important;
    }}

    /* Thẻ thông tin phiên làm việc chính */
    .session-header-card {{
        background: #f4fbf4;
        border: 1px solid #c8e6c9;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* ẢNH KẾT QUẢ */
    div[data-testid="stImage"] {{
        max-width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
    }}

    div[data-testid="stImage"] img {{
        max-width: 100% !important;
        max-height: 480px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
        margin: 0 auto !important;
    }}

    /* Spotlight custom component */
    div[data-testid="stCustomComponentV1"] {{
        max-width: 100% !important;
    }}

    div[data-testid="stCustomComponentV1"] iframe {{
        max-width: 100% !important;
    }}

    .main .block-container {{
        padding-top: 1.2rem !important;
    }}

    .breadcrumb {{
        color: #849187;
        font-size: 0.78rem;
        line-height: 1.2;
        margin: 0 0 4px 2px;
    }}

    .breadcrumb span {{
        color: #b4bdb5;
        padding: 0 4px;
    }}

    .main-title {{
        font-size: clamp(1.45rem, 2.3vw, 2rem) !important;
        color: #174b2a !important;
        font-weight: 750 !important;
        margin: 0 0 2px 0 !important;
        letter-spacing: -0.035em;
        overflow-wrap: anywhere;
    }}

    .sub-title {{
        font-size: 0.95rem !important;
        color: #68796d !important;
        margin-bottom: 10px !important;
    }}

    /* Card Metrics */
    div[data-testid="stMetricValue"] {{
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #2e7d32 !important;
    }}

    div[data-testid="stMetric"] {{
        background-color: #f4fbf4 !important;
        border: 1px solid #c8e6c9 !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }}

    div[data-testid="stButton"] > button {{
        border-radius: 10px !important;
        border-color: #d8e2da !important;
        transition: background-color .15s ease, border-color .15s ease !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"] {{
        background: #e7f4e9 !important;
        color: #1c5a2d !important;
        border-color: #91c49a !important;
    }}

    .upload-caption {{
        color: #647568;
        font-size: .9rem;
        margin: -8px 0 8px;
    }}

    .gallery-card {{
        min-height: 93px;
        border: 1px solid #dfe7e0;
        border-radius: 12px;
        padding: 10px 12px;
        margin: 0 0 7px;
        background: #fff;
    }}

    .gallery-card.active {{
        background: #f0f8f1;
        border: 2px solid #167a36;
        box-shadow: 0 5px 16px rgba(22, 122, 54, .2);
    }}

    [class*="st-key-gallery_card_"] {{
        position: relative !important;
        isolation: isolate;
        height: 150px;
        min-height: 150px;
        padding: 8px 10px;
        border: 1px solid #dfe7e0;
        border-radius: 12px;
        background: #fff;
        box-sizing: border-box;
        overflow: hidden;
        margin-bottom: 8px;
        cursor: pointer;
    }}

    [class*="st-key-gallery_card_"]:hover {{
        border-color: #84b98a;
        box-shadow: 0 3px 12px rgba(22, 122, 54, .12);
    }}

    [class*="st-key-gallery_card_active_"] {{
        border: 2px solid #167a36 !important;
        background: #f0f8f1;
        box-shadow: 0 5px 16px rgba(22, 122, 54, .2);
    }}

    [class*="st-key-gallery_card_"] [data-testid="stButton"] {{
        position: absolute !important;
        inset: 0 !important;
        z-index: 10 !important;
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
    }}
    [class*="st-key-gallery_card_"] [data-testid="stButton"] button {{
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        opacity: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        cursor: pointer !important;
    }}
    [class*="st-key-gallery_card_"] [data-testid="stImage"] button {{
        display: none !important;
        pointer-events: none !important;
    }}

    [class*="st-key-gallery_scroll_area"] {{
        max-height: 460px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 4px 10px 12px;
        overscroll-behavior: contain;
    }}

    div[data-testid="stChatInput"] {{
        position: sticky !important;
        bottom: 0 !important;
        z-index: 99 !important;
        background: #fff !important;
        padding: 8px 20px 10px 0 !important;
        box-sizing: border-box !important;
    }}

    div[data-testid="stChatMessage"] {{
        box-sizing: border-box !important;
        padding-right: 20px !important;
    }}

    .thumb-placeholder {{
        width: 70px; height: 62px; display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:2px;
        border-radius:8px; background:#edf3ee; color:#728477;
        font-size:1.25rem; text-align:center;
    }}
    .thumb-placeholder small {{ font-size:.58rem; line-height:1.1; }}

    .gallery-title {{ color: #203b29; font-weight: 650; }}
    .gallery-stage {{ color: #647568; font-size: .84rem; margin-top: 3px; }}
    .gallery-status {{ color: #42814d; font-size: .78rem; margin-top: 5px; }}

    .analysis-badges {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px; }}
    .analysis-badge {{ display:inline-flex; align-items:center; padding:6px 11px; border-radius:999px; font-size:.86rem; font-weight:650; }}
    .badge-confidence {{ background:#e8f1ff; color:#1e55a3; }}
    .badge-stage {{ background:#eef4ed; color:#385b3d; }}
    .badge-good {{ background:#e5f6e9; color:#176b31; }}
    .badge-warning {{ background:#fff4d8; color:#805900; }}
    .badge-danger {{ background:#ffebeb; color:#a12828; }}

    [class*="st-key-detection_metric_"] {{
        min-height:105px; padding:12px 14px; border:1px solid #dce7de;
        border-radius:12px; background:linear-gradient(150deg,#fff,#f5faf5);
        box-shadow:0 2px 8px rgba(27,75,42,.06);
    }}
    .detection-icon {{ font-size:1.35rem; line-height:1; margin-bottom:5px; }}
    .detection-name {{ color:#526456; font-size:.82rem; min-height:2.2em; }}
    .detection-count {{ color:#1b5e20; font-size:1.8rem; line-height:1.1; font-weight:750; }}
    .bbox-toggle {{ margin: -4px 0 8px; }}

    @media (max-width: 760px) {{
        .stage-step .label {{ font-size: .68rem; }}
        .health-banner {{ padding: 10px 12px; gap: 9px; }}
        div[data-testid="stImage"] img {{ max-height: 330px !important; }}
        div[data-testid="stHorizontalBlock"]:has([class*="st-key-crop_toggle_btn"]) {{
            flex-direction: column !important;
            gap: 1rem !important;
        }}
        div[data-testid="stHorizontalBlock"]:has([class*="st-key-crop_toggle_btn"]) > div[data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }}
        [class*="st-key-gallery_card_"] {{ height: 150px; min-height: 150px; }}
    }}

    [class*="st-key-crop_toggle_btn"] button {{
        background-color: #ffffff !important;
        border: 1.5px solid #2e7d32 !important;
        font-size: .95rem !important;
        padding: 6px 8px !important;
        border-radius: 8px !important;
    }}

    .health-banner {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 18px;
        border-radius: 12px;
        margin-bottom: 14px;
    }}

    .health-banner .icon {{
        font-size: 1.7rem;
    }}

    .health-banner .title {{
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0;
    }}

    .health-banner .desc {{
        font-size: 0.88rem;
        margin: 2px 0 0 0;
        opacity: 0.9;
    }}

    .stage-timeline {{
        display: flex;
        align-items: flex-start;
        margin: 6px 0 16px 0;
        width: 100%;
    }}

    .stage-step {{
        flex: 1;
        text-align: center;
        position: relative;
    }}

    .stage-step .dot {{
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background: #e0e0e0;
        margin: 0 auto 5px auto;
        border: 3px solid #e0e0e0;
        position: relative;
        z-index: 2;
    }}

    .stage-step.done .dot {{
        background: #66bb6a;
        border-color: #66bb6a;
    }}

    .stage-step.current .dot {{
        background: #2e7d32;
        border-color: #2e7d32;
        box-shadow: 0 0 0 4px rgba(46,125,50,0.22);
    }}

    .stage-step::before {{
        position: absolute;
        top: 7px;
        left: -50%;
        width: 100%;
        height: 3px;
        background: #e0e0e0;
        z-index: 1;
        content: "";
    }}

    .stage-step.done::before,
    .stage-step.current::before {{
        background: #66bb6a;
    }}

    .stage-step:first-child::before {{
        content: none;
    }}

    .stage-step .label {{
        font-size: 0.78rem;
        color: #757575;
        font-weight: 500;
        line-height: 1.25;
        padding: 0 4px;
    }}

    .stage-step.done .label {{
        color: #4b6b4e;
        font-weight: 600;
    }}

    .stage-step.current .label {{
        color: #1b5e20;
        font-weight: 700;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# 5. SIDEBAR: THANH ICON MỎNG, SEARCH/SETTINGS MỞ DIALOG
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="icon-bar-btn">', unsafe_allow_html=True)
    if st.button("🌱", help="Trang chính", key="btn_icon_home"):
        start_new_conversation()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="icon-bar-btn">', unsafe_allow_html=True)
    if st.button("✏️", help="Cuộc trò chuyện mới", key="btn_icon_new"):
        start_new_conversation()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="icon-bar-btn">', unsafe_allow_html=True)
    if st.button("🔍", help="Tìm kiếm", key="btn_icon_search"):
        st.session_state["search_dialog_open"] = True
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get("search_dialog_open"):
        show_search_dialog()

    st.markdown("<div style='height: 110px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="icon-bar-btn">', unsafe_allow_html=True)
    if st.button("⚙️", help="Cài đặt", key="btn_icon_settings"):
        show_settings_dialog()
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. MAIN CONTENT: PHIÊN LÀM VIỆC CONVERSATION ĐA HÌNH ẢNH
# -----------------------------------------------------------------------------

current_cid = st.session_state["current_conversation_id"]
active_conv = history_store.get_conversation(current_cid)
conv_title = active_conv.get("title", "Cuộc trò chuyện mới") if active_conv else "Cuộc trò chuyện mới"
analyses_list = active_conv.get("analyses", []) if active_conv else []
chat_messages = active_conv.get("messages", []) if active_conv else []

# HEADER CONVERSATION: TIÊU ĐỀ & SỐ LƯỢNG ẢNH/TIN NHẮN
col_title, col_actions = st.columns([7, 3], vertical_alignment="center")
with col_title:
    st.markdown(
        "<div class='breadcrumb'>Trang chủ <span>/</span> Vườn cây</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<p class='main-title'>🌱 {html.escape(conv_title)}</p>", unsafe_allow_html=True)
    subtitle = (
        "Chào mừng bạn! Hãy tải ảnh cây trồng để hệ thống bắt đầu phân tích."
        if not analyses_list and not chat_messages
        else f"{len(analyses_list)} ảnh · {len(chat_messages)} tin nhắn"
    )
    st.markdown(
        f"<p class='sub-title'>{html.escape(subtitle)}</p>",
        unsafe_allow_html=True
    )

has_named_analysis = any(
    analysis.get("crop")
    and str(analysis.get("crop_name") or "").strip() not in {"", "Cây trồng"}
    for analysis in analyses_list
)
with col_actions:
    if has_named_analysis:
        col_act1, col_act2 = st.columns([1, 1.12])
        with col_act1:
            with st.popover("✏️ Đổi tên", use_container_width=True):
                st.write("**Đổi tên cuộc trò chuyện**")
                new_title = st.text_input("Tên mới", value=conv_title, key="inline_rename_inp")
                if st.button("Lưu tên", key="btn_inline_save_ren", use_container_width=True):
                    if new_title.strip():
                        history_store.rename_conversation(current_cid, new_title.strip())
                        st.rerun()
        with col_act2:
            if st.button("＋ Chat mới", use_container_width=True, type="primary"):
                start_new_conversation()
    elif st.button("＋ Chat mới", use_container_width=True, type="primary"):
        start_new_conversation()

# KHU VỰC TẢI ẢNH MỚI VÀO CONVERSATION HIỆN TẠI
st.markdown("<p class='upload-caption'>Chọn ảnh từ máy hoặc thử nhanh với ảnh mẫu. Mỗi ảnh mới sẽ được lưu trong cuộc trò chuyện hiện tại.</p>", unsafe_allow_html=True)
upload_tab, sample_tab = st.tabs(["📁 Tải ảnh từ máy", "🖼️ Dùng ảnh mẫu"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Chọn hoặc kéo thả ảnh cây trồng vào đây:",
        type=["jpg", "jpeg", "png"],
        key="file_uploader_crop"
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file_id") != file_id:
            st.session_state.last_uploaded_file_id = file_id
            st.session_state.last_file_id = file_id
            st.session_state.image_source = "upload"
            suffix = Path(uploaded_file.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.uploaded_file_path = tmp.name
            reset_cropped_state()

with sample_tab:
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    if col_s1.button("🥥 Cây Dừa", use_container_width=True, key="smpl_dua"):
        dua_imgs = glob.glob(str(SAMPLES_DIR / "dua_test_*.jpg"))
        if dua_imgs:
            st.session_state.selected_sample = random.choice(dua_imgs)
            st.session_state.image_source = "sample"
            st.session_state.original_img_path = st.session_state.selected_sample
            st.session_state.last_file_id = f"sample_{Path(st.session_state.selected_sample).stem}"
            reset_cropped_state()

    if col_s2.button("🥭 Cây Đu Đủ", use_container_width=True, key="smpl_dudu"):
        dudu_imgs = glob.glob(str(SAMPLES_DIR / "dudu_test_*.jpg"))
        if dudu_imgs:
            st.session_state.selected_sample = random.choice(dudu_imgs)
            st.session_state.image_source = "sample"
            st.session_state.original_img_path = st.session_state.selected_sample
            st.session_state.last_file_id = f"sample_{Path(st.session_state.selected_sample).stem}"
            reset_cropped_state()

    if col_s3.button("🍌 Cây Chuối", use_container_width=True, key="smpl_chuoi"):
        chuoi_imgs = glob.glob(str(SAMPLES_DIR / "chuoi_test_*.jpg"))
        if chuoi_imgs:
            st.session_state.selected_sample = random.choice(chuoi_imgs)
            st.session_state.image_source = "sample"
            st.session_state.original_img_path = st.session_state.selected_sample
            st.session_state.last_file_id = f"sample_{Path(st.session_state.selected_sample).stem}"
            reset_cropped_state()

    if col_s4.button("🥭 Cây Xoài", use_container_width=True, key="smpl_xoai"):
        xoai_imgs = glob.glob(str(SAMPLES_DIR / "xoai_test_*.jpg"))
        if xoai_imgs:
            st.session_state.selected_sample = random.choice(xoai_imgs)
            st.session_state.image_source = "sample"
            st.session_state.original_img_path = st.session_state.selected_sample
            st.session_state.last_file_id = f"sample_{Path(st.session_state.selected_sample).stem}"
            reset_cropped_state()


# Xác định ảnh gốc hiện hành
active_input_img = None
if st.session_state.get("image_source") == "upload" and st.session_state.get("uploaded_file_path"):
    if os.path.exists(st.session_state["uploaded_file_path"]):
        active_input_img = st.session_state["uploaded_file_path"]
elif st.session_state.get("image_source") == "sample" and st.session_state.get("selected_sample"):
    if os.path.exists(st.session_state["selected_sample"]):
        active_input_img = st.session_state["selected_sample"]

st.session_state["original_img_path"] = active_input_img


# -----------------------------------------------------------------------------
# 7. CHẠY PIPELINE NHẬN DIỆN & TỰ ĐỘNG LƯU VÀO CONVERSATION HIỆN TẠI
# -----------------------------------------------------------------------------

if active_input_img and os.path.exists(active_input_img):
    if st.session_state.get("cropped_img_path") and os.path.exists(st.session_state["cropped_img_path"]):
        active_target_img = st.session_state["cropped_img_path"]
        is_analyzing_crop = True
    else:
        active_target_img = active_input_img
        is_analyzing_crop = False

    effective_crop_id = st.session_state.get("setting_manual_crop") or st.session_state.pop("manual_crop_override", None)
    result_key = f"{st.session_state.get('last_file_id')}|{effective_crop_id}|{is_analyzing_crop}"

    # Chạy pipeline nếu chưa cache
    if st.session_state.get("last_result_key") != result_key:
        with st.spinner("🔍 AI đang phân tích hình ảnh..."):
            pipeline_out = pipe.run(
                active_target_img,
                crop_id=effective_crop_id,
                skip_quality_check=True
            )
        st.session_state["last_pipeline_out"] = pipeline_out
        st.session_state["last_result_key"] = result_key
    else:
        pipeline_out = st.session_state["last_pipeline_out"]

    # Xử lý kết quả pipeline
    if pipeline_out["status"] == "need_manual_choice":
        st.warning("⚠️ " + pipeline_out["message"])
        top_k = pipeline_out["classifier"].get("top_k", [])
        if top_k:
            st.markdown("### 🖐️ Chọn đúng loài cây trong ảnh:")
            cols = st.columns(len(top_k))
            for col, item in zip(cols, top_k):
                with col:
                    label = f"🌱 {item.get('name_vi', item['crop_id'])}\n({item['conf']*100:.0f}%)"
                    if st.button(label, key=f"pick_{item['crop_id']}", use_container_width=True):
                        st.session_state["manual_crop_override"] = item["crop_id"]
                        st.rerun()

    elif pipeline_out["status"] == "unsupported":
        st.info(pipeline_out["message"])

    elif pipeline_out["status"] == "ok":
        res = pipeline_out["result"]
        current_crop_id = str(res.get("crop") or "")
        stage_vi, _, stage_obj_val = safe_get_stage_info(res)
        detections = res.get("detections", []) if isinstance(res.get("detections"), list) else []
        classifier_conf = pipeline_out["classifier"]["conf"] if pipeline_out.get("classifier") and isinstance(pipeline_out["classifier"], dict) else None

        # Lọc detections
        clean_detections = []
        for d in detections:
            if isinstance(d, dict):
                class_vi = (d.get("class_name_vi") or d.get("class_name") or d.get("name_vi") or "").lower()
                if "cam" not in class_vi and "orange" not in class_vi:
                    clean_detections.append(d)
            elif isinstance(d, str):
                if "cam" not in d.lower() and "orange" not in d.lower():
                    clean_detections.append({"class_name_vi": d, "conf": 1.0})

        conf_thresh = st.session_state.get("setting_conf_thresh", 0.35)
        filtered_dets = [d for d in clean_detections if d.get("conf", 1.0) >= conf_thresh]

        # Vẽ bounding box
        safe_key = result_key.replace("|", "_").replace("/", "_").replace("\\", "_")
        out_img_path = str(Path(tempfile.gettempdir()) / f"cropai_{safe_key[:50]}.jpg")
        if not os.path.exists(out_img_path):
            draw_boxes(active_target_img, filtered_dets, out_img_path)

        # LƯU PHÂN TÍCH VÀO CONVERSATION HIỆN TẠI (CHỈ LƯU 1 LẦN MỖI KHI CÓ ẢNH MỚI)
        if st.session_state.get("last_saved_result_key") != result_key:
            analysis_data = {
                "crop": current_crop_id,
                "crop_name": str(res.get("crop_name_vi") or "Cây trồng"),
                "stage": stage_vi,
                "stage_obj": stage_obj_val,
                "confidence": classifier_conf or 0.0,
                "detections": clean_detections,
                "diseases": res.get("diseases", []) if isinstance(res.get("diseases"), list) else [],
            }
            new_ana = history_store.add_analysis_to_conversation(
                current_cid,
                analysis_data,
                image_src_path=out_img_path,
                original_src_path=active_target_img,
            )
            st.session_state["last_saved_result_key"] = result_key
            st.session_state["selected_analysis_id"] = new_ana["id"]
            st.session_state["conversation_view_mode"] = "analysis"

            # Khởi tạo Gemini chào ban đầu nếu cuộc trò chuyện chưa có tin nhắn
            if st.session_state.get("setting_use_llm") and not chat_messages:
                kfile = pipe.detector.registry.get_config(current_crop_id).get("knowledge_file")
                try:
                    chat, first_msg = CropAdvisor().start_session(res, kfile)
                    st.session_state["chat"] = chat
                    init_msgs = [{"role": "assistant", "content": first_msg}]
                    history_store.update_conversation_messages(current_cid, init_msgs)
                except Exception as e:
                    print(f"[LLM ERROR] {e}")

            # Reload lại active_conv
            active_conv = history_store.get_conversation(current_cid)
            analyses_list = active_conv.get("analyses", [])
            chat_messages = active_conv.get("messages", [])


# Chỉ mở khu vực kết quả sau khi có phân tích trong phiên hiện tại,
# hoặc khi người dùng chủ động chọn một conversation từ lịch sử.
show_analysis_sections = bool(analyses_list) and st.session_state.get("conversation_view_mode") in {"history", "analysis"}

# -----------------------------------------------------------------------------
# 8. HIỂN THỊ KẾT QUẢ CÁC LẦN PHÂN TÍCH TRONG CONVERSATION
# -----------------------------------------------------------------------------

if show_analysis_sections:
    # Hiển thị từng ảnh dưới dạng thẻ có trạng thái đang chọn
    if len(analyses_list) > 1:
        with st.expander("🖼️ Ảnh đã phân tích trong cuộc trò chuyện", expanded=False):
            with st.container(key="gallery_scroll_area"):
                gallery_columns = min(len(analyses_list), 4)
                for row_start in range(0, len(analyses_list), gallery_columns):
                    row_analyses = analyses_list[row_start:row_start + gallery_columns]
                    row_columns = st.columns(gallery_columns)
                    for col_idx, ana in enumerate(row_analyses):
                        idx = row_start + col_idx
                        with row_columns[col_idx]:
                            is_selected = (
                                st.session_state.get("selected_analysis_id") == ana["id"]
                                or (st.session_state.get("selected_analysis_id") is None and idx == len(analyses_list) - 1)
                            )
                            stg_display, _, _ = safe_get_stage_info(ana)
                            crop_display = str(ana.get("crop_name") or "Cây")
                            with st.container(key=f"gallery_card_{'active_' if is_selected else ''}{ana['id']}"):
                                thumb_col, detail_col = st.columns([1, 3], vertical_alignment="center")
                                thumb_path = ana.get("image_path") or ana.get("source_image_path")
                                if not (thumb_path and os.path.exists(thumb_path)):
                                    thumb_path = ana.get("source_image_path")
                                is_sample_thumbnail = False
                                if not (thumb_path and os.path.exists(thumb_path)):
                                    thumb_path = get_legacy_analysis_image_path(ana)
                                if not (thumb_path and os.path.exists(thumb_path)):
                                    thumb_path = get_sample_thumbnail_path(ana)
                                    is_sample_thumbnail = bool(thumb_path)
                                with thumb_col:
                                    if thumb_path and os.path.exists(thumb_path):
                                        try:
                                            with Image.open(thumb_path) as thumb_image:
                                                st.image(thumb_image.copy(), width=70)
                                        except (OSError, ValueError):
                                            st.markdown("<div class='thumb-placeholder'>🖼<small>Không tải được</small></div>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("<div class='thumb-placeholder'>🖼<small>Chưa có ảnh</small></div>", unsafe_allow_html=True)
                                with detail_col:
                                    thumbnail_note = "🖼 Ảnh mẫu minh họa" if is_sample_thumbnail else f"✓ Đã phân tích · Ảnh {idx + 1}"
                                    st.markdown(
                                        f"<div class='gallery-title'>{html.escape(crop_display)}</div>"
                                        f"<div class='gallery-stage'>{html.escape(stg_display)}</div>"
                                        f"<div class='gallery-status'>{html.escape(thumbnail_note)}</div>",
                                        unsafe_allow_html=True,
                                    )
                                st.button(
                                    f"Xem kết quả: {crop_display}",
                                    key=f"btn_select_img_{ana['id']}_{idx}",
                                    use_container_width=True,
                                    on_click=select_analysis,
                                    args=(ana["id"],),
                                )

    # Xác định phân tích đang được chọn để hiển thị (mặc định là phân tích mới nhất)
    sel_id = st.session_state.get("selected_analysis_id")
    target_analysis = None
    if sel_id:
        for a in analyses_list:
            if a["id"] == sel_id:
                target_analysis = a
                break
    if not target_analysis:
        target_analysis = analyses_list[-1]

    # Dữ liệu của phân tích đang chọn (an toàn tuyệt đối)
    cur_crop_id = str(target_analysis.get("crop") or "")
    cur_crop_name = str(target_analysis.get("crop_name") or "Cây trồng")
    cur_stage_name, cur_stage_key, _ = safe_get_stage_info(target_analysis)
    cur_conf = float(target_analysis.get("confidence") or 0.0)
    cur_detections = target_analysis.get("detections") if isinstance(target_analysis.get("detections"), list) else []
    cur_diseases = target_analysis.get("diseases") if isinstance(target_analysis.get("diseases"), list) else []
    cur_img_path = target_analysis.get("image_path")
    if not (cur_img_path and os.path.exists(cur_img_path)):
        cur_img_path = get_legacy_analysis_image_path(target_analysis)
    source_img_path = target_analysis.get("source_image_path")
    if not (source_img_path and os.path.exists(source_img_path)) and cur_img_path and os.path.exists(cur_img_path):
        source_img_path = get_sample_thumbnail_path(target_analysis)

    # 1. Health Status Banner
    health_info = compute_health_status(target_analysis, cur_conf)
    st.html(
        f"""
        <div class="health-banner" style="background:{health_info['bg']}; color:{health_info['fg']};">
            <span class="icon">{health_info['icon']}</span>
            <div>
                <p class="title">{health_info['title']}</p>
                <p class="desc">{health_info['desc']}</p>
            </div>
        </div>
        """
    )

    # 2. Stage timeline
    stage_sequence = get_stage_order(cur_crop_id)

    # 3. Hai cột độc lập: Ảnh & Thống kê
    col_img, col_metrics = st.columns([5, 5], gap="large")

    with col_img:
        col_head1, col_head2 = st.columns([7, 3])
        with col_head1:
            st.subheader("📸 Kết quả Phân tích Hình ảnh")
        with col_head2:
            with st.container(key="crop_toggle_btn"):
                if st.button(
                    "✂️ Cắt",
                    key=f"btn_crop_image_{target_analysis['id']}",
                    help="Cắt vùng chọn để lấy vùng đó đi phân tích",
                    use_container_width=True,
                ):
                    st.session_state.is_cropping = not st.session_state.is_cropping

        bbox_toggle_key = f"show_bbox_{target_analysis['id']}"
        if bbox_toggle_key not in st.session_state:
            st.session_state[bbox_toggle_key] = True
        show_bounding_boxes = st.toggle(
            "Hiện Bounding Box",
            key=bbox_toggle_key,
            help="Bật/tắt các khung nhận diện trên ảnh kết quả.",
            disabled=not (
                cur_img_path and os.path.exists(cur_img_path)
                and source_img_path and os.path.exists(source_img_path)
            ),
        )
        if not (source_img_path and os.path.exists(source_img_path)):
            st.caption("Phân tích cũ chỉ lưu ảnh kết quả có khung nhận diện; các phân tích mới sẽ hỗ trợ ẩn/hiện khung.")

        # SPOTLIGHT CROP SELECTOR
        crop_source_path = source_img_path if source_img_path and os.path.exists(source_img_path) else cur_img_path
        if st.session_state.get("is_cropping") and crop_source_path and os.path.exists(crop_source_path):
            st.info("💡 **Chế độ khoanh vùng:** Di chuột chỉnh vùng sáng, click để cố định, kéo góc đổi kích thước.")
            import base64
            url_cache_key = f"crop_{Path(crop_source_path).name}"
            if st.session_state.get("spotlight_url_key") != url_cache_key:
                with open(crop_source_path, "rb") as f:
                    img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                mime_type = "image/png" if crop_source_path.endswith(".png") else "image/jpeg"
                st.session_state["spotlight_image_url"] = f"data:{mime_type};base64,{img_b64}"
                st.session_state["spotlight_url_key"] = url_cache_key

            spotlight_res = select_spotlight(
                image_url=st.session_state["spotlight_image_url"],
                regions=st.session_state.get("spotlight_regions", []),
                active_index=st.session_state.get("spotlight_active_index", -1),
                key=f"spotlight_{url_cache_key}"
            )
            if spotlight_res:
                if spotlight_res.get("regions") is not None:
                    st.session_state["spotlight_regions"] = spotlight_res["regions"]
                if isinstance(spotlight_res.get("active_index"), int):
                    st.session_state["spotlight_active_index"] = spotlight_res["active_index"]

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🎯 Phân tích vùng chọn", type="primary", use_container_width=True):
                    regs = st.session_state.get("spotlight_regions", [])
                    act_i = st.session_state.get("spotlight_active_index", -1)
                    target_reg = regs[act_i] if 0 <= act_i < len(regs) else (regs[-1] if regs else None)
                    if not target_reg:
                        st.warning("⚠️ Hãy click trên ảnh để đặt vùng trước khi phân tích.")
                    else:
                        cropped_pil = crop_image_from_region(crop_source_path, target_reg)
                        crop_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        cropped_pil.save(crop_tmp.name, quality=95)
                        crop_tmp.close()
                        st.session_state["cropped_img_path"] = crop_tmp.name
                        st.session_state["is_cropping"] = False
                        st.session_state["last_file_id"] = f"crop_{int(time.time())}"
                        st.rerun()
            with c2:
                if st.button("❌ Hủy", use_container_width=True):
                    reset_cropped_state()
                    st.rerun()
            with c3:
                if st.button("🔄 Chọn lại", use_container_width=True):
                    st.session_state["spotlight_regions"] = []
                    st.session_state["spotlight_active_index"] = -1
                    st.rerun()

        # HIỂN THỊ ẢNH THƯỜNG
        else:
            bbox_image_available = bool(cur_img_path and os.path.exists(cur_img_path))
            original_image_available = bool(source_img_path and os.path.exists(source_img_path))
            if show_bounding_boxes:
                visible_img_path = cur_img_path if bbox_image_available else source_img_path if original_image_available else None
            else:
                visible_img_path = source_img_path if original_image_available else cur_img_path if bbox_image_available else None

            if visible_img_path:
                st.image(
                    Image.open(visible_img_path),
                    caption=f"{cur_crop_name} - {cur_stage_name}",
                    use_container_width=True,
                )
            else:
                st.info("Ảnh phân tích của đối tượng này đã được lưu dạng số liệu.")

    with col_metrics:
        st.subheader("📊 Thống kê Sinh trưởng")
        if stage_sequence:
            render_stage_timeline(stage_sequence, cur_stage_name)

        conf_display = f"{cur_conf*100:.1f}%" if cur_conf > 0 else "Đã lưu"
        health_badge_class = "badge-danger" if health_info["fg"] == "#c62828" else "badge-warning" if health_info["fg"] == "#f9a825" else "badge-good"
        st.markdown(
            "<div class='analysis-badges'>"
            f"<span class='analysis-badge badge-confidence'>Độ tin cậy · {html.escape(conf_display)}</span>"
            f"<span class='analysis-badge badge-stage'>Giai đoạn · {html.escape(cur_stage_name)}</span>"
            f"<span class='analysis-badge {health_badge_class}'>{html.escape(health_info['icon'])} {html.escape(health_info['title'])}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Loài cây trồng:** {html.escape(cur_crop_name)}")

        st.write("**Chi tiết số lượng phát hiện (Tầng 2):**")
        if cur_detections:
            det_counts = {}
            for d in cur_detections:
                if isinstance(d, dict):
                    c_name = d.get("class_name_vi") or d.get("class_name") or d.get("name_vi") or "Đối tượng"
                else:
                    c_name = str(d)
                det_counts[c_name] = det_counts.get(c_name, 0) + 1
            metric_cols = st.columns(min(len(det_counts), 3))
            for metric_idx, (detection_name, detection_count) in enumerate(det_counts.items()):
                normalized_name = detection_name.lower()
                if any(term in normalized_name for term in ("bệnh", "sâu", "hại", "disease", "pest")):
                    detection_icon = "🛡️"
                elif any(term in normalized_name for term in ("quả", "trái", "buồng", "fruit")):
                    detection_icon = "🍃"
                elif any(term in normalized_name for term in ("hoa", "bông", "flower")):
                    detection_icon = "🌼"
                else:
                    detection_icon = "🌱"

                with metric_cols[metric_idx % len(metric_cols)]:
                    with st.container(key=f"detection_metric_{metric_idx}"):
                        st.markdown(f"<div class='detection-icon'>{detection_icon}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='detection-name'>{html.escape(detection_name)}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='detection-count'>{detection_count}</div>", unsafe_allow_html=True)
        else:
            st.caption("Không phát hiện thêm dấu hiệu chi tiết nào vượt ngưỡng tin cậy.")

        if cur_diseases:
            dis_names = []
            for d in cur_diseases:
                if isinstance(d, dict):
                    dis_names.append(d.get("name_vi") or d.get("name") or "Bệnh")
                else:
                    dis_names.append(str(d))
            st.error("⚠️ **Cảnh báo bệnh hại:** " + ", ".join(dis_names))

if show_analysis_sections:
    # -----------------------------------------------------------------------------
    # 9. TABS: AI TƯ VẤN TƯƠNG TÁC & CẨM NANG KỸ THUẬT
    # -----------------------------------------------------------------------------

    st.divider()

    tab_ai, tab_guide = st.tabs(["🤖 AI Tư vấn Tương tác", "📚 Cẩm nang Kỹ thuật Sinh trưởng"])

    with tab_ai:
        if st.session_state.get("setting_use_llm", True):
            # Khôi phục ChatSession nếu chưa có
            if st.session_state.get("chat") is None and chat_messages:
                history_tuples = [
                    ("Người dùng" if m.get("role") == "user" else "Trợ lý", m.get("content", ""))
                    for m in chat_messages
                ]
                # Context tổng hợp tất cả các cây trong phiên
                summary_crops = ", ".join(set(a.get("crop_name", "cây") for a in analyses_list)) or "cây trồng"
                combined_context = f"Tư vấn cho nông dân về các cây trồng trong phiên: {summary_crops}."
                st.session_state["chat"] = CropAdvisor().resume_session(combined_context, history_tuples)

            # Hiển thị toàn bộ lịch sử trao đổi trong conversation
            if chat_messages:
                for msg in chat_messages:
                    role = msg.get("role", "assistant")
                    avatar = "👤" if role == "user" else "🌱"
                    with st.chat_message(role, avatar=avatar):
                        st.markdown(msg.get("content", ""))
            else:
                st.caption("Chưa có tin nhắn nào. Bạn có thể hỏi bất kỳ thắc mắc nào về phân bón, tưới tiêu hoặc bệnh hại ở ô bên dưới.")

            # Ô chat_input đặt ở cuối
            user_question = st.chat_input("Hỏi AI thêm về chế độ phân bón, tưới nước hoặc bệnh hại cho cây...")
            if user_question:
                chat_messages.append({"role": "user", "content": user_question})

                with st.spinner("AI đang phân tích câu hỏi..."):
                    try:
                        if st.session_state.get("chat") is None:
                            # Dựng session mới nếu chưa có
                            summary_crops = ", ".join(set(a.get("crop_name", "cây") for a in analyses_list)) or "cây trồng"
                            st.session_state["chat"] = CropAdvisor().resume_session(
                                f"Chuyên gia nông nghiệp tư vấn cho nông dân về: {summary_crops}.",
                                []
                            )
                        ai_reply = st.session_state["chat"].ask(user_question)
                    except Exception as e:
                        ai_reply = f"Lỗi phản hồi từ mô hình AI: {str(e)[:200]}"

                chat_messages.append({"role": "assistant", "content": ai_reply})
                history_store.update_conversation_messages(current_cid, chat_messages)
                st.rerun()

        else:
            st.info("💡 Bạn đã tắt tính năng Tư vấn bằng AI. Bấm vào icon ⚙️ Cài đặt ở thanh bên trái để bật lại.")

    with tab_guide:
        st.markdown("### 📖 Trích xuất Cẩm nang Kỹ thuật")
        if analyses_list and target_analysis:
            sel_c_id = str(target_analysis.get("crop") or "")
            sel_s_vi, _, _ = safe_get_stage_info(target_analysis)
            st.markdown(get_stage_knowledge(sel_c_id, sel_s_vi))
        else:
            st.info("Cẩm nang kỹ thuật sẽ hiển thị chi tiết khi bạn phân tích loại cây trồng trong ảnh.")
