"""
history_store.py – Quản lý lịch sử hội thoại chuẩn ứng dụng AI:
1. Mỗi Conversation là một PHIÊN LÀM VIỆC (Vườn nhà tôi, Kiểm tra chuối...)
   - Chứa nhiều ảnh / phân tích (analyses = [analysis_1, analysis_2, ...])
   - Chứa nhiều tin nhắn trao đổi với AI (messages = [...])
   - Lưu trữ tại: <project_root>/history/<conv_id>/data.json
   - Ảnh các lần phân tích lưu tại: <project_root>/history/<conv_id>/images/
2. Tương thích ngược: Đọc được các conversation cũ và giữ SQLite (history.db).
"""

import json
import os
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Thư mục gốc project
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "history.db"
HISTORY_DIR = ROOT / "history"


# =============================================================================
# A. QUẢN LÝ LỊCH SỬ DẠNG FILE / FOLDER (MULTI-IMAGE CONVERSATION)
# =============================================================================

def init_history_storage():
    """Khởi tạo thư mục history/ nếu chưa tồn tại."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def format_relative_time(iso_str: str) -> str:
    """Định dạng thời gian thân thiện (2 phút trước, 1 giờ trước, Hôm qua...)."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now()
        diff = now - dt

        seconds = diff.total_seconds()
        if seconds < 60:
            return "Vừa xong"
        elif seconds < 3600:
            return f"{int(seconds // 60)} phút trước"
        elif dt.date() == now.date():
            return f"{int(seconds // 3600)} giờ trước"
        elif dt.date() == (now - timedelta(days=1)).date():
            return "Hôm qua"
        elif diff.days < 7:
            return f"{diff.days} ngày trước"
        else:
            return dt.strftime("%d/%m/%Y")
    except Exception:
        return iso_str[:10]


def generate_default_title(crop_id: str, crop_name_vi: str, stage_name_vi: str = "") -> str:
    """Tự động sinh tiêu đề trực quan với icon nông nghiệp."""
    crop_id_clean = (crop_id or "").lower()
    crop_name_vi_clean = (crop_name_vi or "").strip()
    stage_name_vi_clean = (stage_name_vi or "").strip()

    emoji = "🌱"
    if "chuoi" in crop_id_clean or "banana" in crop_id_clean:
        emoji = "🍌"
    elif "dua" in crop_id_clean or "coconut" in crop_id_clean:
        emoji = "🥥"
    elif "dudu" in crop_id_clean or "papaya" in crop_id_clean:
        emoji = "🥭"
    elif "xoai" in crop_id_clean or "mango" in crop_id_clean:
        emoji = "🥭"
    elif "cam" in crop_id_clean or "orange" in crop_id_clean:
        emoji = "🍊"
    elif "buoi" in crop_id_clean or "pomelo" in crop_id_clean:
        emoji = "🍈"

    name = crop_name_vi_clean or "Cây trồng"
    if stage_name_vi_clean and stage_name_vi_clean.lower() != "chưa xác định":
        return f"{emoji} {name} - {stage_name_vi_clean}"
    return f"{emoji} {name}"


def create_conversation(
    title: str = "Cuộc trò chuyện mới",
    conv_id: str = None,
    crop_id: str = None,
    crop_name_vi: str = None,
    stage_name_vi: str = None,
    stage: dict = None,
    confidence: float = 0.0,
    detections: list = None,
    diseases: list = None,
    context: str = "",
    messages: list = None,
    image_src_path: str = None,
) -> str:
    """
    Tạo một cuộc trò chuyện mới.
    Chỉ được gọi khi người dùng bấm 'Chat mới' hoặc khi app khởi tạo phiên đầu tiên.
    """
    init_history_storage()

    if not conv_id:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_id = f"conv_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    conv_folder = HISTORY_DIR / conv_id
    images_folder = conv_folder / "images"
    conv_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now().isoformat(timespec="seconds")

    analyses = []
    # Nếu được gọi cùng ảnh/kết quả ban đầu (backward compatibility)
    if crop_id or image_src_path:
        ana_id = f"ana_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        saved_img_rel = None
        if image_src_path and os.path.exists(image_src_path):
            suffix = Path(image_src_path).suffix or ".jpg"
            dest_img = images_folder / f"{ana_id}{suffix}"
            try:
                shutil.copy2(image_src_path, dest_img)
                saved_img_rel = str(dest_img)
            except Exception:
                pass

        analyses.append({
            "id": ana_id,
            "image_path": saved_img_rel or image_src_path,
            "crop": crop_id or "",
            "crop_name": crop_name_vi or "",
            "stage": stage_name_vi or "",
            "stage_obj": stage or {},
            "confidence": float(confidence or 0.0),
            "detections": detections or [],
            "diseases": diseases or [],
            "created_at": now_iso,
        })
        if title == "Cuộc trò chuyện mới" and crop_id:
            title = generate_default_title(crop_id, crop_name_vi, stage_name_vi)

    conv_data = {
        "id": conv_id,
        "title": title or "Cuộc trò chuyện mới",
        "created_at": now_iso,
        "updated_at": now_iso,
        "messages": messages or [],
        "analyses": analyses,
        "context": context or "",
    }

    json_file = conv_folder / "data.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=2)

    return conv_id


def add_analysis_to_conversation(
    conv_id: str,
    analysis_data: dict,
    image_src_path: str = None,
    original_src_path: str = None,
) -> dict:
    """
    Thêm một lần phân tích ảnh vào conversation hiện tại.
    Dù người dùng upload 1 hay 100 ảnh, TẤT CẢ đều được thêm vào danh sách `analyses`
    của cuộc trò chuyện hiện tại, KHÔNG tạo conversation mới.
    """
    init_history_storage()
    conv_folder = HISTORY_DIR / conv_id
    images_folder = conv_folder / "images"
    conv_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)

    json_file = conv_folder / "data.json"
    conv_data = None
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                conv_data = json.load(f)
        except Exception:
            pass

    now_iso = datetime.now().isoformat(timespec="seconds")
    if not conv_data:
        conv_data = {
            "id": conv_id,
            "title": "Cuộc trò chuyện mới",
            "created_at": now_iso,
            "updated_at": now_iso,
            "messages": [],
            "analyses": [],
            "context": "",
        }

    if "analyses" not in conv_data:
        conv_data["analyses"] = []

    # Lưu ảnh kết quả vào history/<conv_id>/images/ana_<timestamp>_<uuid>.jpg
    ana_id = f"ana_{int(time.time())}_{uuid.uuid4().hex[:4]}"
    saved_img_rel = None
    if image_src_path and os.path.exists(image_src_path):
        suffix = Path(image_src_path).suffix or ".jpg"
        dest_img = images_folder / f"{ana_id}{suffix}"
        try:
            shutil.copy2(image_src_path, dest_img)
            saved_img_rel = str(dest_img)
        except Exception:
            saved_img_rel = image_src_path

    saved_original_rel = None
    if original_src_path and os.path.exists(original_src_path):
        original_suffix = Path(original_src_path).suffix or ".jpg"
        original_dest = images_folder / f"{ana_id}_original{original_suffix}"
        try:
            shutil.copy2(original_src_path, original_dest)
            saved_original_rel = str(original_dest)
        except Exception:
            saved_original_rel = original_src_path

    crop_id = str(analysis_data.get("crop") or analysis_data.get("crop_id") or "")
    crop_name = str(analysis_data.get("crop_name") or analysis_data.get("crop_name_vi") or "Cây trồng")

    # Xử lý stage và stage_obj an toàn tuyệt đối (hỗ trợ cả string lẫn dict)
    stg_obj = analysis_data.get("stage_obj")
    stg_val = analysis_data.get("stage")
    stg_name_vi = analysis_data.get("stage_name_vi")

    if isinstance(stg_obj, dict) and stg_obj:
        final_stage_obj = stg_obj
        final_stage_name = str(stg_val or stg_obj.get("name_vi") or stg_name_vi or "")
    elif isinstance(stg_val, dict):
        final_stage_obj = stg_val
        final_stage_name = str(stg_val.get("name_vi") or stg_name_vi or "")
    elif isinstance(stg_val, str) and stg_val.strip():
        final_stage_name = stg_val.strip()
        final_stage_obj = {"key": final_stage_name, "name_vi": final_stage_name}
    elif isinstance(stg_name_vi, str) and stg_name_vi.strip():
        final_stage_name = stg_name_vi.strip()
        final_stage_obj = {"key": final_stage_name, "name_vi": final_stage_name}
    else:
        final_stage_name = "Chưa xác định"
        final_stage_obj = {}

    if not final_stage_name:
        final_stage_name = "Chưa xác định"

    analysis_item = {
        "id": ana_id,
        "image_path": saved_img_rel,
        "source_image_path": saved_original_rel,
        "crop": crop_id,
        "crop_name": crop_name,
        "stage": final_stage_name,
        "stage_obj": final_stage_obj,
        "confidence": float(analysis_data.get("confidence") or 0.0),
        "detections": analysis_data.get("detections") if isinstance(analysis_data.get("detections"), list) else [],
        "diseases": analysis_data.get("diseases") if isinstance(analysis_data.get("diseases"), list) else [],
        "created_at": now_iso,
    }

    conv_data["analyses"].append(analysis_item)
    conv_data["updated_at"] = now_iso

    # Tự động cập nhật tiêu đề phù hợp nếu vẫn là tiêu đề mặc định
    cur_title = conv_data.get("title", "Cuộc trò chuyện mới")
    if cur_title in ["Cuộc trò chuyện mới", ""]:
        conv_data["title"] = generate_default_title(crop_id, crop_name, final_stage_name)
    else:
        # Nếu đã có nhiều loài cây khác nhau và tiêu đề chưa được đặt tên riêng
        all_crops = {a.get("crop") for a in conv_data["analyses"] if a.get("crop")}
        if len(all_crops) > 1 and not cur_title.startswith("🌱 Vườn cây"):
            # Chỉ cập nhật nếu tiêu đề trước đó là do hệ thống tự sinh
            if any(cur_title.startswith(emoji) for emoji in ["🍌", "🥥", "🥭", "🍊", "🍈"]):
                conv_data["title"] = "🌱 Vườn cây"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=2)

    return analysis_item


def update_conversation_messages(conv_id: str, messages: list) -> bool:
    """Cập nhật tin nhắn cho một cuộc trò chuyện đã có."""
    conv_folder = HISTORY_DIR / conv_id
    json_file = conv_folder / "data.json"
    if not json_file.exists():
        return False

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["messages"] = messages
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def get_conversation(conv_id: str) -> dict | None:
    """Đọc dữ liệu một conversation theo conv_id, tự động chuẩn hóa định dạng."""
    if not conv_id:
        return None
    conv_folder = HISTORY_DIR / conv_id
    json_file = conv_folder / "data.json"
    if not json_file.exists():
        return None

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Chuẩn hóa nếu là cấu trúc cũ (chưa có trường analyses)
        if "analyses" not in data or not isinstance(data["analyses"], list):
            analyses = []
            if data.get("crop_id") or data.get("crop_name_vi"):
                img_path = conv_folder / (data.get("image_file") or "image.jpg")
                stg_raw = data.get("stage")
                if isinstance(stg_raw, dict):
                    stg_name = str(stg_raw.get("name_vi") or "")
                    stg_obj = stg_raw
                elif isinstance(stg_raw, str) and stg_raw.strip():
                    stg_name = stg_raw.strip()
                    stg_obj = {"key": stg_name, "name_vi": stg_name}
                else:
                    stg_name = str(data.get("stage_name_vi") or "Chưa xác định")
                    stg_obj = {}

                analyses.append({
                    "id": "legacy_ana_1",
                    "image_path": str(img_path) if img_path.exists() else None,
                    "crop": str(data.get("crop_id") or ""),
                    "crop_name": str(data.get("crop_name_vi") or "Cây trồng"),
                    "stage": stg_name,
                    "stage_obj": stg_obj,
                    "confidence": float(data.get("confidence") or 0.0),
                    "detections": data.get("detections") if isinstance(data.get("detections"), list) else [],
                    "diseases": data.get("diseases") if isinstance(data.get("diseases"), list) else [],
                    "created_at": data.get("created_at", ""),
                })
            data["analyses"] = analyses

        # Chuẩn hóa từng analysis trong danh sách để tương thích mọi phiên bản
        for ana in data["analyses"]:
            if not isinstance(ana, dict):
                continue
            stg = ana.get("stage")
            stg_obj = ana.get("stage_obj")
            stg_vi = ana.get("stage_name_vi")

            if isinstance(stg_obj, dict) and stg_obj:
                ana["stage_obj"] = stg_obj
                ana["stage"] = str(stg if isinstance(stg, str) and stg else (stg_obj.get("name_vi") or stg_vi or "Chưa xác định"))
            elif isinstance(stg, dict):
                ana["stage_obj"] = stg
                ana["stage"] = str(stg.get("name_vi") or stg_vi or "Chưa xác định")
            elif isinstance(stg, str) and stg.strip():
                ana["stage"] = stg.strip()
                ana["stage_obj"] = {"key": stg.strip(), "name_vi": stg.strip()}
            elif isinstance(stg_vi, str) and stg_vi.strip():
                ana["stage"] = stg_vi.strip()
                ana["stage_obj"] = {"key": stg_vi.strip(), "name_vi": stg_vi.strip()}
            else:
                ana["stage"] = "Chưa xác định"
                ana["stage_obj"] = {}

            if "detections" not in ana or not isinstance(ana["detections"], list):
                ana["detections"] = []
            if "diseases" not in ana or not isinstance(ana["diseases"], list):
                ana["diseases"] = []

            # Đảm bảo đường dẫn ảnh hợp lệ
            img_p = ana.get("image_path")
            if img_p and not os.path.exists(img_p):
                cand = conv_folder / "images" / Path(img_p).name
                if cand.exists():
                    ana["image_path"] = str(cand)
                else:
                    cand_old = conv_folder / "image.jpg"
                    if cand_old.exists():
                        ana["image_path"] = str(cand_old)

        if "messages" not in data:
            data["messages"] = []

        return data
    except Exception:
        return None


def list_conversations(query: str = None) -> list[dict]:
    """
    Liệt kê tất cả conversation, sắp xếp theo updated_at mới nhất.
    Hỗ trợ tìm kiếm theo:
    - Tiêu đề (title)
    - Nội dung tin nhắn (messages)
    - Loài cây trong danh sách analyses (crop, crop_name, stage)
    """
    init_history_storage()
    if not HISTORY_DIR.exists():
        return []

    results = []
    q = query.strip().lower() if query else None

    for entry in HISTORY_DIR.iterdir():
        if entry.is_dir():
            json_file = entry / "data.json"
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Tìm kiếm nâng cao
                    if q:
                        title_match = q in (data.get("title") or "").lower()
                        msg_match = any(
                            q in (m.get("content") or "").lower()
                            for m in data.get("messages", [])
                        )

                        # Tìm trong analyses
                        analyses = data.get("analyses", [])
                        crop_match = any(
                            q in (a.get("crop") or "").lower()
                            or q in (a.get("crop_name") or "").lower()
                            or q in (a.get("stage") or "").lower()
                            for a in analyses
                        )
                        # Hỗ trợ backward compatibility
                        legacy_crop_match = (
                            q in (data.get("crop_id") or "").lower()
                            or q in (data.get("crop_name_vi") or "").lower()
                            or q in (data.get("stage_name_vi") or "").lower()
                        )

                        if not (title_match or msg_match or crop_match or legacy_crop_match):
                            continue

                    # Thêm relative time để render UI siêu mượt
                    data["relative_time"] = format_relative_time(data.get("updated_at") or data.get("created_at"))
                    results.append(data)
                except Exception:
                    continue

    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results


def rename_conversation(conv_id: str, new_title: str) -> bool:
    """Đổi tên một cuộc trò chuyện."""
    conv_folder = HISTORY_DIR / conv_id
    json_file = conv_folder / "data.json"
    if not json_file.exists():
        return False

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = new_title.strip()
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def delete_conversation(conv_id: str) -> bool:
    """Xóa vĩnh viễn một cuộc trò chuyện và thư mục ảnh của nó."""
    conv_folder = HISTORY_DIR / conv_id
    if conv_folder.exists() and conv_folder.is_dir():
        try:
            shutil.rmtree(conv_folder)
            return True
        except Exception:
            return False
    return False


def group_conversations_by_time(conversations: list[dict]) -> dict[str, list[dict]]:
    """
    Phân nhóm danh sách hội thoại theo mốc thời gian:
    - HÔM NAY
    - HÔM QUA
    - 7 NGÀY TRƯỚC
    - CŨ HƠN
    """
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)
    seven_days_ago = today - timedelta(days=7)

    groups = {
        "HÔM NAY": [],
        "HÔM QUA": [],
        "7 NGÀY TRƯỚC": [],
        "CŨ HƠN": [],
    }

    for conv in conversations:
        updated_at_str = conv.get("updated_at") or conv.get("created_at") or ""
        try:
            conv_date = datetime.fromisoformat(updated_at_str).date()
        except Exception:
            conv_date = today

        if conv_date == today:
            groups["HÔM NAY"].append(conv)
        elif conv_date == yesterday:
            groups["HÔM QUA"].append(conv)
        elif conv_date >= seven_days_ago:
            groups["7 NGÀY TRƯỚC"].append(conv)
        else:
            groups["CŨ HƠN"].append(conv)

    return {k: v for k, v in groups.items() if v}


# =============================================================================
# B. TƯƠNG THÍCH NGƯỢC SQLITE (Dành cho các hàm cũ)
# =============================================================================

def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    init_history_storage()
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ten_phien    TEXT NOT NULL,
                crop_id      TEXT,
                crop_name_vi TEXT,
                stage_name_vi TEXT,
                context      TEXT,
                history_json TEXT,
                created_at   TEXT,
                updated_at   TEXT
            )
            """
        )
        conn.commit()


def create_session(
    ten_phien: str,
    crop_id: str,
    crop_name_vi: str,
    stage_name_vi: str,
    context: str,
    history: list,
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    history_json = json.dumps(history, ensure_ascii=False)
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions
                (ten_phien, crop_id, crop_name_vi, stage_name_vi,
                 context, history_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ten_phien,
                crop_id,
                crop_name_vi,
                stage_name_vi,
                context,
                history_json,
                now,
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid


def update_history(session_id: int, history: list):
    now = datetime.now().isoformat(timespec="seconds")
    history_json = json.dumps(history, ensure_ascii=False)
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET history_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (history_json, now, session_id),
        )
        conn.commit()


def list_sessions(ten_phien: str = None) -> list:
    with _get_conn() as conn:
        if ten_phien:
            rows = conn.execute(
                """
                SELECT id, ten_phien, crop_id, crop_name_vi,
                       stage_name_vi, created_at, updated_at
                FROM sessions
                WHERE ten_phien = ?
                ORDER BY updated_at DESC
                """,
                (ten_phien,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, ten_phien, crop_id, crop_name_vi,
                       stage_name_vi, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: int) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    try:
        data["history"] = json.loads(data.get("history_json") or "[]")
    except Exception:
        data["history"] = []
    return data


def list_distinct_ten_phien() -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ten_phien
            FROM sessions
            ORDER BY ten_phien
            """
        ).fetchall()
    return [r["ten_phien"] for r in rows]
