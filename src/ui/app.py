"""Face Recognition Attendance System — Streamlit GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src.xxx import` when launched from final/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎓",
    layout="wide",
)

WEIGHTS = ROOT / "weights"
DB_DIR = ROOT / "face_db"

# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_detector():
    from src.core.face_detector import FaceDetector
    return FaceDetector(device="cpu")


@st.cache_resource
def load_verifier(model_type: str):
    from src.core.face_verifier import FaceVerifier

    weight_map = {
        "classifier": WEIGHTS / "face_classification.pth",
        "arcface": WEIGHTS / "face_classification.pth",
        "triplet": WEIGHTS / "face_metric_learning.pth",
    }
    wp = weight_map[model_type]
    if not wp.exists():
        st.warning(f"⚠️ Weights not found: {wp.name}. Verification disabled.")
        return None
    return FaceVerifier(model_type=model_type, weights_path=str(wp), device="cpu")


@st.cache_resource
def load_liveness():
    from src.core.liveness_checker import LivenessChecker

    wp = WEIGHTS / "anti_spoofing.pth"
    if not wp.exists():
        st.warning(f"⚠️ Weights not found: {wp.name}. Liveness check disabled.")
        return None
    return LivenessChecker(weights_path=str(wp), device="cpu")


@st.cache_resource
def load_emotion():
    from src.core.emotion_recognizer import EmotionRecognizer

    wp = WEIGHTS / "emotion_detection.pth"
    if not wp.exists():
        st.warning(f"⚠️ Weights not found: {wp.name}. Emotion detection disabled.")
        return None
    return EmotionRecognizer(weights_path=str(wp), device="cpu")


@st.cache_resource
def load_database():
    from src.core.face_database import FaceDatabase
    return FaceDatabase(db_dir=str(DB_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def uploaded_to_bgr(uploaded_file) -> np.ndarray:
    pil = Image.open(uploaded_file).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_face_box(
    image_bgr: np.ndarray,
    bbox: list,
    label: str,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    img = image_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎓 Face Attendance")

with st.sidebar.expander("⚙️ Model Settings", expanded=False):
    model_type = st.selectbox("Model type", ["classifier", "arcface", "triplet"], index=0)
    metric = st.selectbox("Distance metric", ["cosine", "euclidean"], index=0)
    verify_threshold = st.slider("Verification threshold", 0.0, 1.0, 0.50, 0.01)
    liveness_threshold = st.slider("Liveness threshold", 0.0, 1.0, 0.50, 0.01)

mode = st.sidebar.radio(
    "Mode",
    ["🕵️ Attendance (Verify)", "➕ Register New Face", "🗂️ View Database"],
    index=0,
)

if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = []

# ---------------------------------------------------------------------------
# Mode 1: Attendance
# ---------------------------------------------------------------------------
if mode == "🕵️ Attendance (Verify)":
    st.title("🕵️ Attendance — Face Verification")
    col_cam, col_result = st.columns([1, 1])

    with col_cam:
        captured = st.camera_input("📷 Capture your face")

    with col_result:
        if captured is not None:
            detector = load_detector()
            verifier = load_verifier(model_type)
            liveness = load_liveness()
            emotion_rec = load_emotion()
            database = load_database()

            img_bgr = uploaded_to_bgr(captured)
            faces = detector.detect_and_crop(img_bgr)

            if not faces:
                st.error("❌ No face detected. Please try again.")
            else:
                face_rgb, info = faces[0]
                bbox = info["bbox"]
                annotated = img_bgr.copy()

                # Liveness
                is_real, live_conf = True, 1.0
                if liveness is not None:
                    is_real, live_conf = liveness.check(face_rgb)

                if not is_real and live_conf > liveness_threshold:
                    annotated = draw_face_box(annotated, bbox, "FAKE", (0, 0, 255))
                    st.image(to_rgb(annotated))
                    st.error(f"🚫 Spoofed face detected! (conf: {live_conf:.1%})")
                else:
                    # Identity
                    identity, id_score = "Unknown", 0.0
                    if verifier is not None:
                        emb = verifier.extract_embedding(face_rgb)
                        identity, id_score = database.identify(
                            emb, metric=metric, threshold=verify_threshold
                        )
                        if identity is None:
                            identity, id_score = "Unknown", 0.0

                    box_color = (0, 255, 0) if identity != "Unknown" else (0, 165, 255)
                    annotated = draw_face_box(annotated, bbox, identity, box_color)

                    # Emotion
                    emo_label, emo_conf, emo_icon = "N/A", 0.0, ""
                    if emotion_rec is not None:
                        emo_label, emo_conf = emotion_rec.recognize(face_rgb)
                        emo_icon = emotion_rec.EMOTION_ICONS.get(emo_label, "")

                    st.image(to_rgb(annotated))

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Identity", identity)
                    m2.metric("Match Score", f"{id_score:.3f}" if id_score else "—")
                    m3.metric("Emotion", f"{emo_icon} {emo_label}")
                    st.metric("Liveness", f"{'✅ Real' if is_real else '❌ Fake'} ({live_conf:.1%})")

                    import datetime
                    st.session_state.attendance_log.insert(0, {
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Identity": identity,
                        "Score": f"{id_score:.3f}" if id_score else "—",
                        "Emotion": f"{emo_icon} {emo_label}",
                        "Liveness": "Real" if is_real else "Fake",
                    })

    if st.session_state.attendance_log:
        st.subheader("📋 Attendance Log")
        import pandas as pd
        st.dataframe(pd.DataFrame(st.session_state.attendance_log), use_container_width=True)
        if st.button("🗑️ Clear log"):
            st.session_state.attendance_log = []
            st.rerun()

# ---------------------------------------------------------------------------
# Mode 2: Register
# ---------------------------------------------------------------------------
elif mode == "➕ Register New Face":
    st.title("➕ Register New Face")
    col_form, col_preview = st.columns([1, 1])

    with col_form:
        name = st.text_input("Full name", placeholder="e.g. Nguyen Van A")
        captured = st.camera_input("📷 Capture face for registration")

    with col_preview:
        if captured and name.strip():
            detector = load_detector()
            verifier = load_verifier(model_type)
            database = load_database()

            img_bgr = uploaded_to_bgr(captured)
            faces = detector.detect_and_crop(img_bgr)

            if not faces:
                st.error("❌ No face detected. Retake photo.")
            elif verifier is None:
                st.error("❌ Verifier model not loaded. Place weights in final/weights/.")
            else:
                face_rgb, info = faces[0]
                annotated = draw_face_box(img_bgr, info["bbox"], name, (0, 255, 0))
                st.image(to_rgb(annotated), caption="Detected face")

                if st.button("✅ Confirm Registration"):
                    emb = verifier.extract_embedding(face_rgb)
                    face_id = database.register(name.strip(), emb, face_rgb)
                    st.success(f"Registered **{name}** (ID: `{face_id[:8]}...`)")
        elif captured and not name.strip():
            st.warning("Please enter a name before registering.")

# ---------------------------------------------------------------------------
# Mode 3: View Database
# ---------------------------------------------------------------------------
elif mode == "🗂️ View Database":
    st.title("🗂️ Face Database")
    database = load_database()
    entries = database.list_all()

    if not entries:
        st.info("No faces registered yet.")
    else:
        st.write(f"**{len(entries)}** registered identities")
        import pandas as pd

        df = pd.DataFrame(entries)[["name", "face_id", "registered_at"]]
        for _, row in df.iterrows():
            c1, c2, c3, c4 = st.columns([2, 3, 3, 1])
            c1.write(row["name"])
            c2.write(str(row["face_id"])[:16] + "...")
            c3.write(str(row["registered_at"]))
            if c4.button("🗑️", key=f"del_{row['face_id']}"):
                database.delete(row["face_id"])
                st.success(f"Deleted {row['name']}")
                st.rerun()
