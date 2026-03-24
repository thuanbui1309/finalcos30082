"""Face Recognition Attendance System — Streamlit GUI."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Face Attendance System",
    layout="wide",
)

WEIGHTS  = ROOT / "weights"
DB_DIR   = ROOT / "face_db"
DATA_DIR = ROOT / "data" / "classification_data"

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
        "arcface":    WEIGHTS / "face_classification.pth",
        "triplet":    WEIGHTS / "face_metric_learning.pth",
    }
    wp = weight_map[model_type]
    if not wp.exists():
        st.warning(f"Weights not found: {wp.name}. Verification disabled.")
        return None
    return FaceVerifier(model_type=model_type, weights_path=str(wp), device="cpu")


@st.cache_resource
def load_liveness():
    from src.core.liveness_checker import LivenessChecker
    return LivenessChecker(device="cpu")


@st.cache_resource
def load_emotion():
    from src.core.emotion_recognizer import EmotionRecognizer
    return EmotionRecognizer()


@st.cache_resource
def load_database():
    from src.core.face_database import FaceDatabase
    return FaceDatabase(db_dir=str(DB_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def uploaded_to_bgr(uploaded_file) -> np.ndarray:
    return pil_to_bgr(Image.open(uploaded_file))

def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def draw_face_box(img_bgr, bbox, label, color=(0, 200, 0)):
    out = img_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def run_full_pipeline(img_bgr, verifier, liveness, emotion_rec, database,
                      metric, verify_threshold, top_k=5):
    """Detect face then run liveness, top-k identity, and emotion.

    Returns dict with keys: face_rgb, bbox, annotated_bgr,
    is_real, live_conf, top_matches, emotion_scores, dominant_emotion.
    """
    detector = load_detector()
    faces = detector.detect_and_crop(img_bgr)
    if not faces:
        return None

    face_rgb, info = faces[0]
    bbox     = info["bbox"]
    annotated = img_bgr.copy()

    # Liveness
    is_real, live_conf = liveness.check(face_rgb)
    live_color = (0, 200, 0) if is_real else (0, 0, 220)
    live_label = "Real" if is_real else "Fake"
    annotated  = draw_face_box(annotated, bbox, live_label, live_color)

    # Top-k identity
    top_matches = []
    if verifier is not None:
        emb         = verifier.extract_embedding(face_rgb)
        top_matches = database.search(emb, metric=metric, threshold=0.0)[:top_k]

    # Emotion — all scores
    emotion_scores   = {}
    dominant_emotion = "N/A"
    if emotion_rec is not None:
        emotion_scores   = emotion_rec.recognize_all(face_rgb)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "N/A"

    return dict(
        face_rgb=face_rgb,
        bbox=bbox,
        annotated_bgr=annotated,
        is_real=is_real,
        live_conf=live_conf,
        top_matches=top_matches,
        emotion_scores=emotion_scores,
        dominant_emotion=dominant_emotion,
    )


def show_results(res: dict, verify_threshold: float):
    """Render the unified results panel."""
    st.image(to_rgb(res["annotated_bgr"]), use_container_width=True)

    st.divider()

    # --- Liveness ---
    st.subheader("Liveness")
    col_l1, col_l2 = st.columns([1, 3])
    status = "Real" if res["is_real"] else "Fake"
    col_l1.metric("Status", status)
    col_l2.progress(res["live_conf"], text=f"Confidence: {res['live_conf']:.1%}")

    st.divider()

    # --- Top-K Identity ---
    st.subheader("Top Identity Matches")
    matches = res["top_matches"]
    if not matches:
        st.info("No faces registered in database yet.")
    else:
        df = pd.DataFrame(matches)[["name", "score"]].copy()
        df.columns = ["Name", "Score"]
        df["Score"] = df["Score"].round(4)
        # Horizontal bar chart
        st.bar_chart(df.set_index("Name")["Score"])
        # Mark best match
        best = matches[0]
        if best["score"] >= verify_threshold:
            st.success(f'Best match: {best["name"]}  (score {best["score"]:.4f})')
        else:
            st.warning(f'No match above threshold {verify_threshold:.2f}. '
                       f'Closest: {best["name"]} ({best["score"]:.4f})')

    st.divider()

    # --- Emotion ---
    st.subheader("Emotion")
    scores = res["emotion_scores"]
    if scores:
        emo_df = pd.DataFrame(
            sorted(scores.items(), key=lambda x: -x[1]),
            columns=["Emotion", "Confidence"]
        ).set_index("Emotion")
        st.bar_chart(emo_df)
        st.metric("Dominant", res["dominant_emotion"].capitalize())
    else:
        st.info("Emotion model not available.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Face Attendance System")

with st.sidebar.expander("Model Settings", expanded=False):
    model_type        = st.selectbox("Embedding model", ["classifier", "arcface", "triplet"])
    metric            = st.selectbox("Distance metric", ["cosine", "euclidean"])
    verify_threshold  = st.slider("Verification threshold", 0.0, 1.0, 0.50, 0.01)
    top_k             = st.slider("Top-K matches", 1, 10, 5)

mode = st.sidebar.radio(
    "Mode",
    ["Attendance (Verify)", "Demo (Dataset Image)", "Register New Face", "View Database"],
)

if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = []

# Pre-load always-on models
liveness    = load_liveness()
emotion_rec = load_emotion()
database    = load_database()

# ---------------------------------------------------------------------------
# Mode: Attendance
# ---------------------------------------------------------------------------
if mode == "Attendance (Verify)":
    st.title("Attendance — Face Verification")

    verifier = load_verifier(model_type)
    col_cam, col_res = st.columns([1, 1])

    with col_cam:
        captured = st.camera_input("Capture your face")

    with col_res:
        if captured is not None:
            img_bgr = uploaded_to_bgr(captured)
            res     = run_full_pipeline(img_bgr, verifier, liveness, emotion_rec,
                                        database, metric, verify_threshold, top_k)
            if res is None:
                st.error("No face detected. Please try again.")
            else:
                show_results(res, verify_threshold)

                best_name = "Unknown"
                best_score = 0.0
                if res["top_matches"] and res["top_matches"][0]["score"] >= verify_threshold:
                    best_name  = res["top_matches"][0]["name"]
                    best_score = res["top_matches"][0]["score"]

                st.session_state.attendance_log.insert(0, {
                    "Time":     datetime.datetime.now().strftime("%H:%M:%S"),
                    "Identity": best_name,
                    "Score":    f"{best_score:.4f}",
                    "Emotion":  res["dominant_emotion"],
                    "Liveness": "Real" if res["is_real"] else "Fake",
                })

    if st.session_state.attendance_log:
        st.subheader("Attendance Log")
        st.dataframe(pd.DataFrame(st.session_state.attendance_log),
                     use_container_width=True)
        if st.button("Clear log"):
            st.session_state.attendance_log = []
            st.rerun()

# ---------------------------------------------------------------------------
# Mode: Demo
# ---------------------------------------------------------------------------
elif mode == "Demo (Dataset Image)":
    st.title("Demo — Dataset Image Analysis")
    st.caption("Pick any image from the dataset to run face recognition, liveness, and emotion.")

    verifier = load_verifier(model_type)

    # Build image list from val_data (faster than train)
    splits = []
    for split in ["val_data", "train_data", "test_data"]:
        d = DATA_DIR / split
        if d.exists():
            splits.append(split)

    chosen_split = st.selectbox("Dataset split", splits)
    split_dir    = DATA_DIR / chosen_split

    # List identity folders
    id_folders = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    if not id_folders:
        st.warning("No identity folders found.")
        st.stop()

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        chosen_id = st.selectbox("Identity folder", id_folders)
    with col_sel2:
        id_dir    = split_dir / chosen_id
        img_files = sorted([f.name for f in id_dir.glob("*.jpg")])
        chosen_img = st.selectbox("Image", img_files)

    img_path = id_dir / chosen_img

    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(str(img_path), caption=f"{chosen_id} / {chosen_img}",
                 use_container_width=True)

    with col_res:
        if st.button("Run Analysis", type="primary"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                st.error("Failed to load image.")
            else:
                with st.spinner("Running pipeline..."):
                    res = run_full_pipeline(img_bgr, verifier, liveness, emotion_rec,
                                            database, metric, verify_threshold, top_k)
                if res is None:
                    st.error("No face detected in this image.")
                else:
                    show_results(res, verify_threshold)

# ---------------------------------------------------------------------------
# Mode: Register
# ---------------------------------------------------------------------------
elif mode == "Register New Face":
    st.title("Register New Face")

    verifier = load_verifier(model_type)
    col_form, col_preview = st.columns([1, 1])

    with col_form:
        name     = st.text_input("Full name", placeholder="e.g. Nguyen Van A")
        captured = st.camera_input("Capture face for registration")

    with col_preview:
        if captured and name.strip():
            detector = load_detector()
            img_bgr  = uploaded_to_bgr(captured)
            faces    = detector.detect_and_crop(img_bgr)

            if not faces:
                st.error("No face detected. Retake photo.")
            elif verifier is None:
                st.error("Verifier model not loaded. Place weights in final/weights/.")
            else:
                face_rgb, info = faces[0]
                annotated = draw_face_box(img_bgr, info["bbox"], name)
                st.image(to_rgb(annotated), caption="Detected face",
                         use_container_width=True)

                if st.button("Confirm Registration", type="primary"):
                    emb     = verifier.extract_embedding(face_rgb)
                    face_id = database.register(name.strip(), emb, face_rgb)
                    st.success(f"Registered '{name}' (ID: {face_id[:8]}...)")
        elif captured and not name.strip():
            st.warning("Please enter a name before registering.")

# ---------------------------------------------------------------------------
# Mode: View Database
# ---------------------------------------------------------------------------
elif mode == "View Database":
    st.title("Face Database")

    entries = database.list_all()
    if not entries:
        st.info("No faces registered yet.")
    else:
        st.write(f"{len(entries)} registered identities")
        for entry in entries:
            c1, c2, c3, c4 = st.columns([2, 3, 3, 1])
            c1.write(entry["name"])
            c2.write(str(entry["face_id"])[:16] + "...")
            c3.write(str(entry["registered_at"]))
            if c4.button("Delete", key=f"del_{entry['face_id']}"):
                database.delete(entry["face_id"])
                st.success(f"Deleted {entry['name']}")
                st.rerun()
