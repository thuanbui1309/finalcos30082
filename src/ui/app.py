"""Face Recognition Attendance System — Streamlit GUI."""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from src.core.emotion_recognizer import EmotionRecognizer
from src.core.face_database import FaceDatabase
from src.core.face_detector import FaceDetector
from src.core.face_verifier import FaceVerifier
from src.core.liveness_checker import LivenessChecker

st.set_page_config(page_title="Face Attendance System", layout="wide")

WEIGHTS = ROOT / "weights"
DB_DIR = ROOT / "face_db"
DATA_DIR = ROOT / "data" / "classification_data"

MODEL_TYPES = ["classifier", "arcface", "triplet"]
WEIGHT_MAP = {
    "classifier": WEIGHTS / "face_classification.pth",
    "arcface": WEIGHTS / "face_classification_arc.pth",
    "triplet": WEIGHTS / "face_metric_learning.pth",
}


# Cached model loading
@st.cache_resource
def load_detector(device: str = "cpu"):
    return FaceDetector(device=device)

@st.cache_resource
def load_verifier(model_type: str, device: str = "cpu"):
    wp = WEIGHT_MAP[model_type]
    if not wp.exists():
        st.warning(f"Weights not found: {wp.name}. Verification disabled.")
        return None
    return FaceVerifier(model_type=model_type, weights_path=str(wp), device=device)

@st.cache_resource
def load_all_verifiers(device: str = "cpu") -> dict[str, FaceVerifier | None]:
    result = {}
    for mt in MODEL_TYPES:
        wp = WEIGHT_MAP[mt]
        if wp.exists():
            result[mt] = FaceVerifier(model_type=mt, weights_path=str(wp), device=device)
        else:
            result[mt] = None
    return result

@st.cache_resource
def load_liveness():
    return LivenessChecker()

@st.cache_resource
def load_emotion():
    return EmotionRecognizer()

@st.cache_resource
def load_database():
    return FaceDatabase(db_dir=str(DB_DIR))

# Helpers

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

def extract_all_embeddings(
    face_rgb: np.ndarray,
    all_verifiers: dict[str, FaceVerifier | None],
) -> dict[str, np.ndarray]:
    embeddings = {}
    for mt, verifier in all_verifiers.items():
        if verifier is not None:
            embeddings[mt] = verifier.extract_embedding(face_rgb)
    return embeddings

def run_full_pipeline(img_bgr, verifier, model_type, liveness, emotion_rec,
                      database, metric, verify_threshold, top_k=5):
    detector = load_detector(device)
    faces = detector.detect_and_crop(img_bgr)
    if not faces:
        return None

    face_rgb, info = faces[0]
    bbox = info["bbox"]
    annotated = img_bgr.copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if liveness is not None:
        is_real, live_conf = liveness.check(img_rgb)
    else:
        is_real, live_conf = None, None
    live_color = (0, 200, 0) if is_real else (180, 180, 180)
    live_label = ("Real" if is_real else "Fake") if is_real is not None else "Face"
    annotated = draw_face_box(annotated, bbox, live_label, live_color)

    top_matches = []
    if verifier is not None:
        emb = verifier.extract_embedding(face_rgb)
        top_matches = database.search(
            emb, model_type=model_type, metric=metric, threshold=0.0
        )[:top_k]

    emotion_scores = {}
    dominant_emotion = "N/A"
    if emotion_rec is not None:
        emotion_scores = emotion_rec.recognize_all(img_rgb)
        dominant_emotion = (
            max(emotion_scores, key=emotion_scores.get) if emotion_scores else "N/A"
        )

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
    st.image(to_rgb(res["annotated_bgr"]), use_container_width=True)
    st.divider()

    st.subheader("Liveness")
    if res["is_real"] is None:
        st.info("Liveness model not available.")
    else:
        col_l1, col_l2 = st.columns([1, 3])
        status = "Real" if res["is_real"] else "Fake"
        col_l1.metric("Status", status)
        col_l2.progress(res["live_conf"], text=f"Confidence: {res['live_conf']:.1%}")

    st.divider()

    st.subheader("Top Identity Matches")
    matches = res["top_matches"]
    if not matches:
        st.info("No faces registered in database yet.")
    else:
        df = pd.DataFrame(matches)[["name", "score"]].copy()
        df.columns = ["Name", "Score"]
        df["Score"] = df["Score"].round(4)
        st.bar_chart(df.set_index("Name")["Score"])
        best = matches[0]
        if best["score"] >= verify_threshold:
            st.success(f'Best match: {best["name"]}  (score {best["score"]:.4f})')
        else:
            st.warning(
                f'No match above threshold {verify_threshold:.2f}. '
                f'Closest: {best["name"]} ({best["score"]:.4f})'
            )

    st.divider()

    st.subheader("Emotion")
    scores = res["emotion_scores"]
    if scores:
        emo_df = pd.DataFrame(
            sorted(scores.items(), key=lambda x: -x[1]),
            columns=["Emotion", "Confidence"],
        ).set_index("Emotion")
        st.bar_chart(emo_df)
        st.metric("Dominant", res["dominant_emotion"].capitalize())
    else:
        st.info("Emotion model not available.")

# Sidebar

st.sidebar.title("Face Attendance System")

with st.sidebar.expander("Model Settings", expanded=False):
    _gpu_available = torch.cuda.is_available()
    device = st.selectbox(
        "Device",
        ["cuda", "cpu"] if _gpu_available else ["cpu"],
        index=0,
        help="GPU detected" if _gpu_available else "No GPU detected",
    )
    model_type = st.selectbox("Embedding model", MODEL_TYPES)
    metric = st.selectbox("Distance metric", ["cosine", "euclidean"])
    verify_threshold = st.slider("Verification threshold", 0.0, 1.0, 0.50, 0.01)
    top_k = st.slider("Top-K matches", 1, 10, 5)

mode = st.sidebar.radio(
    "Mode",
    ["Attendance (Verify)", "Live Recognition", "Register New Face", "View Database"],
)

if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = []

liveness = None
emotion_rec = None
database = load_database()

# Mode: Attendance

if mode == "Attendance (Verify)":
    st.title("Attendance — Face Verification")

    verifier = load_verifier(model_type, device)

    tab_cam, tab_upload, tab_log = st.tabs(["Camera", "Upload Image", "Attendance Log"])
    img_bgr = None

    with tab_cam:
        captured = st.camera_input("Capture your face")
        if captured is not None:
            img_bgr = uploaded_to_bgr(captured)

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a face image", type=["jpg", "jpeg", "png"], key="attend_upload"
        )
        if uploaded is not None:
            img_bgr = uploaded_to_bgr(uploaded)

    with tab_log:
        if st.session_state.attendance_log:
            st.dataframe(
                pd.DataFrame(st.session_state.attendance_log),
                use_container_width=True,
            )
            if st.button("Clear log"):
                st.session_state.attendance_log = []
                st.rerun()
        else:
            st.info("No attendance records yet.")

    if img_bgr is not None:
        res = run_full_pipeline(
            img_bgr, verifier, model_type, liveness, emotion_rec,
            database, metric, verify_threshold, top_k,
        )
        if res is None:
            st.error("No face detected. Please try again.")
        else:
            st.divider()
            col_img, col_info = st.columns([1, 1])

            with col_img:
                st.image(
                    to_rgb(res["annotated_bgr"]),
                    caption="Detected Face",
                    use_container_width=True,
                )

            with col_info:
                best_name = "Unknown"
                best_score = 0.0
                if (
                    res["top_matches"]
                    and res["top_matches"][0]["score"] >= verify_threshold
                ):
                    best_name = res["top_matches"][0]["name"]
                    best_score = res["top_matches"][0]["score"]

                if best_name != "Unknown":
                    st.success(f"**Identified: {best_name}**")
                else:
                    st.warning("**No match found in database**")

                m1, m2 = st.columns(2)
                m1.metric("Similarity", f"{best_score:.4f}")
                m2.metric("Metric", metric.capitalize())

                st.divider()

                st.caption("TOP MATCHES")
                matches = res["top_matches"]
                if matches:
                    df = pd.DataFrame(matches)[["name", "score"]].copy()
                    df.columns = ["Name", "Score"]
                    df["Score"] = df["Score"].round(4)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No faces registered in database.")

            st.session_state.attendance_log.insert(0, {
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Identity": best_name,
                "Score": f"{best_score:.4f}",
                "Model": model_type,
                "Metric": metric,
            })

# Mode: Register

elif mode == "Register New Face":
    st.title("Register New Face")
    st.info(
        "Registration extracts embeddings from **all 3 models** "
        "(classifier, arcface, triplet) so you can switch models freely."
    )

    all_verifiers = load_all_verifiers(device)
    available_count = sum(1 for v in all_verifiers.values() if v is not None)
    st.caption(f"{available_count}/{len(MODEL_TYPES)} models loaded")

    name = st.text_input("Full name", placeholder="e.g. Nguyen Van A")

    tab_cam, tab_upload = st.tabs(["Camera", "Upload Image"])
    img_bgr = None

    with tab_cam:
        captured = st.camera_input("Capture face for registration")
        if captured is not None:
            img_bgr = uploaded_to_bgr(captured)

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a face image", type=["jpg", "jpeg", "png"], key="reg_upload"
        )
        if uploaded is not None:
            img_bgr = uploaded_to_bgr(uploaded)
            st.image(Image.open(uploaded), caption="Uploaded image", use_container_width=True)

    if img_bgr is not None and name.strip():
        detector = load_detector(device)
        faces = detector.detect_and_crop(img_bgr)

        if not faces:
            st.error("No face detected. Try a different image.")
        elif available_count == 0:
            st.error("No model weights found. Place weights in final/weights/.")
        else:
            face_rgb, info = faces[0]
            annotated = draw_face_box(img_bgr, info["bbox"], name)
            st.image(to_rgb(annotated), caption="Detected face", use_container_width=True)

            if st.button("Confirm Registration", type="primary"):
                with st.spinner("Extracting embeddings from all models..."):
                    embeddings = extract_all_embeddings(face_rgb, all_verifiers)
                face_id = database.register(name.strip(), embeddings, face_rgb)
                st.success(
                    f"Registered '{name}' (ID: {face_id[:8]}...) "
                    f"with {len(embeddings)} model(s): {', '.join(embeddings.keys())}"
                )
    elif img_bgr is not None and not name.strip():
        st.warning("Please enter a name before registering.")

# Mode: Live Recognition

elif mode == "Live Recognition":
    st.title("Live Recognition")

    verifier = load_verifier(model_type, device)
    detector = load_detector(device)

    class FaceRecognitionProcessor(VideoProcessorBase):
        def __init__(self):
            self._prev_time = time.time()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")

            now = time.time()
            fps = 1.0 / max(now - self._prev_time, 1e-6)
            self._prev_time = now

            best_name = "Unknown"
            best_score = 0.0
            annotated = img_bgr.copy()

            if verifier is not None:
                faces = detector.detect_and_crop(img_bgr)
                if faces:
                    face_rgb, info = faces[0]
                    bbox = info["bbox"]
                    emb = verifier.extract_embedding(face_rgb)
                    matches = database.search(
                        emb, model_type=model_type, metric=metric, threshold=0.0
                    )[:top_k]

                    if matches and matches[0]["score"] >= verify_threshold:
                        best_name = matches[0]["name"]
                        best_score = matches[0]["score"]

                    color = (0, 200, 0) if best_name != "Unknown" else (0, 0, 255)
                    label = f"{best_name} ({best_score:.2f})"
                    annotated = draw_face_box(annotated, bbox, label, color)

            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="live-recognition",
        video_processor_factory=FaceRecognitionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption(
        f"Model: **{model_type}** | Metric: **{metric}** | "
        f"Threshold: **{verify_threshold}**"
    )

# Mode: View Database
elif mode == "View Database":
    st.title("Face Database")

    entries = database.list_all()
    if not entries:
        st.info("No faces registered yet.")
    else:
        st.write(f"**{len(entries)}** registered identities")

        header = st.columns([2, 3, 3, 1])
        header[0].markdown("**Name**")
        header[1].markdown("**Face ID**")
        header[2].markdown("**Registered**")
        header[3].markdown("**Action**")

        for entry in entries:
            c1, c2, c3, c4 = st.columns([2, 3, 3, 1])
            c1.write(entry["name"])
            c2.write(str(entry["face_id"])[:16] + "...")
            c3.write(str(entry.get("registered_at", "N/A"))[:19])

            if c4.button("Del", key=f"del_{entry['face_id']}"):
                database.delete(entry["face_id"])
                st.success(f"Deleted {entry['name']}")
                st.rerun()
