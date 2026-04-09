"""Core business logic for the face recognition attendance system."""

from src.core.face_verifier import FaceVerifier
from src.core.face_database import FaceDatabase

# Optional modules may depend on external packages (e.g., cv2, deepface).
# Keep imports resilient so lightweight scripts (e.g., verification evaluation)
# can run without the full UI stack installed.
try:
    from src.core.face_detector import FaceDetector
except Exception:  # pragma: no cover
    FaceDetector = None  # type: ignore

try:
    from src.core.liveness_checker import LivenessChecker
except Exception:  # pragma: no cover
    LivenessChecker = None  # type: ignore

try:
    from src.core.emotion_recognizer import EmotionRecognizer
except Exception:  # pragma: no cover
    EmotionRecognizer = None  # type: ignore

__all__ = [
    "FaceVerifier",
    "FaceDatabase",
    "FaceDetector",
    "LivenessChecker",
    "EmotionRecognizer",
]
