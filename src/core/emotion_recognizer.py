"""Facial emotion recognition using DeepFace (pre-trained model)."""

import numpy as np
import cv2
from deepface import DeepFace


class EmotionRecognizer:
    """Recognize facial emotions using DeepFace's pre-trained model.

    No custom training needed — DeepFace auto-downloads weights on first use.
    """

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Map DeepFace labels -> our canonical labels
    _LABEL_MAP = {e: e for e in EMOTIONS}

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        pass

    def recognize(self, face_image: np.ndarray) -> tuple[str, float]:
        """Recognize emotion from a cropped face image.

        Args:
            face_image: HxWx3 numpy array in RGB.

        Returns:
            (emotion_label, confidence) e.g. ('happy', 0.93)
        """
        bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        try:
            results = DeepFace.analyze(
                bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            result   = results[0] if isinstance(results, list) else results
            dominant = result["dominant_emotion"]
            label    = self._LABEL_MAP.get(dominant, "neutral")
            conf     = result["emotion"][dominant] / 100.0
            return (label, float(conf))
        except Exception:
            return ("neutral", 0.0)

    def recognize_all(self, face_image: np.ndarray) -> dict[str, float]:
        """Return confidence scores for all 7 emotions.

        Returns:
            dict mapping emotion label -> probability (0-1).
        """
        bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        try:
            results = DeepFace.analyze(
                bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            result = results[0] if isinstance(results, list) else results
            return {k: v / 100.0 for k, v in result["emotion"].items()
                    if k in self._LABEL_MAP}
        except Exception:
            return {e: 0.0 for e in self.EMOTIONS}
