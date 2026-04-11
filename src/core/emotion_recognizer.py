"""Facial emotion recognition using HSEmotion-ONNX (EfficientNet-B0, AffectNet)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Canonical lowercase labels (8 classes from AffectNet)
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


class EmotionRecognizer:
    """Recognize facial emotions using HSEmotion-ONNX.

    Model: EfficientNet-B0 trained on AffectNet (~15 MB, downloaded on first use).
    Input: pre-cropped RGB face image (any size — resized internally to 224x224).
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self._model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")

    def _run(self, face_image: np.ndarray) -> dict[str, float]:
        """Run inference on a pre-cropped RGB face. Returns emotion -> probability."""
        try:
            label, scores = self._model.predict_emotions(face_image, logits=False)
            return {
                self._model.idx_to_class[i].lower(): float(scores[i])
                for i in range(len(scores))
            }
        except Exception as e:
            logger.error("Emotion recognition failed: %s", e, exc_info=True)
            return {}

    def recognize(self, face_image: np.ndarray) -> tuple[str, float]:
        """Return (dominant_emotion, confidence)."""
        scores = self._run(face_image)
        if not scores:
            return ("neutral", 0.0)
        dominant = max(scores, key=scores.get)
        return (dominant, float(scores[dominant]))

    def recognize_all(self, face_image: np.ndarray) -> dict[str, float]:
        """Return confidence scores for all emotions (0–1)."""
        scores = self._run(face_image)
        if not scores:
            return {e: 0.0 for e in EMOTIONS}
        return scores
