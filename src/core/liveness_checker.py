"""Anti-spoofing liveness detection using DeepFace."""

from __future__ import annotations

import logging
import cv2
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


class LivenessChecker:
    """Determine whether a face image is from a real person or a spoof.

    Uses DeepFace's built-in anti-spoofing via ``extract_faces(anti_spoofing=True)``.
    No custom weights needed — DeepFace handles model download automatically.
    """

    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        # weights_path kept for API compatibility but unused with DeepFace
        pass

    def check(self, face_image: np.ndarray) -> tuple[bool, float]:
        """Check whether a cropped face is real or spoofed.

        Args:
            face_image: HxWx3 numpy array in RGB.

        Returns:
            (is_real, confidence)
        """
        bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        try:
            results = DeepFace.extract_faces(
                bgr,
                anti_spoofing=True,
                enforce_detection=False,
            )
            logger.debug("DeepFace anti-spoof raw results: %s", results)
            if results:
                face = results[0]
                logger.debug("Face keys: %s", list(face.keys()))
                is_real = face.get("is_real", False)
                confidence = face.get("antispoof_score", 1.0 if is_real else 0.0)
                return (bool(is_real), float(confidence))
            logger.warning("DeepFace returned empty results for anti-spoofing")
            return (False, 0.0)
        except Exception as e:
            logger.error("Liveness check failed: %s", e, exc_info=True)
            return (True, 0.5)  # fail-open: unknown = pass through
