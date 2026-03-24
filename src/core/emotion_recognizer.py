"""Facial emotion recognition."""

import numpy as np
import torch
from PIL import Image

from src.models import EmotionNet
from src.utils.transforms import get_inference_transform


class EmotionRecognizer:
    """Recognize facial emotions from cropped face images."""

    EMOTION_ICONS = {
        "angry": "\U0001f620",     # angry face
        "disgust": "\U0001f922",   # nauseated face
        "fear": "\U0001f628",      # fearful face
        "happy": "\U0001f60a",     # smiling face with smiling eyes
        "sad": "\U0001f622",       # crying face
        "surprise": "\U0001f632",  # astonished face
        "neutral": "\U0001f610",   # neutral face
    }

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.transform = get_inference_transform(img_size=112)

        self.model = EmotionNet(num_classes=7)

        if weights_path is not None:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Weights file not found: {weights_path}"
                )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def recognize(self, face_image: np.ndarray) -> tuple:
        """Recognize the emotion in a cropped face image.

        Args:
            face_image: HxWx3 numpy array in RGB.

        Returns:
            (emotion_label, confidence) tuple, e.g. ('happy', 0.93).
        """
        pil_img = Image.fromarray(face_image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        emotion_label, confidence = self.model.predict(tensor)

        return (str(emotion_label), float(confidence))
