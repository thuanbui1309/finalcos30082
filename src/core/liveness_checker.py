"""Anti-spoofing liveness detection."""

import numpy as np
import torch
from PIL import Image

from src.models import AntiSpoofNet
from src.utils.transforms import get_inference_transform


class LivenessChecker:
    """Determine whether a face image is from a real person or a spoof."""

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.transform = get_inference_transform(img_size=112)

        self.model = AntiSpoofNet()

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
    def check(self, face_image: np.ndarray) -> tuple:
        """Check whether a cropped face is real or spoofed.

        Args:
            face_image: HxWx3 numpy array in RGB.

        Returns:
            (is_real, confidence) where is_real is True for live faces.
        """
        pil_img = Image.fromarray(face_image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        label, confidence = self.model.predict(tensor)

        is_real = label == 0  # 0 = real, 1 = spoof
        return (bool(is_real), float(confidence))
