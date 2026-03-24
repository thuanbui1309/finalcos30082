"""Anti-spoofing liveness detection using MiniFASNet (pre-trained).

Weights are downloaded automatically from HuggingFace on first use.
Model: minivision-ai/Silent-Face-Anti-Spoofing (MiniFASNetV2)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MiniFASNetV2 — lightweight anti-spoof model (binary: real vs fake)
# ---------------------------------------------------------------------------

class _DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class MiniFASNet(nn.Module):
    """Compact face anti-spoofing network."""

    def __init__(self, num_classes=2, embedding_size=128):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(32)
        self.layers = nn.Sequential(
            _DepthwiseSeparable(32,  64),
            _DepthwiseSeparable(64, 128, stride=2),
            _DepthwiseSeparable(128, 128),
            _DepthwiseSeparable(128, 256, stride=2),
            _DepthwiseSeparable(256, 256),
            _DepthwiseSeparable(256, 512, stride=2),
            *[_DepthwiseSeparable(512, 512) for _ in range(5)],
            _DepthwiseSeparable(512, 1024, stride=2),
            _DepthwiseSeparable(1024, 1024),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Pre-trained weights URL (HuggingFace)
# ---------------------------------------------------------------------------

_WEIGHTS_URL = (
    "https://huggingface.co/minivision-ai/silent-face-anti-spoofing/"
    "resolve/main/MiniFASNetV2.pth"
)


class LivenessChecker:
    """Determine whether a face image is from a real person or a spoof.

    Uses MiniFASNetV2 pre-trained weights. Weights are downloaded automatically
    to `weights/anti_spoofing.pth` on first use.
    """

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)

        # Resolve weights path
        if weights_path is None:
            weights_path = str(
                Path(__file__).resolve().parents[2] / "weights" / "anti_spoofing.pth"
            )

        wp = Path(weights_path)
        if not wp.exists():
            self._download_weights(wp)

        self.model = MiniFASNet(num_classes=2)
        state = torch.load(str(wp), map_location=self.device)
        # Support both raw state_dict and checkpoint dicts
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    @staticmethod
    def _download_weights(dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading anti-spoofing weights to {dest} ...")
        urllib.request.urlretrieve(_WEIGHTS_URL, str(dest))
        print("Download complete.")

    def _preprocess(self, face_rgb: np.ndarray) -> torch.Tensor:
        img = cv2.resize(face_rgb, (80, 80))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    @torch.no_grad()
    def check(self, face_image: np.ndarray) -> tuple[bool, float]:
        """Check whether a cropped face is real or spoofed.

        Args:
            face_image: HxWx3 numpy array in RGB.

        Returns:
            (is_real, confidence)
        """
        tensor = self._preprocess(face_image)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        # class 1 = real, class 0 = spoof (MiniFASNet convention)
        real_prob = float(probs[1])
        is_real   = real_prob > 0.5
        return (is_real, real_prob)
