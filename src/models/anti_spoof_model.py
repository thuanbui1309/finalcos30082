"""Anti-spoofing (liveness detection) model based on MobileNetV2."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AntiSpoofNet(nn.Module):
    """Binary classifier for face anti-spoofing (real vs. fake).

    Uses a MobileNetV2 backbone pre-trained on ImageNet with a replaced
    final classifier layer outputting 2 logits.
    """

    def __init__(self) -> None:
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # MobileNetV2 has `classifier = Sequential(Dropout, Linear(1280, 1000))`
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 2),
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 2)``."""
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Run inference on a single (or batched) input and return a prediction.

        Args:
            x: Input tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.

        Returns:
            Tuple of ``(label, confidence)`` where *label* is ``0`` (fake) or
            ``1`` (real) and *confidence* is the softmax probability of the
            predicted class.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidence, label = torch.max(probs, dim=1)
        return int(label.item()), float(confidence.item())
