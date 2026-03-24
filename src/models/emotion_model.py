"""Facial emotion recognition model based on ResNet-18."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmotionNet(nn.Module):
    """Seven-class facial emotion classifier.

    Uses a ResNet-18 backbone pre-trained on ImageNet with a replaced
    final fully-connected layer mapping to 7 emotion categories.

    Attributes:
        EMOTIONS: Ordered list of emotion labels aligned with class indices.
    """

    EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features  # 512
        base.fc = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)``."""
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[str, float]:
        """Run inference and return the predicted emotion.

        Args:
            x: Input tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``.

        Returns:
            Tuple of ``(emotion_label, confidence)`` where *emotion_label* is
            a string from :attr:`EMOTIONS` and *confidence* is the softmax
            probability of the predicted class.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)
        emotion_label = self.EMOTIONS[idx.item()]
        return emotion_label, float(confidence.item())
