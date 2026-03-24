"""Classification-based face embedding models with softmax and ArcFace heads."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _build_resnet_backbone(backbone: str = "resnet50") -> tuple:
    """Return a ResNet backbone (without its FC layer) and the feature dim.

    Supported values for *backbone*: ``'resnet50'``, ``'resnet101'``.
    """
    if backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    elif backbone == "resnet101":
        base = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Remove the original fully-connected layer.
    layers = list(base.children())[:-1]  # everything up to (and including) avgpool
    backbone_net = nn.Sequential(*layers)
    return backbone_net, feat_dim


# ---------------------------------------------------------------------------
# ArcFace head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace) classification head.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
    Face Recognition", CVPR 2019.

    Args:
        embedding_dim: Dimensionality of the input embeddings.
        num_classes: Number of identity classes.
        s: Feature scale (re-scaling factor).
        m: Angular margin in radians.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin constants.
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)   # threshold to fall back
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace logits.

        Args:
            embeddings: L2-normalised embeddings of shape ``(B, D)``.
            labels: Ground-truth class indices of shape ``(B,)``.

        Returns:
            Scaled logits of shape ``(B, num_classes)``.
        """
        # Normalise weight vectors.
        normed_weight = F.normalize(self.weight, dim=1)
        # Cosine similarity: (B, num_classes)
        cosine = F.linear(F.normalize(embeddings, dim=1), normed_weight)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety: when cos(theta) <= cos(pi - m), use fallback.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for the target class.
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits


# ---------------------------------------------------------------------------
# Softmax classifier
# ---------------------------------------------------------------------------

class FaceClassifier(nn.Module):
    """Face embedding + softmax classification model.

    The network produces a 512-d embedding that is L2-normalised at
    inference time and can be used for face verification/identification.

    Args:
        num_classes: Number of identity classes (default 4000).
        embedding_dim: Dimensionality of the face embedding (default 512).
        backbone: Name of the ResNet backbone (default ``'resnet50'``).
    """

    def __init__(
        self,
        num_classes: int = 4000,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_resnet_backbone(backbone)
        self.embed_fc = nn.Linear(feat_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits of shape ``(B, num_classes)``."""
        feat = self.backbone(x).flatten(1)
        emb = self.bn(self.embed_fc(feat))
        logits = self.classifier(emb)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding of shape ``(B, embedding_dim)``."""
        feat = self.backbone(x).flatten(1)
        emb = self.bn(self.embed_fc(feat))
        emb = F.normalize(emb, p=2, dim=1)
        return emb


# ---------------------------------------------------------------------------
# ArcFace classifier
# ---------------------------------------------------------------------------

class ArcFaceClassifier(nn.Module):
    """Face embedding + ArcFace classification model.

    Uses the same backbone and embedding layer as :class:`FaceClassifier`
    but replaces the linear classifier with an :class:`ArcFaceHead`.

    Args:
        num_classes: Number of identity classes (default 4000).
        embedding_dim: Dimensionality of the face embedding (default 512).
        backbone: Name of the ResNet backbone (default ``'resnet50'``).
        s: ArcFace scale factor (default 30.0).
        m: ArcFace angular margin in radians (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 4000,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        s: float = 30.0,
        m: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_resnet_backbone(backbone)
        self.embed_fc = nn.Linear(feat_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, s=s, m=m)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Return ArcFace logits of shape ``(B, num_classes)``.

        Note:
            Labels are required during the forward pass because the ArcFace
            margin is only applied to the target class.
        """
        feat = self.backbone(x).flatten(1)
        emb = self.bn(self.embed_fc(feat))
        logits = self.arcface_head(emb, labels)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding of shape ``(B, embedding_dim)``."""
        feat = self.backbone(x).flatten(1)
        emb = self.bn(self.embed_fc(feat))
        emb = F.normalize(emb, p=2, dim=1)
        return emb
