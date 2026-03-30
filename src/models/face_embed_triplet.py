"""Triplet-loss based face embedding model and triplet dataset."""

import os
import random
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms


def _build_resnet_backbone(backbone: str = "resnet50") -> Tuple[nn.Module, int]:
    """Return a ResNet backbone (without FC) and its feature dimension."""
    if backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    elif backbone == "resnet101":
        base = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    layers = list(base.children())[:-1]
    return nn.Sequential(*layers), feat_dim


# ---------------------------------------------------------------------------
# Embedding network
# ---------------------------------------------------------------------------

class FaceEmbedNet(nn.Module):
    """Face embedding network trained with triplet loss.

    Produces an L2-normalised embedding vector for each input face image.

    Args:
        embedding_dim: Output embedding dimensionality (default 512).
        backbone: ResNet variant to use (default ``'resnet50'``).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_resnet_backbone(backbone)
        self.embed = nn.Linear(feat_dim, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding of shape ``(B, embedding_dim)``."""
        feat = self.backbone(x).flatten(1)
        emb = self.bn(self.embed(feat))
        emb = F.normalize(emb, p=2, dim=1)
        return emb


# ---------------------------------------------------------------------------
# Triplet dataset
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):
    """Dataset that yields (anchor, positive, negative) image triplets.

    Expects an ImageFolder-style directory layout::

        root_dir/
            class_0/
                img_001.jpg
                img_002.jpg
            class_1/
                img_003.jpg
                ...

    For each sample the dataset randomly selects:
    - *anchor* and *positive* from the same class (two different images),
    - *negative* from a randomly chosen different class.

    Args:
        root_dir: Path to the ImageFolder root.
        transform: Optional torchvision transform applied to every image.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Build class -> list of image paths mapping.
        self.class_to_images: dict[str, list[str]] = defaultdict(list)
        self.classes: list[str] = sorted(
            entry.name
            for entry in os.scandir(root_dir)
            if entry.is_dir()
        )

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath):
                    self.class_to_images[cls_name].append(fpath)

        # Only keep classes that have at least 2 images (needed for anchor+pos).
        self.valid_classes = [
            c for c in self.classes if len(self.class_to_images[c]) >= 2
        ]

        # Flat list of (image_path, class_name) for indexing.
        self.samples = []
        for cls_name in self.valid_classes:
            for img_path in self.class_to_images[cls_name]:
                self.samples.append((img_path, cls_name))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, anchor_class = self.samples[index]

        # Positive: different image from the same class.
        pos_candidates = [
            p for p in self.class_to_images[anchor_class] if p != anchor_path
        ]
        positive_path = random.choice(pos_candidates)

        # Negative: random image from a different class.
        neg_class = random.choice(
            [c for c in self.valid_classes if c != anchor_class]
        )
        negative_path = random.choice(self.class_to_images[neg_class])

        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        return anchor, positive, negative
