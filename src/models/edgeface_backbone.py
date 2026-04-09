"""EdgeFace backbone for efficient face recognition on edge devices.

Reference: "EdgeFace: Efficient Face Recognition Model for Edge Devices"
           IEEE T-BIOM 2024 — Anjith George, Idiap Research Institute.
           https://arxiv.org/abs/2307.01838
"""

import timm
import torch
import torch.nn as nn


class LoRaLin(nn.Module):
    """Low-Rank Linear layer for parameter compression."""

    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def _replace_linear_lowrank(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and "head" not in name:
            rank = max(2, int(min(module.in_features, module.out_features) * rank_ratio))
            setattr(
                model,
                name,
                LoRaLin(module.in_features, module.out_features, rank, module.bias is not None),
            )
        else:
            _replace_linear_lowrank(module, rank_ratio)


class TimmFRWrapperV2(nn.Module):
    """EdgeNext backbone wrapped for face recognition — outputs 512-d embeddings."""

    def __init__(self, model_name: str = "edgenext_x_small", featdim: int = 512):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        self.model = timm.create_model(model_name)
        self.model.reset_classifier(featdim)

    def forward(self, x):
        return self.model(x)


_BACKBONE_MAP = {
    "edgeface_base": ("edgenext_base", None),
    "edgeface_s_gamma_05": ("edgenext_small", 0.5),
    "edgeface_xs_gamma_06": ("edgenext_x_small", 0.6),
    "edgeface_xxs": ("edgenext_xx_small", None),
}

_CHECKPOINT_URLS = {
    "edgeface_base": "https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_base.pt",
    "edgeface_s_gamma_05": "https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_s_gamma_05.pt",
    "edgeface_xs_gamma_06": "https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_xs_gamma_06.pt",
    "edgeface_xxs": "https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface/-/raw/master/checkpoints/edgeface_xxs.pt",
}

EDGEFACE_VARIANTS = list(_BACKBONE_MAP.keys())


def get_edgeface_model(name: str) -> TimmFRWrapperV2:
    """Instantiate an EdgeFace model architecture (no weights loaded).

    Args:
        name: Variant name, e.g. 'edgeface_xs_gamma_06'.

    Returns:
        TimmFRWrapperV2 instance.
    """
    if name not in _BACKBONE_MAP:
        raise ValueError(
            f"Unknown EdgeFace variant '{name}'. Choose from: {EDGEFACE_VARIANTS}"
        )
    backbone_name, rank_ratio = _BACKBONE_MAP[name]
    model = TimmFRWrapperV2(model_name=backbone_name)
    if rank_ratio is not None:
        _replace_linear_lowrank(model, rank_ratio)
    return model


def load_edgeface(
    name: str,
    weights_path: str = None,
    device: str = "cpu",
) -> TimmFRWrapperV2:
    """Create and load an EdgeFace model with pretrained weights.

    Args:
        name: Variant name, e.g. 'edgeface_xs_gamma_06'.
        weights_path: Path to a local .pt checkpoint. If None, downloads
            the official pretrained weights automatically.
        device: Torch device string ('cpu' or 'cuda').

    Returns:
        TimmFRWrapperV2 in eval mode on the requested device.
    """
    model = get_edgeface_model(name)
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device)
    else:
        url = _CHECKPOINT_URLS[name]
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
