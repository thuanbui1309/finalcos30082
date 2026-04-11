"""Model definitions for the face recognition attendance system."""

from .face_embed_classifier import ArcFaceClassifier, ArcFaceHead, FaceClassifier
from .face_embed_triplet import FaceEmbedNet, TripletDataset

# EdgeFace backbone is optional in lightweight setups.
try:
    from .edgeface_backbone import (
        EDGEFACE_VARIANTS,
        TimmFRWrapperV2,
        get_edgeface_model,
        load_edgeface,
    )
except ModuleNotFoundError:  # pragma: no cover
    EDGEFACE_VARIANTS = []
    TimmFRWrapperV2 = None  # type: ignore
    get_edgeface_model = None  # type: ignore
    load_edgeface = None  # type: ignore

__all__ = [
    "ArcFaceClassifier",
    "ArcFaceHead",
    "FaceClassifier",
    "FaceEmbedNet",
    "TripletDataset",
    "EDGEFACE_VARIANTS",
    "TimmFRWrapperV2",
    "get_edgeface_model",
    "load_edgeface",
]
