"""Model definitions for the face recognition attendance system."""

from .face_embed_classifier import ArcFaceClassifier, ArcFaceHead, FaceClassifier
from .face_embed_triplet import FaceEmbedNet, TripletDataset
from .edgeface_backbone import EDGEFACE_VARIANTS, TimmFRWrapperV2, get_edgeface_model, load_edgeface

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
