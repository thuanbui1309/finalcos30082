"""Model definitions for the face recognition attendance system."""

from .face_embed_classifier import ArcFaceClassifier, ArcFaceHead, FaceClassifier
from .face_embed_triplet import FaceEmbedNet, TripletDataset

__all__ = [
    "ArcFaceClassifier",
    "ArcFaceHead",
    "FaceClassifier",
    "FaceEmbedNet",
    "TripletDataset",
]
