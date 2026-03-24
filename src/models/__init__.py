"""Model definitions for the face recognition attendance system."""

from .anti_spoof_model import AntiSpoofNet
from .emotion_model import EmotionNet
from .face_embed_classifier import ArcFaceClassifier, ArcFaceHead, FaceClassifier
from .face_embed_triplet import FaceEmbedNet, TripletDataset

__all__ = [
    "AntiSpoofNet",
    "ArcFaceClassifier",
    "ArcFaceHead",
    "EmotionNet",
    "FaceClassifier",
    "FaceEmbedNet",
    "TripletDataset",
]
