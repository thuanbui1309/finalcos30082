"""Face verification via embedding comparison."""

import numpy as np
import torch
from PIL import Image

from src.models import ArcFaceClassifier, FaceClassifier, FaceEmbedNet

try:
    from src.models.edgeface_backbone import EDGEFACE_VARIANTS, load_edgeface
except ModuleNotFoundError:  # pragma: no cover
    EDGEFACE_VARIANTS = []
    load_edgeface = None  # type: ignore
from src.utils.metrics import cosine_similarity, euclidean_distance
from src.utils.transforms import get_edgeface_transform, get_inference_transform


class FaceVerifier:
    """Extract face embeddings and compare them for verification.

    Supports the following model types:
        - 'classifier'              -> FaceClassifier (ResNet50 + softmax head)
        - 'arcface'                 -> ArcFaceClassifier (ResNet50 + ArcFace head)
        - 'triplet'                 -> FaceEmbedNet (ResNet50 + triplet loss)
        - 'edgeface_xs_gamma_06'    -> EdgeFace XS (1.77 M params, recommended)
        - 'edgeface_s_gamma_05'     -> EdgeFace S  (3.65 M params)
        - 'edgeface_xxs'            -> EdgeFace XXS (1.24 M params)
        - 'edgeface_base'           -> EdgeFace Base (18.23 M params)
    """

    def __init__(
        self,
        model_type: str = "classifier",
        weights_path: str = None,
        device: str = "cpu",
        num_classes: int = 4000,
    ):
        self.device = torch.device(device)
        self.model_type = model_type

        if model_type in EDGEFACE_VARIANTS:
            if load_edgeface is None:
                raise ImportError(
                    "EdgeFace backbone is not available in this installation. "
                    "Install/restore src.models.edgeface_backbone to use EdgeFace variants."
                )
            self.transform = get_edgeface_transform(img_size=112)
            self.model = load_edgeface(
                name=model_type,
                weights_path=weights_path,
                device=str(self.device),
            )
            return

        self.transform = get_inference_transform(img_size=112)

        if model_type == "classifier":
            self.model = FaceClassifier(
                num_classes=num_classes, embedding_dim=512, backbone="resnet50"
            )
        elif model_type == "arcface":
            self.model = ArcFaceClassifier(
                num_classes=num_classes, embedding_dim=512, backbone="resnet50"
            )
        elif model_type == "triplet":
            self.model = FaceEmbedNet(embedding_dim=512, backbone="resnet50")
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from 'classifier', 'arcface', 'triplet', or one of {EDGEFACE_VARIANTS}."
            )

        if weights_path is not None:
            try:
                state = torch.load(weights_path, map_location=self.device)
                if model_type == "arcface":
                    remapped = {}
                    for k, v in state.items():
                        if k.startswith("head."):
                            remapped[k.replace("head.", "arcface_head.", 1)] = v
                        else:
                            remapped[k] = v
                    state = remapped
                self.model.load_state_dict(state, strict=False)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Weights file not found: {weights_path}"
                )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract a 512-d embedding from a cropped face image.

        Args:
            face_image: HxWx3 numpy array in RGB, any size.

        Returns:
            1-D numpy array of length 512.
        """
        pil_img = Image.fromarray(face_image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        if self.model_type == "triplet" or self.model_type in EDGEFACE_VARIANTS:
            embedding = self.model(tensor)
        else:
            embedding = self.model.extract_embedding(tensor)

        return embedding.squeeze(0).cpu().numpy()

    def compare(
        self, emb1: np.ndarray, emb2: np.ndarray, metric: str = "cosine"
    ) -> float:
        """Compare two embeddings and return a similarity score.

        Higher values always indicate greater similarity. For euclidean
        distance the negative distance is returned.

        Args:
            emb1: 1-D numpy embedding.
            emb2: 1-D numpy embedding.
            metric: 'cosine' or 'euclidean'.

        Returns:
            Similarity score (float).
        """
        if metric == "cosine":
            return cosine_similarity(emb1, emb2)
        elif metric == "euclidean":
            return -euclidean_distance(emb1, emb2)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'cosine' or 'euclidean'.")

    def verify(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = 0.5,
        metric: str = "cosine",
    ) -> tuple:
        """Verify whether two embeddings belong to the same person.

        Args:
            emb1: 1-D numpy embedding.
            emb2: 1-D numpy embedding.
            threshold: Minimum similarity score to consider a match.
            metric: 'cosine' or 'euclidean'.

        Returns:
            (is_same_person, score) tuple.
        """
        score = self.compare(emb1, emb2, metric=metric)
        return (score >= threshold, score)
