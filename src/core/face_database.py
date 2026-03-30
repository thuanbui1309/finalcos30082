"""Persistent face embedding database backed by JSON + .npy files."""

import glob
import json
import os
import uuid
from datetime import datetime

import cv2
import numpy as np

from src.utils.metrics import cosine_similarity, euclidean_distance

ALL_MODEL_TYPES = ("classifier", "arcface", "triplet")


class FaceDatabase:
    """Store and query face embeddings on disk.

    Layout inside *db_dir*::

        db_dir/
            metadata.json
            embeddings/
                <face_id>_classifier.npy
                <face_id>_arcface.npy
                <face_id>_triplet.npy
            thumbnails/
                <face_id>.png
    """

    def __init__(self, db_dir: str = "face_db"):
        self.db_dir = db_dir
        self.emb_dir = os.path.join(db_dir, "embeddings")
        self.thumb_dir = os.path.join(db_dir, "thumbnails")
        self.meta_path = os.path.join(db_dir, "metadata.json")

        os.makedirs(self.emb_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

        if not os.path.isfile(self.meta_path):
            self._save_db([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        embeddings: dict[str, np.ndarray],
        image: np.ndarray = None,
    ) -> str:
        """Register a new face with embeddings from multiple models.

        Args:
            name: Person's name.
            embeddings: Mapping of model_type -> 1-D embedding array,
                e.g. {"classifier": arr, "arcface": arr, "triplet": arr}.
            image: Optional face thumbnail (RGB numpy array).

        Returns:
            Unique face_id string.
        """
        face_id = str(uuid.uuid4())

        for model_type, emb in embeddings.items():
            path = os.path.join(self.emb_dir, f"{face_id}_{model_type}.npy")
            np.save(path, emb)

        if image is not None:
            thumb_path = os.path.join(self.thumb_dir, f"{face_id}.png")
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(thumb_path, bgr)

        db = self._load_db()
        db.append(
            {
                "face_id": face_id,
                "name": name,
                "models": list(embeddings.keys()),
                "registered_at": datetime.now().isoformat(),
            }
        )
        self._save_db(db)
        return face_id

    def search(
        self,
        embedding: np.ndarray,
        model_type: str = "classifier",
        metric: str = "cosine",
        threshold: float = 0.5,
    ) -> list:
        """Search for matching faces using embeddings from a specific model.

        Args:
            embedding: Query embedding (1-D numpy array).
            model_type: Which model's stored embedding to compare against.
            metric: 'cosine' or 'euclidean'.
            threshold: Minimum similarity score to include.

        Returns:
            List of dicts sorted by score descending:
            [{"face_id", "name", "score"}, ...]
        """
        db = self._load_db()
        matches = []

        for record in db:
            fid = record["face_id"]
            emb_path = os.path.join(self.emb_dir, f"{fid}_{model_type}.npy")
            if not os.path.isfile(emb_path):
                continue

            stored_emb = np.load(emb_path)
            score = self._compute_score(embedding, stored_emb, metric)

            if score >= threshold:
                matches.append(
                    {"face_id": fid, "name": record["name"], "score": score}
                )

        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    def identify(
        self,
        embedding: np.ndarray,
        model_type: str = "classifier",
        metric: str = "cosine",
        threshold: float = 0.5,
    ) -> tuple:
        """Identify the best-matching person for the given embedding.

        Returns:
            (name, score) of the best match, or (None, 0.0) if none.
        """
        matches = self.search(
            embedding, model_type=model_type, metric=metric, threshold=threshold
        )
        if matches:
            best = matches[0]
            return (best["name"], best["score"])
        return (None, 0.0)

    def list_all(self) -> list:
        """Return all registered face records with available model info.

        Returns:
            List of dicts with keys: face_id, name, registered_at, models.
        """
        db = self._load_db()
        for record in db:
            if "models" not in record:
                available = []
                for mt in ALL_MODEL_TYPES:
                    p = os.path.join(self.emb_dir, f"{record['face_id']}_{mt}.npy")
                    if os.path.isfile(p):
                        available.append(mt)
                record["models"] = available
        return db

    def delete(self, face_id: str) -> bool:
        """Delete a registered face and all its embedding files."""
        db = self._load_db()
        new_db = [r for r in db if r["face_id"] != face_id]

        if len(new_db) == len(db):
            return False

        self._save_db(new_db)

        for path in glob.glob(os.path.join(self.emb_dir, f"{face_id}_*.npy")):
            os.remove(path)

        # Legacy single-file cleanup
        legacy_path = os.path.join(self.emb_dir, f"{face_id}.npy")
        if os.path.isfile(legacy_path):
            os.remove(legacy_path)

        thumb_path = os.path.join(self.thumb_dir, f"{face_id}.png")
        if os.path.isfile(thumb_path):
            os.remove(thumb_path)

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_db(self) -> list:
        try:
            with open(self.meta_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_db(self, db: list) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(db, f, indent=2)

    @staticmethod
    def _compute_score(
        emb1: np.ndarray, emb2: np.ndarray, metric: str
    ) -> float:
        if metric == "cosine":
            return cosine_similarity(emb1, emb2)
        elif metric == "euclidean":
            return -euclidean_distance(emb1, emb2)
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Use 'cosine' or 'euclidean'."
            )
