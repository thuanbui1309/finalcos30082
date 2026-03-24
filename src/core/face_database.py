"""Persistent face embedding database backed by JSON + .npy files."""

import json
import os
import uuid
from datetime import datetime

import cv2
import numpy as np

from src.utils.metrics import cosine_similarity, euclidean_distance


class FaceDatabase:
    """Store and query face embeddings on disk.

    Layout inside *db_dir*::

        db_dir/
            metadata.json          # list of registered face records
            embeddings/
                <face_id>.npy      # one file per registered face
            thumbnails/
                <face_id>.png      # optional face thumbnail
    """

    def __init__(self, db_dir: str = "face_db"):
        self.db_dir = db_dir
        self.emb_dir = os.path.join(db_dir, "embeddings")
        self.thumb_dir = os.path.join(db_dir, "thumbnails")
        self.meta_path = os.path.join(db_dir, "metadata.json")

        os.makedirs(self.emb_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

        # Ensure metadata file exists
        if not os.path.isfile(self.meta_path):
            self._save_db([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self, name: str, embedding: np.ndarray, image: np.ndarray = None
    ) -> str:
        """Register a new face embedding.

        Args:
            name: Person's name.
            embedding: 1-D numpy embedding (e.g. 512-d).
            image: Optional face thumbnail (RGB numpy array) to store.

        Returns:
            Unique face_id string.
        """
        face_id = str(uuid.uuid4())

        # Save embedding
        np.save(os.path.join(self.emb_dir, f"{face_id}.npy"), embedding)

        # Save optional thumbnail
        if image is not None:
            thumb_path = os.path.join(self.thumb_dir, f"{face_id}.png")
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(thumb_path, bgr)

        # Update metadata
        db = self._load_db()
        db.append(
            {
                "face_id": face_id,
                "name": name,
                "registered_at": datetime.now().isoformat(),
            }
        )
        self._save_db(db)

        return face_id

    def search(
        self,
        embedding: np.ndarray,
        metric: str = "cosine",
        threshold: float = 0.5,
    ) -> list:
        """Search for matching faces above the similarity threshold.

        Args:
            embedding: Query embedding (1-D numpy array).
            metric: 'cosine' or 'euclidean'.
            threshold: Minimum similarity score to include.

        Returns:
            List of dicts sorted by score (descending):
            [{"face_id", "name", "score"}, ...]
        """
        db = self._load_db()
        matches = []

        for record in db:
            fid = record["face_id"]
            emb_path = os.path.join(self.emb_dir, f"{fid}.npy")
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
        metric: str = "cosine",
        threshold: float = 0.5,
    ) -> tuple:
        """Identify the best-matching person for the given embedding.

        Args:
            embedding: Query embedding.
            metric: 'cosine' or 'euclidean'.
            threshold: Minimum score for a valid match.

        Returns:
            (name, score) of the best match, or (None, 0.0) if none.
        """
        matches = self.search(embedding, metric=metric, threshold=threshold)
        if matches:
            best = matches[0]
            return (best["name"], best["score"])
        return (None, 0.0)

    def list_all(self) -> list:
        """Return all registered face records.

        Returns:
            List of dicts [{"face_id", "name", "registered_at"}, ...].
        """
        return self._load_db()

    def delete(self, face_id: str) -> bool:
        """Delete a registered face by its ID.

        Returns:
            True if the face was found and deleted, False otherwise.
        """
        db = self._load_db()
        new_db = [r for r in db if r["face_id"] != face_id]

        if len(new_db) == len(db):
            return False

        self._save_db(new_db)

        # Remove embedding file
        emb_path = os.path.join(self.emb_dir, f"{face_id}.npy")
        if os.path.isfile(emb_path):
            os.remove(emb_path)

        # Remove thumbnail if present
        thumb_path = os.path.join(self.thumb_dir, f"{face_id}.png")
        if os.path.isfile(thumb_path):
            os.remove(thumb_path)

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_db(self) -> list:
        """Load the metadata JSON file."""
        try:
            with open(self.meta_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_db(self, db: list) -> None:
        """Persist the metadata list to JSON."""
        with open(self.meta_path, "w") as f:
            json.dump(db, f, indent=2)

    @staticmethod
    def _compute_score(
        emb1: np.ndarray, emb2: np.ndarray, metric: str
    ) -> float:
        """Compute a similarity score (higher = more similar)."""
        if metric == "cosine":
            return cosine_similarity(emb1, emb2)
        elif metric == "euclidean":
            return -euclidean_distance(emb1, emb2)
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Use 'cosine' or 'euclidean'."
            )
