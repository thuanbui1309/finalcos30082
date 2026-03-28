#!/usr/bin/env python3
"""Bulk-import face embeddings from the dataset into the face database.

For each identity folder in the dataset, picks ONE image, detects the face,
extracts the embedding, and registers it in the face database.

Usage:
    cd final/
    python scripts/bulk_import.py [--split train_data] [--model classifier] [--max N]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from tqdm import tqdm

from src.core.face_database import FaceDatabase
from src.core.face_detector import FaceDetector
from src.core.face_verifier import FaceVerifier


WEIGHTS = ROOT / "weights"
DB_DIR  = ROOT / "face_db"
DATA_DIR = ROOT / "data" / "classification_data"


def main():
    parser = argparse.ArgumentParser(description="Bulk import faces into the database")
    parser.add_argument("--split", default="train_data", help="Dataset split to use (default: train_data)")
    parser.add_argument("--model", default="classifier", choices=["classifier", "arcface", "triplet"],
                        help="Embedding model to use (default: classifier)")
    parser.add_argument("--max", type=int, default=0, help="Max identities to import (0 = all)")
    parser.add_argument("--images-per-id", type=int, default=1, help="Number of images per identity (default: 1)")
    parser.add_argument("--clear", action="store_true", help="Clear database before importing")
    args = parser.parse_args()

    split_dir = DATA_DIR / args.split
    if not split_dir.exists():
        print(f"ERROR: Split directory not found: {split_dir}")
        sys.exit(1)

    # Load models
    print("Loading models...")
    detector = FaceDetector(device="cpu")

    weight_map = {
        "classifier": WEIGHTS / "face_classification.pth",
        "arcface":    WEIGHTS / "face_classification_arc.pth",
        "triplet":    WEIGHTS / "face_metric_learning.pth",
    }
    wp = weight_map[args.model]
    if not wp.exists():
        print(f"ERROR: Weights not found: {wp}")
        sys.exit(1)
    verifier = FaceVerifier(model_type=args.model, weights_path=str(wp), device="cpu")

    database = FaceDatabase(db_dir=str(DB_DIR))

    # Optionally clear existing database
    if args.clear:
        existing = database.list_all()
        if existing:
            print(f"Clearing {len(existing)} existing entries...")
            for entry in existing:
                database.delete(entry["face_id"])

    # List identity folders
    id_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if args.max > 0:
        id_folders = id_folders[:args.max]

    print(f"Importing from: {split_dir}")
    print(f"Model: {args.model} | Images per ID: {args.images_per_id}")
    print(f"Identities to process: {len(id_folders)}")
    print()

    success = 0
    skipped = 0
    failed  = 0

    for id_dir in tqdm(id_folders, desc="Importing"):
        identity_name = id_dir.name
        images = sorted(id_dir.glob("*.jpg"))
        if not images:
            skipped += 1
            continue

        # Take first N images
        selected = images[:args.images_per_id]

        for img_path in selected:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                failed += 1
                continue

            faces = detector.detect_and_crop(img_bgr)
            if not faces:
                failed += 1
                continue

            face_rgb, _ = faces[0]
            emb = verifier.extract_embedding(face_rgb)
            database.register(identity_name, emb, face_rgb)
            success += 1

    print()
    print(f"Done! Imported: {success} | Skipped: {skipped} | Failed: {failed}")
    print(f"Total entries in database: {len(database.list_all())}")


if __name__ == "__main__":
    main()
