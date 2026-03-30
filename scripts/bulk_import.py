#!/usr/bin/env python3
"""Bulk-import face embeddings from the dataset into the face database.

For each identity folder, picks ONE image, detects the face, extracts
embeddings from ALL 3 models, and registers them in the database.

Usage:
    cd final/
    python scripts/bulk_import.py [--split train_data] [--max N] [--clear]
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
DB_DIR = ROOT / "face_db"
DATA_DIR = ROOT / "data" / "classification_data"

MODEL_TYPES = ["classifier", "arcface", "triplet"]
WEIGHT_MAP = {
    "classifier": WEIGHTS / "face_classification.pth",
    "arcface": WEIGHTS / "face_classification_arc.pth",
    "triplet": WEIGHTS / "face_metric_learning.pth",
}


def main():
    parser = argparse.ArgumentParser(description="Bulk import faces into the database")
    parser.add_argument("--split", default="train_data",
                        help="Dataset split to use (default: train_data)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max identities to import (0 = all)")
    parser.add_argument("--images-per-id", type=int, default=1,
                        help="Number of images per identity (default: 1)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear database before importing")
    args = parser.parse_args()

    split_dir = DATA_DIR / args.split
    if not split_dir.exists():
        print(f"ERROR: Split directory not found: {split_dir}")
        sys.exit(1)

    print("Loading models...")
    detector = FaceDetector(device="cpu")

    verifiers = {}
    for mt in MODEL_TYPES:
        wp = WEIGHT_MAP[mt]
        if wp.exists():
            verifiers[mt] = FaceVerifier(model_type=mt, weights_path=str(wp), device="cpu")
            print(f"  ✅ {mt}: loaded")
        else:
            print(f"  ❌ {mt}: weights not found ({wp.name})")

    if not verifiers:
        print("ERROR: No model weights found.")
        sys.exit(1)

    database = FaceDatabase(db_dir=str(DB_DIR))

    if args.clear:
        existing = database.list_all()
        if existing:
            print(f"Clearing {len(existing)} existing entries...")
            for entry in existing:
                database.delete(entry["face_id"])

    id_folders = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if args.max > 0:
        id_folders = id_folders[:args.max]

    print(f"\nImporting from: {split_dir}")
    print(f"Models: {', '.join(verifiers.keys())} | Images per ID: {args.images_per_id}")
    print(f"Identities to process: {len(id_folders)}\n")

    success = 0
    skipped = 0
    failed = 0

    for id_dir in tqdm(id_folders, desc="Importing"):
        identity_name = id_dir.name
        images = sorted(id_dir.glob("*.jpg"))
        if not images:
            skipped += 1
            continue

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

            embeddings = {}
            for mt, verifier in verifiers.items():
                embeddings[mt] = verifier.extract_embedding(face_rgb)

            database.register(identity_name, embeddings, face_rgb)
            success += 1

    print(f"\nDone! Imported: {success} | Skipped: {skipped} | Failed: {failed}")
    print(f"Models per entry: {', '.join(verifiers.keys())}")
    print(f"Total entries in database: {len(database.list_all())}")


if __name__ == "__main__":
    main()
