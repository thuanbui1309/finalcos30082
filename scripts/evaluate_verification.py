"""Evaluate face verification on verification pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.face_verifier import FaceVerifier
from src.utils.metrics import compute_auc, compute_roc, find_best_threshold


@dataclass(frozen=True)
class Pair:
    img1: str
    img2: str
    label: int


def _read_pairs(pairs_path: Path) -> List[Pair]:
    pairs: List[Pair] = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid line {line_no} in {pairs_path}: expected 3 columns, got {len(parts)}"
                )
            p1, p2, lab = parts
            pairs.append(Pair(img1=p1, img2=p2, label=int(lab)))
    return pairs


def _resolve_image_path(base_dir: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_face_rgb(
    img_path: Path,
    detector,
) -> np.ndarray:
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot read image: {img_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {img_path}") from e

    if detector is None:
        return np.array(pil_img)

    try:
        import cv2  # lazy import; only required when using detector
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cv2 is required for --use-detector. Install opencv-python or run without --use-detector."
        ) from e

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = detector.detect_and_crop(img_bgr)
    if not faces:
        raise RuntimeError(f"No face detected in: {img_path}")
    face_rgb, _info = faces[0]
    return face_rgb


def _iter_unique_images(pairs: Iterable[Pair]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in pairs:
        for x in (p.img1, p.img2):
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _sample_pairs_by_images(
    pairs: List[Pair],
    max_images: int,
    seed: int,
) -> List[Pair]:
    if max_images <= 0:
        raise ValueError("--max-images must be a positive integer")

    uniq = _iter_unique_images(pairs)
    if max_images >= len(uniq):
        return pairs

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(pairs)).tolist()
    chosen_imgs: set[str] = set()
    chosen_pairs: List[Pair] = []
    for idx in order:
        p = pairs[idx]
        cand = {p.img1, p.img2}
        new_imgs = cand - chosen_imgs
        if len(chosen_imgs) + len(new_imgs) > int(max_images):
            continue
        chosen_pairs.append(p)
        chosen_imgs.update(cand)
        if len(chosen_imgs) >= int(max_images):
            break

    if len(chosen_pairs) < 10:
        n_pairs = min(len(pairs), max(10, int(max_images)))
        idxs = rng.choice(len(pairs), size=n_pairs, replace=False)
        chosen_pairs = [pairs[i] for i in idxs.tolist()]
    return chosen_pairs


def _compute_scores(
    pairs: List[Pair],
    verifier: FaceVerifier,
    base_dir: Path,
    metric: str,
    use_detector: bool,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    detector = None
    if use_detector:
        from src.core.face_detector import FaceDetector

        detector = FaceDetector(device=device)

    emb_cache: Dict[str, np.ndarray] = {}
    for rel in tqdm(_iter_unique_images(pairs), desc="Extract embeddings", unit="img"):
        img_path = _resolve_image_path(base_dir, rel)
        face_rgb = _load_face_rgb(img_path, detector)
        emb_cache[rel] = verifier.extract_embedding(face_rgb)

    labels = np.array([p.label for p in pairs], dtype=np.int64)
    scores = np.empty((len(pairs),), dtype=np.float32)
    for i, p in enumerate(tqdm(pairs, desc=f"Compute scores ({metric})", unit="pair")):
        scores[i] = float(verifier.compare(emb_cache[p.img1], emb_cache[p.img2], metric=metric))
    return labels, scores


def _rates_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = int(labels.shape[0])
    acc = (tp + tn) / max(total, 1)
    far = fp / max(fp + tn, 1)  # false accept rate
    frr = fn / max(fn + tp, 1)  # false reject rate
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    tpr = recall
    tnr = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (tpr + tnr)
    return {
        "acc": float(acc),
        "far": float(far),
        "frr": float(frr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(balanced_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs",
        type=str,
        default=str(ROOT / "data" / "verification_pairs_val.txt"),
        help="Path to verification_pairs_val.txt",
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        default=str(ROOT / "data"),
        help="Base directory used to resolve relative image paths in pairs file",
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default="classifier",
        help="Model type: classifier|arcface|triplet|edgeface_xs_gamma_06|...",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to weights file (optional; EdgeFace can auto-download if omitted)",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Similarity metric for verification",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    ap.add_argument(
        "--use-detector",
        action="store_true",
        help="Run MTCNN face detection/cropping before embedding extraction",
    )
    ap.add_argument(
        "--num-classes",
        type=int,
        default=4000,
        help="Only used for classifier/arcface models",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold. If omitted, uses best_thr (Youden J).",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit: sample up to N pairs for quick evaluation.",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit: sample up to N unique images, then evaluate pairs within them.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --max-pairs is set.",
    )
    args = ap.parse_args()

    pairs_path = Path(args.pairs).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    weights_path = None if args.weights in (None, "", "none", "null") else str(Path(args.weights).expanduser().resolve())

    pairs = _read_pairs(pairs_path)
    if args.max_images is not None:
        pairs = _sample_pairs_by_images(pairs, int(args.max_images), int(args.seed))
    if args.max_pairs is not None:
        n = int(args.max_pairs)
        if n <= 0:
            raise ValueError("--max-pairs must be a positive integer")
        rng = np.random.default_rng(int(args.seed))
        if n < len(pairs):
            idx = rng.choice(len(pairs), size=n, replace=False)
            pairs = [pairs[i] for i in idx.tolist()]

    verifier = FaceVerifier(
        model_type=args.model_type,
        weights_path=weights_path,
        device=args.device,
        num_classes=int(args.num_classes),
    )

    labels, scores = _compute_scores(
        pairs=pairs,
        verifier=verifier,
        base_dir=base_dir,
        metric=args.metric,
        use_detector=bool(args.use_detector),
        device=args.device,
    )

    auc = compute_auc(labels, scores)
    fpr, tpr, thresholds = compute_roc(labels, scores)
    best_thr = find_best_threshold(fpr, tpr, thresholds)
    thr = float(best_thr) if args.threshold is None else float(args.threshold)
    rates = _rates_at_threshold(labels, scores, thr)

    print("=== Face Verification Evaluation ===")
    print(f"pairs:        {pairs_path}")
    print(f"base_dir:     {base_dir}")
    print(f"model_type:   {args.model_type}")
    print(f"weights:      {weights_path}")
    print(f"metric:       {args.metric}")
    print(f"use_detector: {bool(args.use_detector)}")
    print(f"device:       {args.device}")
    if args.max_images is not None:
        print(f"max_images:   {int(args.max_images)}")
    print(f"num_pairs:    {len(pairs)}")
    print()
    print(f"AUC:          {auc:.6f}")
    print(f"best_thr:     {best_thr:.6f}  (Youden J)")
    print(
        f"at thr={thr:.6f}:  "
        f"acc={rates['acc']:.4f}  bal_acc={rates['balanced_acc']:.4f}  "
        f"prec={rates['precision']:.4f}  rec={rates['recall']:.4f}  f1={rates['f1']:.4f}  "
        f"far={rates['far']:.4f}  frr={rates['frr']:.4f}  "
        f"(tp={rates['tp']} tn={rates['tn']} fp={rates['fp']} fn={rates['fn']} n={rates['n']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

