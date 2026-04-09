"""Evaluate verification metrics for multiple models."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
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


def read_pairs(pairs_path: Path) -> List[Pair]:
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


def sample_pairs(pairs: List[Pair], max_pairs: Optional[int], seed: int) -> List[Pair]:
    if max_pairs is None:
        return pairs
    n = int(max_pairs)
    if n <= 0:
        raise ValueError("--max-pairs must be a positive integer")
    if n >= len(pairs):
        return pairs
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[i] for i in idx.tolist()]


def sample_pairs_by_images(pairs: List[Pair], max_images: Optional[int], seed: int) -> List[Pair]:
    if max_images is None:
        return pairs
    n = int(max_images)
    if n <= 0:
        raise ValueError("--max-images must be a positive integer")
    uniq = iter_unique_images(pairs)
    if n >= len(uniq):
        return pairs

    # Build a subset of pairs such that the number of unique images is <= n.
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(pairs)).tolist()
    chosen_imgs: set[str] = set()
    chosen_pairs: List[Pair] = []
    for idx in order:
        p = pairs[idx]
        cand = {p.img1, p.img2}
        new_imgs = cand - chosen_imgs
        if len(chosen_imgs) + len(new_imgs) > n:
            continue
        chosen_pairs.append(p)
        chosen_imgs.update(cand)
        if len(chosen_imgs) >= n:
            break

    # Ensure we have at least a few pairs.
    if len(chosen_pairs) < 10:
        n_pairs = min(len(pairs), max(10, n))
        idxs = rng.choice(len(pairs), size=n_pairs, replace=False)
        chosen_pairs = [pairs[i] for i in idxs.tolist()]
    return chosen_pairs


def rates_at_threshold(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    preds = (scores >= thr).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = int(labels.shape[0])
    acc = (tp + tn) / max(total, 1)
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    tnr = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + tnr)
    return {
        "acc": float(acc),
        "balanced_acc": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": total,
    }


def iter_unique_images(pairs: Iterable[Pair]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in pairs:
        for x in (p.img1, p.img2):
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def resolve_image_path(base_dir: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_face_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    try:
        pil = Image.open(path).convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot read image: {path}") from e
    return np.array(pil)


def compute_scores(
    pairs: List[Pair],
    verifier: FaceVerifier,
    base_dir: Path,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    # Cache embeddings per image for speed.
    emb_cache: Dict[str, np.ndarray] = {}
    for rel in tqdm(iter_unique_images(pairs), desc="Extract embeddings", unit="img"):
        img_path = resolve_image_path(base_dir, rel)
        face_rgb = load_face_rgb(img_path)
        emb_cache[rel] = verifier.extract_embedding(face_rgb)

    labels = np.array([p.label for p in pairs], dtype=np.int64)
    scores = np.empty((len(pairs),), dtype=np.float32)
    for i, p in enumerate(tqdm(pairs, desc=f"Compute scores ({metric})", unit="pair")):
        scores[i] = float(verifier.compare(emb_cache[p.img1], emb_cache[p.img2], metric=metric))
    return labels, scores


def default_weight_map(weights_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "classifier": weights_dir / "face_classification.pth",
        "arcface": weights_dir / "face_classification_arc.pth",
        "triplet": weights_dir / "face_metric_learning.pth",
        "edgeface_xs_gamma_06": weights_dir / "edgeface_xs_gamma_06.pt",
    }


def as_md_table(rows: List[dict]) -> str:
    cols = [
        "model",
        "metric",
        "auc",
        "best_thr",
        "acc",
        "balanced_acc",
        "precision",
        "recall",
        "f1",
        "far",
        "frr",
        "n",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["metric"]),
                    f'{r["auc"]:.6f}',
                    f'{r["best_thr"]:.6f}',
                    f'{r["acc"]:.4f}',
                    f'{r["balanced_acc"]:.4f}',
                    f'{r["precision"]:.4f}',
                    f'{r["recall"]:.4f}',
                    f'{r["f1"]:.4f}',
                    f'{r["far"]:.4f}',
                    f'{r["frr"]:.4f}',
                    str(int(r["n"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=str(ROOT / "data" / "verification_pairs_val.txt"))
    ap.add_argument("--base-dir", type=str, default=str(ROOT / "data"))
    ap.add_argument("--weights-dir", type=str, default=str(ROOT / "weights"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--num-classes", type=int, default=4000)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        type=str,
        default="classifier,arcface,triplet",
        help="Comma-separated list. Add edgeface_xs_gamma_06 to include EdgeFace.",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="cosine,euclidean",
        help="Comma-separated list of metrics: cosine,euclidean",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "logs"),
        help="Directory to write CSV/Markdown report.",
    )
    args = ap.parse_args()

    pairs_path = Path(args.pairs).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = read_pairs(pairs_path)
    pairs = sample_pairs_by_images(all_pairs, args.max_images, args.seed)
    pairs = sample_pairs(pairs, args.max_pairs, args.seed)

    weight_map = default_weight_map(weights_dir)
    model_types = [m.strip() for m in str(args.models).split(",") if m.strip()]
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    rows: List[dict] = []
    for model_type in tqdm(model_types, desc="Models", unit="model"):
        wp = weight_map.get(model_type)
        weights_path: Optional[str]
        if wp is None:
            weights_path = None
        else:
            weights_path = str(wp) if wp.exists() else None
            if model_type not in ("edgeface_xs_gamma_06",) and weights_path is None:
                continue

        verifier = FaceVerifier(
            model_type=model_type,
            weights_path=weights_path,
            device=args.device,
            num_classes=int(args.num_classes),
        )

        for metric in tqdm(metrics, desc=f"Metrics ({model_type})", unit="metric", leave=False):
            labels, scores = compute_scores(pairs, verifier, base_dir, metric=metric)
            auc = compute_auc(labels, scores)
            fpr, tpr, thr = compute_roc(labels, scores)
            best_thr = find_best_threshold(fpr, tpr, thr)
            rates = rates_at_threshold(labels, scores, best_thr)
            rows.append(
                {
                    "model": model_type,
                    "metric": metric,
                    "auc": float(auc),
                    "best_thr": float(best_thr),
                    **rates,
                }
            )

    rows.sort(key=lambda r: (r["model"], r["metric"]))
    md = as_md_table(rows)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"pairs{len(pairs)}-seed{int(args.seed)}"
    md_path = out_dir / f"verification_report_{tag}_{stamp}.md"
    csv_path = out_dir / f"verification_report_{tag}_{stamp}.csv"

    md_path.write_text(
        "\n".join(
            [
                "# Verification report",
                "",
                f"- pairs: `{pairs_path}`",
                f"- base_dir: `{base_dir}`",
                f"- num_pairs: **{len(pairs)}**",
                f"- device: `{args.device}`",
                "",
                md,
                "",
            ]
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(md)
    print()
    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

