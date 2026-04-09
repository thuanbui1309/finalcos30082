"""Evaluate classification accuracy on val/test splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import ArcFaceClassifier, FaceClassifier
from src.utils.transforms import get_val_transforms


@dataclass(frozen=True)
class Result:
    split: str
    model_type: str
    top1: float
    top5: float
    n: int


def _load_classifier(
    model_type: str,
    weights_path: Path,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    if model_type == "classifier":
        model = FaceClassifier(num_classes=num_classes, embedding_dim=512, backbone="resnet50")
    elif model_type == "arcface":
        model = ArcFaceClassifier(num_classes=num_classes, embedding_dim=512, backbone="resnet50")
    else:
        raise ValueError("model_type must be 'classifier' or 'arcface'")

    state = torch.load(str(weights_path), map_location=device)
    if model_type == "arcface":
        # In `FaceVerifier`, arcface weights may be saved with 'head.' prefix.
        remapped = {}
        for k, v in state.items():
            if k.startswith("head."):
                remapped[k.replace("head.", "arcface_head.", 1)] = v
            else:
                remapped[k] = v
        state = remapped

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _arcface_logits(model: ArcFaceClassifier, x: torch.Tensor) -> torch.Tensor:
    emb = model.extract_embedding(x)  # (B, D), L2-normalised
    w = F.normalize(model.arcface_head.weight, dim=1)  # (C, D)
    cosine = F.linear(emb, w)  # (B, C)
    return cosine * float(model.arcface_head.s)


@torch.no_grad()
def _eval_split(
    split_name: str,
    split_dir: Path,
    model_type: str,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Result:
    ds = datasets.ImageFolder(str(split_dir), transform=get_val_transforms(img_size=112))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    correct1 = 0
    correct5 = 0
    n = 0

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)

        if model_type == "classifier":
            logits = model(xb)
        else:
            logits = _arcface_logits(model, xb)  # type: ignore[arg-type]

        # Top-1
        pred1 = logits.argmax(dim=1)
        correct1 += int((pred1 == yb).sum().item())

        # Top-5
        k = min(5, logits.shape[1])
        topk = logits.topk(k=k, dim=1).indices
        correct5 += int((topk == yb.unsqueeze(1)).any(dim=1).sum().item())

        n += int(yb.shape[0])

    top1 = correct1 / max(n, 1)
    top5 = correct5 / max(n, 1)
    return Result(split=split_name, model_type=model_type, top1=float(top1), top5=float(top5), n=n)


def _format_md_table(results: List[Result]) -> str:
    cols = ["model", "split", "top1", "top5", "n"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.model_type,
                    r.split,
                    f"{r.top1:.4f}",
                    f"{r.top5:.4f}",
                    str(r.n),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    ap.add_argument("--weights-dir", type=str, default=str(ROOT / "weights"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--num-classes", type=int, default=4000)
    ap.add_argument(
        "--models",
        type=str,
        default="classifier,arcface",
        help="Comma-separated: classifier,arcface",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated: val,test",
    )
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "logs"))
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    weights_dir = Path(args.weights_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    split_map: Dict[str, Path] = {
        "val": data_root / "classification_data" / "val_data",
        "test": data_root / "classification_data" / "test_data",
    }
    weight_map: Dict[str, Path] = {
        "classifier": weights_dir / "face_classification.pth",
        "arcface": weights_dir / "face_classification_arc.pth",
    }

    model_types = [m.strip() for m in str(args.models).split(",") if m.strip()]
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    results: List[Result] = []
    for mt in model_types:
        wp = weight_map.get(mt)
        if wp is None or not wp.exists():
            continue
        model = _load_classifier(
            model_type=mt,
            weights_path=wp,
            num_classes=int(args.num_classes),
            device=device,
        )

        for sp in splits:
            sd = split_map.get(sp)
            if sd is None or not sd.exists():
                continue
            results.append(
                _eval_split(
                    split_name=sp,
                    split_dir=sd,
                    model_type=mt,
                    model=model,
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
            )

    results.sort(key=lambda r: (r.model_type, r.split))
    md = _format_md_table(results)

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"classification_accuracy_{ts}.md"
    md_path.write_text(
        "\n".join(
            [
                "# Classification accuracy report",
                "",
                f"- data_root: `{data_root}`",
                f"- weights_dir: `{weights_dir}`",
                f"- device: `{device}`",
                "",
                md,
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(md)
    print()
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

