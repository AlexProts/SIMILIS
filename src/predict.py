from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from description import DEFAULT_CONFIDENCE_THRESHOLDS, build_auto_description
from data import TARGET_FIELDS, list_images
from model import SIMILISMultiTask
from transforms import build_transforms


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def normalize_decoders(raw_decoders: Dict[str, Dict[Any, str]]) -> Dict[str, Dict[int, str]]:
    return {
        field: {int(idx): label for idx, label in decoder.items()}
        for field, decoder in raw_decoders.items()
    }


def predict_image(
    model: SIMILISMultiTask,
    image_path: Path,
    transform,
    label_decoders: Dict[str, Dict[int, str]],
    thresholds: Dict[str, float],
    device: torch.device,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    row: Dict[str, Any] = {}
    fields: Dict[str, Dict[str, Any]] = {}
    for field in TARGET_FIELDS:
        probs = torch.softmax(logits[field][0], dim=0)
        pred_idx = int(probs.argmax().item())
        conf = float(probs[pred_idx].item())
        label = label_decoders[field][pred_idx]
        short = field.replace("_norm", "")
        row[f"pred_{short}"] = label
        row[f"confidence_{short}"] = conf
        fields[field] = {"label": label, "confidence": conf}

    row["auto_description"] = build_auto_description(
        fields["name_norm"]["label"],
        fields["name_norm"]["confidence"],
        fields["material_norm"]["label"],
        fields["material_norm"]["confidence"],
        fields["part_norm"]["label"],
        fields["part_norm"]["confidence"],
        fields["fragm_norm"]["label"],
        fields["fragm_norm"]["confidence"],
        thresholds=thresholds,
    )
    return row


def run_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)

    n_classes = ckpt.get("n_classes")
    if n_classes is None:
        label_encoders = ckpt["label_encoders"]
        n_classes = {field: len(label_encoders[field]) for field in TARGET_FIELDS}

    label_decoders = normalize_decoders(ckpt["label_decoders"])
    thresholds = ckpt.get("confidence_thresholds", DEFAULT_CONFIDENCE_THRESHOLDS)
    image_size = int(args.image_size or ckpt.get("image_size", 384))

    _, eval_transform = build_transforms(image_size=image_size, pad_fill=args.pad_fill)
    model = SIMILISMultiTask(n_classes=n_classes, pretrained=False, drop_rate=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image_dir = Path(args.image_dir)
    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rows = []
    for image_path in tqdm(image_paths, desc="predict"):
        row = predict_image(
            model=model,
            image_path=image_path,
            transform=eval_transform,
            label_decoders=label_decoders,
            thresholds=thresholds,
            device=device,
        )
        row["image_file"] = image_path.relative_to(image_dir).as_posix()
        rows.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_file", "auto_description"] + [
        col for col in rows[0].keys() if col.startswith("pred_") or col.startswith("confidence_")
    ]
    pd.DataFrame(rows)[columns].to_csv(output_path, index=False)
    print(f"saved {output_path} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIMILIS baseline inference")
    parser.add_argument("--image-dir", required=True, help="Directory with input images")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")
    parser.add_argument("--image-size", type=int, default=None, help="Override checkpoint image size")
    parser.add_argument("--pad-fill", type=int, default=255)
    parser.add_argument("--device", default=None, help="cuda, cpu, or empty for auto")
    return parser.parse_args()


if __name__ == "__main__":
    run_predict(parse_args())
