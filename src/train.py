from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import SIMILISDataset, TARGET_FIELDS
from description import DEFAULT_CONFIDENCE_THRESHOLDS
from model import SIMILISMultiTask
from transforms import build_transforms


class MaskedMultiTaskLoss(nn.Module):
    def __init__(self, field_names, class_weights: Dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.field_names = tuple(field_names)
        self.criteria = nn.ModuleDict()
        for field in self.field_names:
            weight = class_weights[field] if class_weights and field in class_weights else None
            self.criteria[field] = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets, masks):
        total = torch.zeros((), device=next(iter(logits.values())).device)
        per_field = {}
        for field in self.field_names:
            ce = self.criteria[field](logits[field], targets[field])
            mask = masks[field].to(ce.device)
            loss = (ce * mask).sum() / mask.sum().clamp(min=1.0)
            per_field[field] = loss
            total = total + loss
        return total, per_field


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_class_weights(df: pd.DataFrame, field: str, encoder: dict, cap: float = 10.0):
    counts = df[f"{field}_idx"].dropna().value_counts()
    n_total = counts.sum()
    n_classes = len(encoder)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for label, idx in encoder.items():
        count = counts.get(idx, counts.get(label, 1))
        weights[int(idx)] = min(float(n_total / (n_classes * max(1, count))), cap)
    return weights


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds = {field: [] for field in TARGET_FIELDS}
    targets = {field: [] for field in TARGET_FIELDS}
    masks = {field: [] for field in TARGET_FIELDS}
    total_loss = 0.0
    n = 0

    for images, batch_targets, batch_masks, _ in loader:
        images = images.to(device)
        targets_dev = {f: t.to(device) for f, t in batch_targets.items()}
        masks_dev = {f: m.to(device) for f, m in batch_masks.items()}
        logits = model(images)
        loss, _ = criterion(logits, targets_dev, masks_dev)
        total_loss += float(loss.item()) * images.size(0)
        n += images.size(0)
        for field in TARGET_FIELDS:
            preds[field].append(logits[field].argmax(dim=1).cpu())
            targets[field].append(batch_targets[field])
            masks[field].append(batch_masks[field])

    metrics = {"loss": total_loss / max(1, n)}
    f1_values = []
    for field in TARGET_FIELDS:
        p = torch.cat(preds[field]).numpy()
        t = torch.cat(targets[field]).numpy()
        m = torch.cat(masks[field]).bool().numpy()
        metrics[f"acc_{field}"] = accuracy_score(t[m], p[m])
        metrics[f"macro_f1_{field}"] = f1_score(t[m], p[m], average="macro", zero_division=0)
        f1_values.append(metrics[f"macro_f1_{field}"])
    metrics["mean_macro_f1"] = float(np.mean(f1_values))
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets, masks, _ in loader:
        images = images.to(device)
        targets = {f: t.to(device) for f, t in targets.items()}
        masks = {f: m.to(device) for f, m in masks.items()}
        logits = model(images)
        loss, _ = criterion(logits, targets, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        n += images.size(0)
    return total_loss / max(1, n)


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    df = pd.read_csv(args.data_csv)
    label_encoders = load_json(args.label_encoders)
    label_encoders = {
        field: {label: int(idx) for label, idx in encoder.items()}
        for field, encoder in label_encoders.items()
    }
    label_decoders = {
        field: {idx: label for label, idx in encoder.items()}
        for field, encoder in label_encoders.items()
    }
    n_classes = {field: len(label_encoders[field]) for field in TARGET_FIELDS}

    train_transform, eval_transform = build_transforms(args.image_size, args.pad_fill)
    train_loader = DataLoader(
        SIMILISDataset(df[df["split"] == "train"], train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        SIMILISDataset(df[df["split"] == "val"], eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    class_weights = {
        field: compute_class_weights(df[df["split"] == "train"], field, label_encoders[field]).to(device)
        for field in TARGET_FIELDS
    }
    train_criterion = MaskedMultiTaskLoss(TARGET_FIELDS, class_weights).to(device)
    val_criterion = MaskedMultiTaskLoss(TARGET_FIELDS).to(device)

    model = SIMILISMultiTask(n_classes=n_classes, pretrained=True, drop_rate=args.drop_rate).to(device)
    model.freeze_backbone_stages(unfreeze_from_stage=args.unfreeze_from_stage)
    optimizer = AdamW(model.get_param_groups(args.lr_backbone, args.lr_heads, args.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, val_criterion, device)
        scheduler.step()

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": max(best_metric, val_metrics["mean_macro_f1"]),
            "label_encoders": label_encoders,
            "label_decoders": label_decoders,
            "n_classes": n_classes,
            "target_fields": TARGET_FIELDS,
            "confidence_thresholds": DEFAULT_CONFIDENCE_THRESHOLDS,
            "image_size": args.image_size,
        }
        torch.save(payload, output_dir / "last.pt")
        if val_metrics["mean_macro_f1"] > best_metric:
            best_metric = val_metrics["mean_macro_f1"]
            payload["best_metric"] = best_metric
            torch.save(payload, output_dir / "best.pt")

        print(
            f"epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mean_macro_f1={val_metrics['mean_macro_f1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SIMILIS baseline")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--label-encoders", required=True)
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--pad-fill", type=int, default=255)
    parser.add_argument("--lr-backbone", type=float, default=3e-5)
    parser.add_argument("--lr-heads", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--unfreeze-from-stage", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_train(parse_args())
