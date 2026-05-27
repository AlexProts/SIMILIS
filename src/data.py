from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


TARGET_FIELDS = ["name_norm", "fragm_norm", "material_norm", "part_norm"]
MASKABLE_FIELDS = {"part_norm"}


class SIMILISDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable[[Image.Image], torch.Tensor],
        target_fields: Tuple[str, ...] = tuple(TARGET_FIELDS),
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_fields = target_fields

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_file"]).convert("RGB")
        image = self.transform(image)

        targets: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        for field in self.target_fields:
            raw_idx = row[f"{field}_idx"]
            if pd.isna(raw_idx):
                targets[field] = torch.tensor(0, dtype=torch.long)
                masks[field] = torch.tensor(0.0, dtype=torch.float32)
            else:
                targets[field] = torch.tensor(int(raw_idx), dtype=torch.long)
                missing = field in MASKABLE_FIELDS and bool(row.get("part_is_missing", 0))
                masks[field] = torch.tensor(0.0 if missing else 1.0, dtype=torch.float32)

        meta = {
            "code": str(row.get("code", "")),
            "image_file": str(row["image_file"]),
            "group_key": str(row.get("group_key", "")),
        }
        return image, targets, masks, meta


def list_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)
