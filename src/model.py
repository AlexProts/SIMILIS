from __future__ import annotations

from typing import Dict, List

import timm
import torch
import torch.nn as nn


class SIMILISMultiTask(nn.Module):
    def __init__(
        self,
        n_classes: Dict[str, int],
        backbone_name: str = "convnext_tiny.fb_in22k_ft_in1k",
        pretrained: bool = False,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=0.0,
        )
        self.feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(self.feat_dim, n) for field, n in n_classes.items()}
        )
        self._field_order: List[str] = list(n_classes.keys())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.dropout(features)
        return {field: self.heads[field](features) for field in self._field_order}

    def freeze_backbone_stages(self, unfreeze_from_stage: int = 3) -> None:
        if hasattr(self.backbone, "stem"):
            for param in self.backbone.stem.parameters():
                param.requires_grad = False
        if hasattr(self.backbone, "stages"):
            for stage_idx, stage in enumerate(self.backbone.stages):
                trainable = stage_idx >= unfreeze_from_stage
                for param in stage.parameters():
                    param.requires_grad = trainable
        if hasattr(self.backbone, "norm_pre"):
            for param in self.backbone.norm_pre.parameters():
                param.requires_grad = True

    def get_param_groups(
        self,
        lr_backbone: float = 3e-5,
        lr_heads: float = 3e-4,
        weight_decay: float = 1e-4,
    ) -> List[dict]:
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.heads.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
        ]
