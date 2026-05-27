from __future__ import annotations

from typing import Tuple

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeLongestSide:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if width >= height:
            new_width = self.size
            new_height = max(1, round(height * self.size / width))
        else:
            new_height = self.size
            new_width = max(1, round(width * self.size / height))
        return img.resize((new_width, new_height), Image.BILINEAR)


class PadToSquare:
    def __init__(self, size: int, fill: int = 255) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        pad_width = self.size - width
        pad_height = self.size - height
        if pad_width < 0 or pad_height < 0:
            raise ValueError(f"Image {(width, height)} is larger than target {self.size}")
        padding = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        return TF.pad(img, padding, fill=self.fill)


def build_transforms(
    image_size: int = 384,
    pad_fill: int = 255,
) -> Tuple[T.Compose, T.Compose]:
    base = [ResizeLongestSide(image_size), PadToSquare(image_size, fill=pad_fill)]

    train_transform = T.Compose(
        base
        + [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10, fill=pad_fill),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = T.Compose(
        base
        + [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
