from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Lambda

from .config import SlowFastShotFeatureConfig


class ApplyTransformToKey:
    def __init__(self, key: str, transform):
        self.key = key
        self.transform = transform

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        output = dict(data)
        output[self.key] = self.transform(output[self.key])
        return output


class ShortSideScale(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        if min(height, width) == self.size:
            return frames

        if height < width:
            new_height = self.size
            new_width = int(round(width * self.size / height))
        else:
            new_width = self.size
            new_height = int(round(height * self.size / width))

        return F.interpolate(
            frames,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )


class PackPathway(nn.Module):
    def __init__(self, alpha: int):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor) -> List[torch.Tensor]:
        fast_pathway = frames
        slow_count = max(1, frames.shape[1] // self.alpha)
        slow_indices = torch.linspace(0, frames.shape[1] - 1, slow_count).long()
        slow_pathway = torch.index_select(frames, 1, slow_indices)
        return [slow_pathway, fast_pathway]


def temporal_resample_with_repeat(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    total_frames = frames.shape[1]
    if total_frames <= 0:
        raise ValueError("Clip khong co frame hop le")

    indices = torch.linspace(0, total_frames - 1, num_frames).round().long()
    return torch.index_select(frames, 1, indices)


def uniform_crop(frames: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, height, width = frames.shape
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return frames[:, :, top:top + crop_size, left:left + crop_size]


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= eps:
        return vec
    return vec / norm


def build_video_transform(config: SlowFastShotFeatureConfig):
    mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                Lambda(lambda x: temporal_resample_with_repeat(x, config.num_frames)),
                Lambda(lambda x: x / 255.0),
                Lambda(lambda x: (x - mean) / std),
                ShortSideScale(size=config.side_size),
                Lambda(lambda x: uniform_crop(x, config.crop_size)),
                PackPathway(config.alpha),
            ]
        ),
    )
