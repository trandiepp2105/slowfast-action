from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import SlowFastShotFeatureConfig


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


def temporal_resample_with_repeat_batch(video_batch: torch.Tensor, num_frames: int) -> torch.Tensor:
    total_frames = video_batch.shape[2]
    if total_frames <= 0:
        raise ValueError("Batch clip khong co frame hop le")

    indices = torch.linspace(
        0,
        total_frames - 1,
        num_frames,
        device=video_batch.device,
    ).round().long()
    return torch.index_select(video_batch, 2, indices)


def short_side_scale_batch(video_batch: torch.Tensor, size: int) -> torch.Tensor:
    _, _, _, height, width = video_batch.shape
    if min(height, width) == size:
        return video_batch

    if height < width:
        new_height = size
        new_width = int(round(width * size / height))
    else:
        new_width = size
        new_height = int(round(height * size / width))

    batch_size, channels, frames, _, _ = video_batch.shape
    flat = video_batch.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
    flat = F.interpolate(
        flat,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    return flat.reshape(batch_size, frames, channels, new_height, new_width).permute(0, 2, 1, 3, 4)


def uniform_crop_batch(video_batch: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, _, height, width = video_batch.shape
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return video_batch[:, :, :, top:top + crop_size, left:left + crop_size]


def pack_pathway_batch(video_batch: torch.Tensor, alpha: int) -> List[torch.Tensor]:
    fast_pathway = video_batch
    slow_count = max(1, video_batch.shape[2] // alpha)
    slow_indices = torch.linspace(
        0,
        video_batch.shape[2] - 1,
        slow_count,
        device=video_batch.device,
    ).long()
    slow_pathway = torch.index_select(video_batch, 2, slow_indices)
    return [slow_pathway, fast_pathway]


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= eps:
        return vec
    return vec / norm


class BatchedVideoTransform(nn.Module):
    def __init__(self, config: SlowFastShotFeatureConfig):
        super().__init__()
        self.num_frames = int(config.num_frames)
        self.side_size = int(config.side_size)
        self.crop_size = int(config.crop_size)
        self.alpha = int(config.alpha)
        self.register_buffer("mean", torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1))

    def forward(self, video_batch: torch.Tensor) -> List[torch.Tensor]:
        video_batch = temporal_resample_with_repeat_batch(video_batch, self.num_frames)
        video_batch = video_batch.float().div_(255.0)
        video_batch = (video_batch - self.mean) / self.std
        video_batch = short_side_scale_batch(video_batch, self.side_size)
        video_batch = uniform_crop_batch(video_batch, self.crop_size)
        return pack_pathway_batch(video_batch, self.alpha)


def build_batched_video_transform(config: SlowFastShotFeatureConfig) -> BatchedVideoTransform:
    return BatchedVideoTransform(config)


def temporal_resample_single_video(video_tensor: torch.Tensor, num_frames: int) -> torch.Tensor:
    return temporal_resample_with_repeat(video_tensor, num_frames)
