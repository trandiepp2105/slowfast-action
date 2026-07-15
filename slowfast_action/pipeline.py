import math
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from .config import SlowFastShotFeatureConfig
from .io_utils import DatasetScanner, ShotLoader
from .transforms import build_video_transform, l2_normalize

try:
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.models.hub import slowfast_r50
except ImportError as error:
    raise ImportError(
        "Can cai pytorchvideo truoc khi chay module nay. "
        "Vi du: pip install pytorchvideo torchvision av"
    ) from error


class SlowFastShotFeatureExtractor:
    def __init__(self, config: SlowFastShotFeatureConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.transform = build_video_transform(config)
        self.model = self._load_model()
        self.scanner = DatasetScanner(config)
        self.shot_loader = ShotLoader()

    def _load_model(self):
        use_pretrained = self.config.pretrained and self.config.model_path is None
        model = slowfast_r50(pretrained=use_pretrained)
        model.blocks[-1].proj = nn.Identity()

        if self.config.model_path is not None:
            checkpoint = torch.load(self.config.model_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=True)

        model = model.to(self.device)
        model.eval()
        return model

    def _build_subshot_ranges(self, shot: Dict[str, Any]) -> List[tuple[float, float]]:
        start_sec = float(shot["start_time_sec"])
        end_sec = float(shot["end_time_sec"])
        if end_sec <= start_sec:
            end_sec = start_sec + 1e-3

        shot_duration = end_sec - start_sec
        if shot_duration <= self.config.clip_duration_sec:
            return [(start_sec, end_sec)]

        num_subshots = int(math.ceil(shot_duration / self.config.clip_duration_sec))
        boundaries = np.linspace(start_sec, end_sec, num_subshots + 1)
        ranges = []
        for idx in range(num_subshots):
            sub_start = float(boundaries[idx])
            sub_end = float(boundaries[idx + 1])
            if sub_end <= sub_start:
                sub_end = sub_start + 1e-3
            ranges.append((sub_start, sub_end))
        return ranges

    def _build_subshot_records(self, shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        subshot_records = []
        for shot in shots:
            for subshot_index, (clip_start, clip_end) in enumerate(self._build_subshot_ranges(shot)):
                subshot_records.append(
                    {
                        "shot_id": int(shot["shot_id"]),
                        "subshot_index": int(subshot_index),
                        "clip_start": float(clip_start),
                        "clip_end": float(clip_end),
                    }
                    )
        return subshot_records

    def _decode_video_tensor(
        self,
        video: EncodedVideo,
        clip_start: float,
        clip_end: float,
    ) -> torch.Tensor:
        clip = video.get_clip(start_sec=clip_start, end_sec=clip_end)
        if clip is None or "video" not in clip or clip["video"] is None:
            raise RuntimeError(f"Khong doc duoc clip trong khoang {clip_start:.3f}-{clip_end:.3f}")
        return clip["video"]

    def _slice_subshot_video_tensor(
        self,
        shot_video: torch.Tensor,
        shot_start: float,
        shot_end: float,
        subshot_start: float,
        subshot_end: float,
    ) -> torch.Tensor:
        total_frames = int(shot_video.shape[1])
        if total_frames <= 0:
            raise RuntimeError("Shot video tensor khong co frame de cat subshot")

        shot_duration = max(float(shot_end - shot_start), 1e-3)
        rel_start = np.clip((subshot_start - shot_start) / shot_duration, 0.0, 1.0)
        rel_end = np.clip((subshot_end - shot_start) / shot_duration, 0.0, 1.0)

        start_index = int(math.floor(rel_start * total_frames))
        end_index = int(math.ceil(rel_end * total_frames))

        start_index = min(max(start_index, 0), total_frames - 1)
        end_index = min(max(end_index, start_index + 1), total_frames)
        return shot_video[:, start_index:end_index]

    def _transform_video_tensor(self, video_tensor: torch.Tensor) -> List[torch.Tensor]:
        clip = self.transform({"video": video_tensor})
        return clip["video"]

    def _extract_subshot_features_from_shot(
        self,
        shot: Dict[str, Any],
        shot_video: torch.Tensor,
    ) -> tuple[List[Dict[str, Any]], List[np.ndarray]]:
        subshot_ranges = self._build_subshot_ranges(shot)
        subshot_records: List[Dict[str, Any]] = []
        subshot_features: List[np.ndarray] = []
        shot_start = float(shot["start_time_sec"])
        shot_end = float(shot["end_time_sec"])

        for batch_start in range(0, len(subshot_ranges), self.config.batch_size):
            batch_ranges = subshot_ranges[batch_start:batch_start + self.config.batch_size]
            slow_batch = []
            fast_batch = []

            for local_index, (clip_start, clip_end) in enumerate(batch_ranges):
                subshot_index = batch_start + local_index
                subshot_video = self._slice_subshot_video_tensor(
                    shot_video=shot_video,
                    shot_start=shot_start,
                    shot_end=shot_end,
                    subshot_start=float(clip_start),
                    subshot_end=float(clip_end),
                )
                slow_pathway, fast_pathway = self._transform_video_tensor(subshot_video)
                slow_batch.append(slow_pathway)
                fast_batch.append(fast_pathway)
                subshot_records.append(
                    {
                        "shot_id": int(shot["shot_id"]),
                        "subshot_index": int(subshot_index),
                        "clip_start": float(clip_start),
                        "clip_end": float(clip_end),
                    }
                )

            model_inputs = [
                torch.stack(slow_batch, dim=0).to(self.device),
                torch.stack(fast_batch, dim=0).to(self.device),
            ]

            with torch.no_grad():
                batch_features = self.model(model_inputs)

            batch_features = batch_features.detach().cpu().numpy().astype(np.float32)
            subshot_features.extend(feature for feature in batch_features)

        return subshot_records, subshot_features

    def _pool_subshot_features(self, subshot_features: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(subshot_features, axis=0).astype(np.float32)
        pooled = stacked.max(axis=0)
        return l2_normalize(pooled.astype(np.float32))

    def _cast_output_feature(self, feature: np.ndarray) -> np.ndarray:
        if self.config.save_dtype == "float16":
            return feature.astype(np.float16)
        if self.config.save_dtype == "float32":
            return feature.astype(np.float32)
        raise ValueError(f"Unsupported save_dtype: {self.config.save_dtype}")

    def process_video_item(self, item: Dict[str, str]) -> Dict[str, Any]:
        shots = self.shot_loader.load(item["shots_json_path"])
        video = EncodedVideo.from_path(item["video_path"])
        subshot_records: List[Dict[str, Any]] = []
        shot_features: List[Dict[str, Any]] = []

        for shot in shots:
            shot_video = self._decode_video_tensor(
                video=video,
                clip_start=float(shot["start_time_sec"]),
                clip_end=float(shot["end_time_sec"]),
            )
            shot_subshot_records, shot_subshot_features = self._extract_subshot_features_from_shot(
                shot=shot,
                shot_video=shot_video,
            )
            pooled_feature = self._pool_subshot_features(shot_subshot_features)
            subshot_records.extend(shot_subshot_records)
            shot_features.append(
                {
                    "shot_id": int(shot["shot_id"]),
                    "start_frame": int(shot["start_frame"]),
                    "end_frame": int(shot["end_frame"]),
                    "start_time_sec": float(shot["start_time_sec"]),
                    "end_time_sec": float(shot["end_time_sec"]),
                    "duration_sec": float(shot["duration_sec"]),
                    "num_subshots": int(len(shot_subshot_records)),
                    "pooling": "max",
                    "action_feature": self._cast_output_feature(pooled_feature),
                }
            )

        feature_dim = int(shot_features[0]["action_feature"].shape[0]) if shot_features else 0
        output = {
            "video_name": item["video_name"],
            "shots_json_path": item["shots_json_path"],
            "video_path": item["video_path"],
            "model_name": "slowfast_r50",
            "pretrained": bool(self.config.pretrained),
            "model_path": self.config.model_path,
            "num_frames": int(self.config.num_frames),
            "sampling_rate": int(self.config.sampling_rate),
            "alpha": int(self.config.alpha),
            "target_fps": float(self.config.target_fps),
            "clip_duration_sec": float(self.config.clip_duration_sec),
            "batch_size": int(self.config.batch_size),
            "feature_dim": feature_dim,
            "feature_dtype": self.config.save_dtype,
            "num_shots": len(shot_features),
            "num_subshots_total": int(len(subshot_records)),
            "subshot_pooling": "max",
            "shots": shot_features,
        }

        output_path = Path(item["output_pkl_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(output, file)

        return output

    def run(self) -> Dict[str, Any]:
        items = self.scanner.get_video_items()
        print(f"Found {len(items)} videos to process")

        summary: Dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for item in items:
            output_path = Path(item["output_pkl_path"])

            try:
                if output_path.exists() and not self.config.overwrite:
                    print(f"[SKIP] {item['video_name']}")
                    summary["skipped"].append(item["video_name"])
                    continue

                output = self.process_video_item(item)
                print(
                    f"[DONE] {item['video_name']} | num_shots={output['num_shots']} | "
                    f"num_subshots_total={output['num_subshots_total']} | feature_dim={output['feature_dim']}"
                )
                summary["done"].append(
                    {
                        "video_name": item["video_name"],
                        "output_pkl_path": item["output_pkl_path"],
                        "num_shots": output["num_shots"],
                        "num_subshots_total": output["num_subshots_total"],
                        "feature_dim": output["feature_dim"],
                    }
                )
            except Exception as error:
                print(f"[FAILED] {item['video_name']}: {error}")
                summary["failed"].append(
                    {
                        "video_name": item["video_name"],
                        "video_path": item["video_path"],
                        "shots_json_path": item["shots_json_path"],
                        "output_pkl_path": item["output_pkl_path"],
                        "error": repr(error),
                    }
                )

        return summary
