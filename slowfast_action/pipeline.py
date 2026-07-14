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
        model = slowfast_r50(pretrained=self.config.pretrained)
        model.blocks[-1].proj = nn.Identity()
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

    def _transform_clip(self, video: EncodedVideo, clip_start: float, clip_end: float) -> List[torch.Tensor]:
        clip = video.get_clip(start_sec=clip_start, end_sec=clip_end)
        if clip is None or "video" not in clip or clip["video"] is None:
            raise RuntimeError(f"Khong doc duoc clip trong khoang {clip_start:.3f}-{clip_end:.3f}")
        clip = self.transform(clip)
        return clip["video"]

    def _extract_subshot_features_batched(
        self,
        video: EncodedVideo,
        subshot_records: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        all_features: List[np.ndarray] = []

        for batch_start in range(0, len(subshot_records), self.config.batch_size):
            batch_records = subshot_records[batch_start:batch_start + self.config.batch_size]
            slow_batch = []
            fast_batch = []

            for record in batch_records:
                slow_pathway, fast_pathway = self._transform_clip(
                    video,
                    record["clip_start"],
                    record["clip_end"],
                )
                slow_batch.append(slow_pathway)
                fast_batch.append(fast_pathway)

            model_inputs = [
                torch.stack(slow_batch, dim=0).to(self.device),
                torch.stack(fast_batch, dim=0).to(self.device),
            ]

            with torch.no_grad():
                batch_features = self.model(model_inputs)

            batch_features = batch_features.detach().cpu().numpy().astype(np.float32)
            all_features.extend(feature for feature in batch_features)

        return all_features

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

    def _build_shot_outputs(
        self,
        shots: List[Dict[str, Any]],
        subshot_records: List[Dict[str, Any]],
        subshot_features: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        features_by_shot: Dict[int, List[np.ndarray]] = {}
        for record, feature in zip(subshot_records, subshot_features):
            features_by_shot.setdefault(record["shot_id"], []).append(feature)

        shot_outputs = []
        for shot in shots:
            shot_id = int(shot["shot_id"])
            pooled_feature = self._pool_subshot_features(features_by_shot[shot_id])
            shot_outputs.append(
                {
                    "shot_id": shot_id,
                    "start_frame": int(shot["start_frame"]),
                    "end_frame": int(shot["end_frame"]),
                    "start_time_sec": float(shot["start_time_sec"]),
                    "end_time_sec": float(shot["end_time_sec"]),
                    "duration_sec": float(shot["duration_sec"]),
                    "num_subshots": int(len(features_by_shot[shot_id])),
                    "pooling": "max",
                    "action_feature": self._cast_output_feature(pooled_feature),
                }
            )

        return shot_outputs

    def process_video_item(self, item: Dict[str, str]) -> Dict[str, Any]:
        shots = self.shot_loader.load(item["shots_json_path"])
        video = EncodedVideo.from_path(item["video_path"])
        subshot_records = self._build_subshot_records(shots)
        subshot_features = self._extract_subshot_features_batched(video, subshot_records)
        shot_features = self._build_shot_outputs(shots, subshot_records, subshot_features)

        feature_dim = int(shot_features[0]["action_feature"].shape[0]) if shot_features else 0
        output = {
            "video_name": item["video_name"],
            "shots_json_path": item["shots_json_path"],
            "video_path": item["video_path"],
            "model_name": "slowfast_r50",
            "pretrained": bool(self.config.pretrained),
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
