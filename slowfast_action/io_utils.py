import json
from pathlib import Path
from typing import Any, Dict, List

from .config import SlowFastShotFeatureConfig


class DatasetScanner:
    def __init__(self, config: SlowFastShotFeatureConfig):
        self.config = config

    def get_video_items(self) -> List[Dict[str, str]]:
        video_paths = sorted(Path(self.config.video_dataset_dir).glob("*.mp4"))
        items: List[Dict[str, str]] = []

        for video_path in video_paths:
            video_name = video_path.stem
            shots_path = Path(self.config.shots_json_dir) / f"{video_name}.json"
            output_path = Path(self.config.output_dir) / f"{video_name}_slowfast_r50_shot_features.pkl"

            if not shots_path.exists():
                print(f"[WARN] Missing shots JSON for {video_name}: {shots_path}")
                continue

            items.append(
                {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "shots_json_path": str(shots_path),
                    "output_pkl_path": str(output_path),
                }
            )

        if self.config.video_ids:
            wanted = {str(video_id) for video_id in self.config.video_ids}
            return [
                item
                for item in items
                if item["video_name"] in wanted or Path(item["video_path"]).name in wanted
            ]

        start = max(0, int(self.config.start_index))
        end = self.config.end_index

        if start >= len(items):
            return []
        if end is None:
            return items[start:]
        return items[start:end + 1]


class ShotLoader:
    def load(self, shots_json_path: str) -> List[Dict[str, Any]]:
        shots = json.loads(Path(shots_json_path).read_text())
        if not isinstance(shots, list):
            raise ValueError("File shots JSON phai la list cac shot")
        return shots
