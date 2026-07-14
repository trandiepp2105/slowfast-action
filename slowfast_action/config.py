from dataclasses import dataclass
from typing import Optional


@dataclass
class SlowFastShotFeatureConfig:
    video_dataset_dir: str
    shots_json_dir: str
    output_dir: str
    start_index: int = 0
    end_index: Optional[int] = None
    device: str = "cuda"
    pretrained: bool = True
    model_path: Optional[str] = None
    num_frames: int = 32
    sampling_rate: int = 2
    alpha: int = 4
    side_size: int = 256
    crop_size: int = 256
    target_fps: float = 30.0
    clip_duration_sec: float = 32 * 2 / 30.0
    batch_size: int = 8
    save_dtype: str = "float16"
    overwrite: bool = False
