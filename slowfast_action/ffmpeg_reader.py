import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


class FFmpegVideoReader:
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.video_width, self.video_height, self.video_fps = self._probe_video(self.video_path)

    def _probe_video(self, video_path: str) -> tuple[int, int, float]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if len(streams) == 0:
            raise RuntimeError(f"ffprobe khong tim thay video stream trong file: {video_path}")

        stream = streams[0]
        width = int(stream["width"])
        height = int(stream["height"])
        fps_text = str(stream.get("r_frame_rate", "0/1"))
        fps = float(Fraction(fps_text)) if fps_text not in {"", "0/0"} else 0.0
        return width, height, fps

    def get_clip(self, start_sec: float, end_sec: float) -> torch.Tensor:
        if end_sec <= start_sec:
            end_sec = start_sec + 1e-3

        duration_sec = end_sec - start_sec
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            self.video_path,
            "-an",
            "-sn",
            "-dn",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, check=True)
        raw = result.stdout
        bytes_per_frame = self.video_width * self.video_height * 3
        if bytes_per_frame <= 0:
            raise RuntimeError(f"Kich thuoc frame khong hop le cho video: {self.video_path}")
        if len(raw) < bytes_per_frame:
            raise RuntimeError(f"Khong doc duoc clip trong khoang {start_sec:.3f}-{end_sec:.3f}")

        frame_count = len(raw) // bytes_per_frame
        usable_bytes = frame_count * bytes_per_frame
        frames = np.frombuffer(raw[:usable_bytes], dtype=np.uint8)
        frames = frames.reshape(frame_count, self.video_height, self.video_width, 3)
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
        return video_tensor


def ensure_ffmpeg_available():
    for binary_name in ("ffmpeg", "ffprobe"):
        command = [binary_name, "-version"]
        try:
            subprocess.run(command, capture_output=True, check=True)
        except Exception as error:
            raise RuntimeError(
                f"Khong tim thay {binary_name}. Hay cai ffmpeg/ffprobe truoc khi chay module nay."
            ) from error
