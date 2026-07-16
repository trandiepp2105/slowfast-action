import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class FFmpegVideoReader:
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.video_width, self.video_height, self.video_fps, self.video_duration = self._probe_video(
            self.video_path
        )

    def _probe_video(self, video_path: str) -> tuple[int, int, float, Optional[float]]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
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
        format_payload = payload.get("format", {})
        duration_text = format_payload.get("duration")
        duration = float(duration_text) if duration_text not in {None, "", "N/A"} else None
        return width, height, fps, duration

    def _normalize_time_range(self, start_sec: float, end_sec: float) -> tuple[float, float]:
        start_sec = max(0.0, float(start_sec))
        end_sec = max(start_sec + 1e-3, float(end_sec))

        if self.video_duration is not None:
            max_end = max(start_sec + 1e-3, self.video_duration)
            end_sec = min(end_sec, max_end)
            start_sec = min(start_sec, max(0.0, end_sec - 1e-3))

        return start_sec, end_sec

    def _decode_raw_clip(
        self,
        start_sec: float,
        end_sec: float,
        *,
        accurate_seek: bool,
        lenient_decode: bool,
    ) -> bytes:
        start_sec, end_sec = self._normalize_time_range(start_sec, end_sec)
        duration_sec = end_sec - start_sec

        command = ["ffmpeg", "-v", "error"]
        if lenient_decode:
            command.extend(["-fflags", "+discardcorrupt+genpts", "-err_detect", "ignore_err"])

        if accurate_seek:
            command.extend(["-i", self.video_path, "-ss", f"{start_sec:.6f}", "-t", f"{duration_sec:.6f}"])
        else:
            command.extend(["-ss", f"{start_sec:.6f}", "-t", f"{duration_sec:.6f}", "-i", self.video_path])

        command.extend(
            [
                "-an",
                "-sn",
                "-dn",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        result = subprocess.run(command, capture_output=True, check=True)
        return result.stdout

    def get_clip(self, start_sec: float, end_sec: float) -> torch.Tensor:
        bytes_per_frame = self.video_width * self.video_height * 3
        if bytes_per_frame <= 0:
            raise RuntimeError(f"Kich thuoc frame khong hop le cho video: {self.video_path}")

        start_sec, end_sec = self._normalize_time_range(start_sec, end_sec)
        decode_attempts = [
            (0.0, 0.0, False, False),
            (0.0, 0.0, False, True),
            (0.0, 0.0, True, True),
            (-0.25, 0.00, True, True),
            (0.00, 0.25, True, True),
            (-0.25, 0.25, True, True),
            (-0.50, 0.00, True, True),
            (0.00, 0.50, True, True),
        ]
        raw = b""
        attempt_errors = []

        for start_shift, end_shift, accurate_seek, lenient_decode in decode_attempts:
            shifted_start, shifted_end = self._normalize_time_range(
                start_sec + start_shift,
                end_sec + end_shift,
            )
            try:
                raw = self._decode_raw_clip(
                    shifted_start,
                    shifted_end,
                    accurate_seek=accurate_seek,
                    lenient_decode=lenient_decode,
                )
            except subprocess.CalledProcessError as error:
                stderr_text = error.stderr.decode("utf-8", errors="ignore") if error.stderr else ""
                attempt_errors.append(stderr_text.strip())
                continue

            if len(raw) >= bytes_per_frame:
                break
        else:
            error_suffix = ""
            non_empty_errors = [text for text in attempt_errors if text]
            if non_empty_errors:
                error_suffix = f" | ffmpeg: {non_empty_errors[-1][:240]}"
            raise RuntimeError(
                f"Khong doc duoc clip trong khoang {start_sec:.3f}-{end_sec:.3f}{error_suffix}"
            )

        frame_count = len(raw) // bytes_per_frame
        usable_bytes = frame_count * bytes_per_frame
        frames = np.frombuffer(raw[:usable_bytes], dtype=np.uint8).copy()
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
