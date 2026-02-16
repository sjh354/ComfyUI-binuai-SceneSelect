from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, Tuple

import cv2
import numpy as np


def open_video_meta(video_path: Path) -> Tuple[cv2.VideoCapture, Dict[str, float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(frame_count / src_fps) if frame_count > 0 else 0.0

    return cap, {
        "source_fps": float(src_fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def sampled_frames(
    cap: cv2.VideoCapture,
    source_fps: float,
    sample_fps: float,
    max_frames: int,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    step = max(1, int(round(source_fps / max(sample_fps, 0.1))))

    idx = 0
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % step == 0:
            yield idx, float(idx / source_fps), frame
            sampled += 1
            if 0 < max_frames <= sampled:
                break

        idx += 1
