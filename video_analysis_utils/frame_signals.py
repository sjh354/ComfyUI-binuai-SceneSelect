from __future__ import annotations

from typing import Dict, Generator, Optional, Tuple

import cv2
import numpy as np

from .motion_zoom import estimate_pair_transform, pair_motion_signal
from .video_io import sampled_frames


def iter_sampled_frames_with_signals(
    cap: cv2.VideoCapture,
    source_fps: float,
    sample_fps: float,
    max_frames: int,
    width: int,
    height: int,
) -> Generator[Tuple[int, float, np.ndarray, np.ndarray, Optional[Dict[str, float]], Optional[float]], None, None]:
    """Yield sampled frames with grayscale + pairwise transform-derived motion signal.

    motion_signal is None for the first sampled frame or when transform estimation fails.
    """
    prev_gray: Optional[np.ndarray] = None

    for idx, time_sec, frame in sampled_frames(cap, source_fps, sample_fps, max_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        params: Optional[Dict[str, float]] = None
        motion_signal: Optional[float] = None
        if prev_gray is not None:
            params = estimate_pair_transform(prev_gray, gray)
            if params is not None:
                motion_signal = pair_motion_signal(params, width=width, height=height)

        prev_gray = gray
        yield idx, time_sec, frame, gray, params, motion_signal
