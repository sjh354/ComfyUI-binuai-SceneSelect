from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def get_video_meta(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return (max(0.1, fps), max(0, total))


def read_scene_tensor(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    total_frames: int,
    max_decode_frames_per_scene: int,
) -> torch.Tensor:
    start_idx = max(0, int(np.floor(float(start_sec) * float(fps))))
    end_exclusive = max(start_idx + 1, int(np.ceil(float(end_sec) * float(fps))))
    if total_frames > 0:
        end_exclusive = min(end_exclusive, total_frames)

    frame_ids = list(range(start_idx, end_exclusive))
    if max_decode_frames_per_scene > 0 and len(frame_ids) > int(max_decode_frames_per_scene):
        step = int(np.ceil(len(frame_ids) / float(max_decode_frames_per_scene)))
        frame_ids = frame_ids[::max(1, step)]
    if not frame_ids:
        frame_ids = [start_idx]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frames = []
    for idx in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(rgb))
    cap.release()

    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.stack(frames, dim=0).to(dtype=torch.float32)
