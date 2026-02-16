#!/usr/bin/env python3
"""Estimate camera zoom strength from a video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_analysis_utils import compute_zoom_metrics, iter_sampled_frames_with_signals, open_video_meta


def analyze_zoom(video_path: Path, sample_fps: float, max_frames: int) -> dict:
    cap, meta = open_video_meta(video_path)
    src_fps = meta["source_fps"]
    width = int(meta["width"])
    height = int(meta["height"])

    zoom_logs = []

    for _, _, _, _, params, _ in iter_sampled_frames_with_signals(
        cap=cap,
        source_fps=src_fps,
        sample_fps=sample_fps,
        max_frames=max_frames,
        width=width,
        height=height,
    ):
        if params is not None:
            zoom_logs.append(params["log_scale"])

    cap.release()
    zoom_detected, zoom_strength = compute_zoom_metrics(zoom_logs)

    return {
        "zoom_detected": zoom_detected,
        "zoom_strength": zoom_strength,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate zoom strength from a single video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    print(json.dumps(analyze_zoom(args.video, args.sample_fps, args.max_frames), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
