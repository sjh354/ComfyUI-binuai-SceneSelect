#!/usr/bin/env python3
"""Estimate camera motion strength (translation/rotation, excluding zoom) from a video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_analysis_utils import (
    compute_motion_strength,
    iter_sampled_frames_with_signals,
    open_video_meta,
)


def analyze_motion(video_path: Path, sample_fps: float, max_frames: int) -> dict:
    cap, meta = open_video_meta(video_path)
    src_fps = meta["source_fps"]
    width = int(meta["width"])
    height = int(meta["height"])

    motion_signals = []

    for _, _, _, _, _, motion_signal in iter_sampled_frames_with_signals(
        cap=cap,
        source_fps=src_fps,
        sample_fps=sample_fps,
        max_frames=max_frames,
        width=width,
        height=height,
    ):
        if motion_signal is not None:
            motion_signals.append(motion_signal)

    cap.release()
    motion_strength = compute_motion_strength(motion_signals)

    return {
        "motion_strength": motion_strength,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate motion strength from a single video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    print(json.dumps(analyze_motion(args.video, args.sample_fps, args.max_frames), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
