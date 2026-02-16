#!/usr/bin/env python3
"""Single-pass Kling position stability analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_analysis_utils import DepthEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Kling position stability from a single video")
    parser.add_argument("video", type=Path, help="Input video path")

    parser.add_argument("--depth-model", type=str, default="LiheYoung/depth-anything-small-hf")
    parser.add_argument("--depth-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument("--yolo-model", type=str, default="yolov8m-seg.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-imgsz", type=int, default=640)

    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--invert-depth", action="store_true")
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--window-name", type=str, default="Kling Stability Analysis")

    parser.add_argument("--w1-plane", type=float, default=0.3)
    parser.add_argument("--w2-camera-motion", type=float, default=0.2)
    parser.add_argument("--w3-occlusion", type=float, default=0.2)
    parser.add_argument("--w4-flatness", type=float, default=0.1)
    parser.add_argument("--w5-lighting", type=float, default=0.1)
    parser.add_argument("--w6-texture", type=float, default=0.1)

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    from ultralytics import YOLO
    from video_analysis_utils.kling_pipeline import KlingWeights, analyze_kling_stability

    depth_estimator = DepthEstimator(
        model_name=args.depth_model,
        device=args.depth_device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )
    yolo_model = YOLO(args.yolo_model)

    weights = KlingWeights(
        w1_plane=args.w1_plane,
        w2_camera_motion=args.w2_camera_motion,
        w3_occlusion=args.w3_occlusion,
        w4_flatness=args.w4_flatness,
        w5_lighting=args.w5_lighting,
        w6_texture=args.w6_texture,
    )

    result = analyze_kling_stability(
        video_path=args.video,
        depth_estimator=depth_estimator,
        yolo_model=yolo_model,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        invert_depth=args.invert_depth,
        weights=weights,
        yolo_conf=args.yolo_conf,
        yolo_imgsz=args.yolo_imgsz,
        show_window=args.show_window,
        window_name=args.window_name,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
