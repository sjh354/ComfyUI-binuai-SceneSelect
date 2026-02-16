#!/usr/bin/env python3
"""Depth-based geometry analyzer for a single video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from video_analysis_utils import (
    DepthEstimator,
    compute_motion_strength,
    compute_plane_metrics,
    compute_zoom_metrics,
    depth_metrics,
    derive_hints,
    extract_planes_from_depth,
    iter_sampled_frames_with_signals,
    open_video_meta,
    stats,
    structure_metrics,
    update_plane_tracks,
)


def analyze_video(
    video_path: Path,
    estimator: DepthEstimator,
    sample_fps: float,
    max_frames: int,
    near_q: float,
    far_q: float,
    invert_depth: bool,
) -> Dict[str, object]:
    cap, meta = open_video_meta(video_path)
    src_fps = meta["source_fps"]
    width = int(meta["width"])
    height = int(meta["height"])

    per_frame: List[Dict[str, float]] = []
    plane_tracks: List[Dict[str, object]] = []
    residual_thresholds: List[float] = []

    zoom_logs: List[float] = []
    motion_signals: List[float] = []
    for idx, time_sec, frame, _, params, motion_signal in iter_sampled_frames_with_signals(
        cap=cap,
        source_fps=src_fps,
        sample_fps=sample_fps,
        max_frames=max_frames,
        width=width,
        height=height,
    ):
        if params is not None:
            zoom_logs.append(params["log_scale"])
        if motion_signal is not None:
            motion_signals.append(motion_signal)

        depth = estimator.predict_depth(frame)
        d_metrics = depth_metrics(depth, near_q=near_q, far_q=far_q, invert_depth=invert_depth)
        s_metrics = structure_metrics(frame)

        frame_planes, frame_residual_thr = extract_planes_from_depth(depth)
        update_plane_tracks(plane_tracks, frame_planes, frame_index=idx)
        residual_thresholds.append(frame_residual_thr)

        per_frame.append(
            {
                **d_metrics,
                **s_metrics,
                "frame_index": float(idx),
                "time_sec": float(time_sec),
            }
        )

    cap.release()

    if not per_frame:
        raise RuntimeError("No sampled frame was analyzed.")

    metric_keys = [k for k in per_frame[0].keys() if k not in ("frame_index", "time_sec")]
    aggregates = {k: stats([x[k] for x in per_frame]) for k in metric_keys}

    global_residual_thr = float(sum(residual_thresholds) / len(residual_thresholds)) if residual_thresholds else 0.02
    plane_count, plane_score = compute_plane_metrics(plane_tracks, residual_threshold=global_residual_thr)
    zoom_detected, zoom_strength = compute_zoom_metrics(zoom_logs)
    motion_strength = compute_motion_strength(motion_signals)
    hints = derive_hints(aggregates)

    return {
        "top_mean": aggregates["top_mean"]["mean"],
        "mid_mean": aggregates["mid_mean"]["mean"],
        "bot_mean": aggregates["bottom_mean"]["mean"],
        "vertical_gradient": aggregates["vertical_gradient"]["mean"],
        "near_ratio": aggregates["near_ratio"]["mean"],
        "far_ratio": aggregates["far_ratio"]["mean"],
        "camera_height_hint": hints["camera_height_hint"],
        "pitch_hint": hints["pitch_hint"],
        "plane_count": plane_count,
        "plane_score": plane_score,
        "zoom_detected": zoom_detected,
        "zoom_strength": zoom_strength,
        "motion_strength": motion_strength,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth model based video geometry analyzer")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--model", type=str, default="LiheYoung/depth-anything-small-hf", help="Hugging Face model id")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--local-files-only", action="store_true", help="Load model only from local Hugging Face cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=0, help="Max sampled frames (0 means unlimited)")
    parser.add_argument("--near-q", type=float, default=0.15, help="Near quantile threshold")
    parser.add_argument("--far-q", type=float, default=0.85, help="Far quantile threshold")
    parser.add_argument("--invert-depth", action="store_true", help="Invert depth if your model outputs opposite convention")
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not (0.01 <= args.near_q < args.far_q <= 0.99):
        raise ValueError("near-q and far-q must satisfy 0.01 <= near-q < far-q <= 0.99")

    estimator = DepthEstimator(
        model_name=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )

    result = analyze_video(
        video_path=args.video,
        estimator=estimator,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        near_q=args.near_q,
        far_q=args.far_q,
        invert_depth=args.invert_depth,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
