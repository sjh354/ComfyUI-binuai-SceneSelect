from __future__ import annotations

from pathlib import Path

import torch

from binu_domain.kling_analysis.pipeline import analyze_scene_chunk, compact_score_meta, normalize_choice
from binu_domain.scene_split.splitter import split_from_video
from binu_domain.scene_types import AnalysisSummary, SelectedScene, to_analysis_json
from binu_infra.ffmpeg_io import read_audio_segment
from binu_infra.video_io import get_video_meta, read_scene_tensor


def select_best_scene_from_video(
    *,
    node_name: str,
    video_path: Path,
    depth_model: str,
    depth_device: str,
    yolo_model: str,
    yolo_conf: float,
    yolo_imgsz: int,
    sample_fps: float,
    max_frames: int,
    max_decode_frames_per_scene: int,
    camera_mavg_window: int,
    invert_depth: bool,
    local_files_only: bool,
    cache_dir: str,
    depth_model_presets: list[str],
    yolo_model_presets: list[str],
    weights,
    occlusion_percentile: float,
    sam2_model=None,
    sam2_prompt: str = "floor, ground, road",
    sam2_min_mask_ratio: float = 0.01,
    sam2_max_mask_ratio: float = 0.90,
):
    from ultralytics import YOLO
    from video_analysis_utils import DepthEstimator

    fps, total_frames = get_video_meta(video_path)
    fps_estimate = int(round(fps))

    scenes, used_single_scene_fallback = split_from_video(
        video_path=video_path,
        fps=fps,
        total_frames=total_frames,
        camera_mavg_window=int(camera_mavg_window),
    )

    selected_depth_model = normalize_choice(
        value=depth_model,
        choices=depth_model_presets,
        fallback="LiheYoung/depth-anything-small-hf",
    )
    selected_yolo_model = normalize_choice(
        value=yolo_model,
        choices=yolo_model_presets,
        fallback="yolov8m-seg.pt",
    )

    depth_estimator = DepthEstimator(
        model_name=selected_depth_model,
        device=str(depth_device),
        local_files_only=bool(local_files_only),
        cache_dir=str(cache_dir).strip() if str(cache_dir).strip() else None,
    )
    yolo = YOLO(selected_yolo_model)

    best_score = -1.0
    best_images = None
    best_start_sec = 0.0
    best_end_sec = 0.0
    best_meta = {}

    for segment in scenes:
        scene_images = read_scene_tensor(
            video_path=video_path,
            start_sec=float(segment.start_sec),
            end_sec=float(segment.end_sec),
            fps=fps,
            total_frames=total_frames,
            max_decode_frames_per_scene=int(max_decode_frames_per_scene),
        )
        score_meta = analyze_scene_chunk(
            scene_images=scene_images,
            fps=fps,
            depth_estimator=depth_estimator,
            yolo_model=yolo,
            sample_fps=float(sample_fps),
            max_frames=int(max_frames),
            invert_depth=bool(invert_depth),
            yolo_conf=float(yolo_conf),
            yolo_imgsz=int(yolo_imgsz),
            weights=weights,
            occlusion_percentile=float(occlusion_percentile),
            sam2_model=sam2_model,
            sam2_prompt=str(sam2_prompt),
            sam2_min_mask_ratio=float(sam2_min_mask_ratio),
            sam2_max_mask_ratio=float(sam2_max_mask_ratio),
        )
        score = float(score_meta.get("score", 0.0))
        if score > best_score:
            best_score = float(score)
            best_images = scene_images
            best_start_sec = float(segment.start_sec)
            best_end_sec = float(segment.end_sec)
            best_meta = dict(score_meta)

    if best_images is None:
        best_images = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        best_start_sec = 0.0
        best_end_sec = max(0.1, float(total_frames) / float(fps) if total_frames > 0 else 0.1)

    out_audio = read_audio_segment(
        video_path=video_path,
        start_sec=best_start_sec,
        end_sec=best_end_sec,
        sample_rate=44100,
    )

    selected_scene = SelectedScene(
        start_sec=float(best_start_sec),
        end_sec=float(best_end_sec),
        duration_sec=float(max(0.0, best_end_sec - best_start_sec)),
        selected_frame_count=int(best_images.shape[0]) if isinstance(best_images, torch.Tensor) else 0,
        kling_stability=float(best_score) if best_score >= 0 else 0.0,
    )

    summary = AnalysisSummary(
        node=str(node_name),
        scene_detection={
            "scene_count": int(len(scenes)),
            "used_single_scene_fallback": bool(used_single_scene_fallback),
        },
        selected_scene=selected_scene.__dict__,
        analysis=compact_score_meta(best_meta),
    )
    analysis_json = to_analysis_json(summary)
    return best_images, out_audio, fps_estimate, analysis_json
