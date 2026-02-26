from __future__ import annotations

from pathlib import Path

from binu_domain.scene_types import SceneSegment


def split_from_video(video_path: Path, fps: float, total_frames: int, camera_mavg_window: int) -> tuple[list[SceneSegment], bool]:
    from scene_split_fusion import detect_scenes

    scenes, _ = detect_scenes(
        str(video_path),
        camera_mavg_window=int(camera_mavg_window),
    )

    used_single_scene_fallback = False
    if not scenes:
        used_single_scene_fallback = True
        duration = float(total_frames) / float(fps) if total_frames > 0 else 0.0
        scenes = [(0.0, max(duration, 0.1))]

    segments = [
        SceneSegment(start_sec=float(start), end_sec=float(end), signals_used=["transnet", "audio", "camera"]) for start, end in scenes
    ]
    return segments, used_single_scene_fallback
