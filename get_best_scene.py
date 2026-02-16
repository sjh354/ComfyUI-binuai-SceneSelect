#!/usr/bin/env python3
"""Split a video into scenes and find the best scene by Kling stability.

Pipeline:
1) Detect scene boundaries with `scene_split_fusion.detect_scenes`.
2) Split input video into scene clips.
3) Run `video_analysis_utils.apps.analyze_kling_stability` on each clip in parallel (4 workers by default).
4) Return kling_stability scores and best scene.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from scene_split_fusion import detect_scenes


@dataclass
class SceneClip:
    scene_index: int
    start_sec: float
    end_sec: float
    path: Path


def _split_scene_clip(video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _build_scene_clips(video_path: Path, scenes: List[Tuple[float, float]], out_dir: Path) -> List[SceneClip]:
    clips: List[SceneClip] = []
    for i, (s, e) in enumerate(scenes):
        clip_path = out_dir / f"scene_{i:04d}.mp4"
        _split_scene_clip(video_path, s, e, clip_path)
        clips.append(SceneClip(scene_index=i, start_sec=float(s), end_sec=float(e), path=clip_path))
    return clips


def _run_kling_on_clip(
    clip: SceneClip,
    depth_model: str,
    depth_device: str,
    yolo_model: str,
    sample_fps: float,
    max_frames: int,
    invert_depth: bool,
    local_files_only: bool,
    cache_dir: Optional[str],
) -> dict:
    this_dir = Path(__file__).resolve().parent
    env = dict(os.environ)
    prev_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(this_dir) + (os.pathsep + prev_py if prev_py else "")

    cmd = [
        sys.executable,
        "-m",
        "video_analysis_utils.apps.analyze_kling_stability",
        str(clip.path),
        "--depth-model",
        depth_model,
        "--depth-device",
        depth_device,
        "--yolo-model",
        yolo_model,
        "--sample-fps",
        str(sample_fps),
        "--max-frames",
        str(max_frames),
    ]

    if invert_depth:
        cmd.append("--invert-depth")
    if local_files_only:
        cmd.append("--local-files-only")
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    data = json.loads(proc.stdout)

    return {
        "scene_index": clip.scene_index,
        "start_sec": clip.start_sec,
        "end_sec": clip.end_sec,
        "clip_path": str(clip.path),
        "kling_stability": float(data["kling_stability"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find best scene by Kling stability")
    parser.add_argument("video", type=Path, help="Input video path")

    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count")
    parser.add_argument("--scene-dir", type=Path, default=Path("__scene_clips"), help="Directory to store split scene clips")
    parser.add_argument("--keep-scenes", action="store_true", help="Keep split scene clips")

    parser.add_argument("--depth-model", type=str, default="LiheYoung/depth-anything-small-hf")
    parser.add_argument("--depth-device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument("--yolo-model", type=str, default="yolov8m-seg.pt")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--invert-depth", action="store_true")

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    scenes, _ = detect_scenes(str(args.video))
    if not scenes:
        raise RuntimeError("No scenes detected.")

    scene_root = args.scene_dir / args.video.stem
    scene_root.mkdir(parents=True, exist_ok=True)

    clips = _build_scene_clips(args.video, scenes, scene_root)

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(
                _run_kling_on_clip,
                clip,
                args.depth_model,
                args.depth_device,
                args.yolo_model,
                args.sample_fps,
                args.max_frames,
                args.invert_depth,
                args.local_files_only,
                args.cache_dir,
            )
            for clip in clips
        ]

        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                errors.append(str(e))

    results.sort(key=lambda x: x["scene_index"])
    kling_stability_list = [r["kling_stability"] for r in results]

    best_scene = max(results, key=lambda x: x["kling_stability"]) if results else None

    out = {
        "video": str(args.video),
        "scene_count": len(clips),
        "scenes": [{"scene_index": c.scene_index, "start_sec": c.start_sec, "end_sec": c.end_sec, "clip_path": str(c.path)} for c in clips],
        "kling_stability_list": kling_stability_list,
        "results": results,
        "best_scene": best_scene,
        "errors": errors,
    }

    print(json.dumps(out, ensure_ascii=True, indent=2))

    if not args.keep_scenes:
        for c in clips:
            try:
                c.path.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            scene_root.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
