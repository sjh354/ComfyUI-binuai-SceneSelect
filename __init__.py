from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
try:
    import folder_paths
except Exception:
    folder_paths = None


def _ensure_local_pkg_on_path() -> Path:
    node_dir = Path(__file__).resolve().parent
    if str(node_dir) not in sys.path:
        sys.path.insert(0, str(node_dir))
    return node_dir


NODE_DIR = _ensure_local_pkg_on_path()


def _parse_first_json_obj(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _extract_frames_batch(video_path: str, max_frames: int) -> tuple[torch.Tensor, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return dummy, 1, int(round(fps))

    stride = 1
    if max_frames > 0 and total > int(max_frames):
        stride = max(1, int(np.ceil(total / float(max_frames))))

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(rgb))
        idx += 1

    cap.release()

    if not frames:
        dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return dummy, 1, int(round(fps))

    batch = torch.stack(frames, dim=0).to(dtype=torch.float32)
    return batch, int(batch.shape[0]), int(round(fps))


def _load_audio_from_video(video_path: str):
    args = ["ffmpeg", "-i", video_path, "-f", "f32le", "-"]
    try:
        res = subprocess.run(args, capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(r", (\d+) Hz, (\w+), ", res.stderr.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Failed to extract audio from selected clip:\n{err}")

    if match:
        sample_rate = int(match.group(1))
        channels_name = match.group(2)
        channels = {"mono": 1, "stereo": 2}.get(channels_name, 2)
    else:
        sample_rate = 44100
        channels = 2

    if audio.numel() == 0:
        waveform = torch.zeros((1, channels, 1), dtype=torch.float32)
    else:
        usable = (audio.numel() // channels) * channels
        audio = audio[:usable]
        waveform = audio.reshape((-1, channels)).transpose(0, 1).unsqueeze(0).contiguous()

    return {"waveform": waveform, "sample_rate": sample_rate}


class SceneSelectorKling:
    @classmethod
    def INPUT_TYPES(cls):
        files = []
        if folder_paths is not None:
            input_dir = folder_paths.get_input_directory()
            for f in os.listdir(input_dir):
                fp = os.path.join(input_dir, f)
                if not os.path.isfile(fp):
                    continue
                parts = f.rsplit(".", 1)
                if len(parts) == 2 and parts[1].lower() in {"webm", "mp4", "mkv", "gif", "mov"}:
                    files.append(f)

        return {
            "required": {
                "video": (sorted(files),) if files else ("STRING", {"default": ""}),
                "workers": ("INT", {"default": 4, "min": 1, "max": 16}),
                "max_output_frames": ("INT", {"default": 96, "min": 1, "max": 600}),
                "depth_model": ("STRING", {"default": "LiheYoung/depth-anything-small-hf"}),
                "depth_device": (["auto", "cpu", "cuda", "mps"],),
                "yolo_model": ("STRING", {"default": "yolov8m-seg.pt"}),
                "sample_fps": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cache_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("IMAGE", "audio", "fps_estimate")
    FUNCTION = "run"
    CATEGORY = "video/scenedetect"

    def run(
        self,
        video,
        workers,
        max_output_frames,
        depth_model,
        depth_device,
        yolo_model,
        sample_fps,
        max_frames,
        invert_depth,
        local_files_only,
        cache_dir,
    ):
        if folder_paths is not None and isinstance(video, str):
            video_path = folder_paths.get_annotated_filepath(video.strip())
        else:
            video_path = str(video).strip()
        if not video_path or not os.path.isfile(video_path):
            raise ValueError(f"video_path not found: {video_path}")

        get_best_scene_path = NODE_DIR / "get_best_scene.py"
        if not get_best_scene_path.is_file():
            raise ValueError(f"get_best_scene.py not found: {get_best_scene_path}")

        with tempfile.TemporaryDirectory(prefix="kling_best_scene_", dir=str(NODE_DIR)) as tmp_dir:
            scene_dir = Path(tmp_dir) / "scene_clips"
            cmd = [
                sys.executable,
                str(get_best_scene_path),
                str(video_path),
                "--workers",
                str(int(workers)),
                "--scene-dir",
                str(scene_dir),
                "--keep-scenes",
                "--depth-model",
                str(depth_model),
                "--depth-device",
                str(depth_device),
                "--yolo-model",
                str(yolo_model),
                "--sample-fps",
                str(float(sample_fps)),
                "--max-frames",
                str(int(max_frames)),
            ]
            if bool(invert_depth):
                cmd.append("--invert-depth")
            if bool(local_files_only):
                cmd.append("--local-files-only")
            if str(cache_dir).strip():
                cmd.extend(["--cache-dir", str(cache_dir).strip()])

            env = dict(os.environ)
            prev_py = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(NODE_DIR) + (os.pathsep + prev_py if prev_py else "")

            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(NODE_DIR), env=env)
            selection = _parse_first_json_obj(proc.stdout)
            if not selection:
                raise RuntimeError("Failed to parse get_best_scene output JSON.")

            best = selection.get("best_scene", {}) or {}
            if not isinstance(best, dict) or not best:
                dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                out_audio = {"waveform": torch.zeros((1, 2, 1), dtype=torch.float32), "sample_rate": 44100}
                return (dummy, out_audio, 24)

            clip_path = str(best.get("clip_path", "") or "")
            if not clip_path or not os.path.isfile(clip_path):
                raise RuntimeError(f"Best scene clip not found: {clip_path}")

            out_images, _, fps_estimate = _extract_frames_batch(clip_path, max_frames=int(max_output_frames))
            out_audio = _load_audio_from_video(clip_path)
            return (out_images, out_audio, int(fps_estimate))


NODE_CLASS_MAPPINGS = {
    "SceneSelectorKling": SceneSelectorKling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneSelectorKling": "Scene Selector (Kling Best Scene)",
}
