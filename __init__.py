from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import folder_paths
except Exception:
    folder_paths = None


def _ensure_local_pkg_on_path() -> None:
    node_dir = Path(__file__).resolve().parent
    if str(node_dir) not in sys.path:
        sys.path.insert(0, str(node_dir))


_ensure_local_pkg_on_path()


class SceneSelectorKling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO", {"forceInput": True}),
                "video_info": ("VHS_VIDEOINFO", {"forceInput": True}),
                "cut_threshold": ("FLOAT", {"default": 0.12, "min": 0.001, "max": 1.0, "step": 0.001}),
                "min_scene_len_frames": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "depth_model": ("STRING", {"default": "LiheYoung/depth-anything-small-hf"}),
                "depth_device": (["auto", "cpu", "cuda", "mps"],),
                "yolo_model": ("STRING", {"default": "yolov8m-seg.pt"}),
                "yolo_conf": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01}),
                "yolo_imgsz": ("INT", {"default": 640, "min": 256, "max": 2048}),
                "sample_fps": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cache_dir": ("STRING", {"default": ""}),
                "w1_plane": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w2_camera_motion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w3_occlusion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w4_flatness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w5_lighting": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w6_texture": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("IMAGE", "audio", "fps_estimate")
    FUNCTION = "run"
    CATEGORY = "video/scenedetect"

    def _detect_scenes_from_images(
        self,
        images: torch.Tensor,
        cut_threshold: float,
        min_scene_len_frames: int,
    ) -> list[tuple[int, int]]:
        total = int(images.shape[0])
        if total <= 1:
            return [(0, total)]

        diffs = torch.mean(torch.abs(images[1:] - images[:-1]), dim=(1, 2, 3))
        cut_positions = [i for i, v in enumerate(diffs.tolist(), start=1) if float(v) >= float(cut_threshold)]

        scenes: list[tuple[int, int]] = []
        start = 0
        for cut in cut_positions:
            if (cut - start) >= int(min_scene_len_frames):
                scenes.append((start, cut))
                start = cut
        if start < total:
            scenes.append((start, total))
        return scenes if scenes else [(0, total)]

    def _tensor_frame_to_bgr(self, frame: torch.Tensor) -> np.ndarray:
        arr = frame.detach().float().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        u8 = (arr * 255.0).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)

    def _slice_audio_by_time(self, audio, start_sec: float, end_sec: float):
        if not isinstance(audio, Mapping):
            print("[KlingSelector AUDIO] pass-through: audio is not Mapping, type=", type(audio))
            return audio
        if "waveform" not in audio or "sample_rate" not in audio:
            try:
                keys = list(audio.keys())
            except Exception:
                keys = []
            print("[KlingSelector AUDIO] pass-through: missing keys. keys=", keys)
            return audio

        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate") or 0)
        if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
            print(
                "[KlingSelector AUDIO] pass-through: invalid waveform/sample_rate. "
                f"waveform_type={type(waveform)} sample_rate={sample_rate}"
            )
            return audio

        start_idx = max(0, int(round(float(start_sec) * float(sample_rate))))
        end_idx = max(start_idx, int(round(float(end_sec) * float(sample_rate))))
        if end_idx <= start_idx:
            end_idx = start_idx + 1

        print(
            "[KlingSelector AUDIO] slicing:",
            f"start_sec={start_sec:.6f}",
            f"end_sec={end_sec:.6f}",
            f"sample_rate={sample_rate}",
            f"start_idx={start_idx}",
            f"end_idx={end_idx}",
            f"waveform_shape={tuple(waveform.shape)}",
        )

        x = waveform
        y = None
        # Try common layouts first, then fallback to "largest dim is time".
        if x.ndim == 3:
            # [B, C, T]
            if x.shape[2] >= x.shape[1]:
                y = x[:, :, start_idx:end_idx]
            # [B, T, C]
            else:
                y = x[:, start_idx:end_idx, :]
        elif x.ndim == 2:
            # [C, T] or [T, C]
            if x.shape[1] >= x.shape[0]:
                y = x[:, start_idx:end_idx]
            else:
                y = x[start_idx:end_idx, :]
        elif x.ndim == 1:
            y = x[start_idx:end_idx]

        if y is None:
            time_dim = int(np.argmax(list(x.shape)))
            sl = [slice(None)] * x.ndim
            sl[time_dim] = slice(start_idx, end_idx)
            y = x[tuple(sl)]

        # Comfy video save path expects C-contiguous numpy conversion.
        # Ensure contiguous tensor layout before returning AUDIO waveform.
        if isinstance(y, torch.Tensor):
            y = y.contiguous()
        out_audio = {"waveform": y, "sample_rate": sample_rate}
        out_audio["waveform"] = y
        return out_audio

    def _extract_dynamic_mask_local(self, result, frame_shape: tuple[int, int], dynamic_class_ids: set[int]) -> np.ndarray:
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=bool)

        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "cls", None) is None:
            return mask

        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        seg_masks = getattr(result, "masks", None)
        if seg_masks is not None and getattr(seg_masks, "data", None) is not None:
            m = seg_masks.data.detach().cpu().numpy()
            for i, cls_id in enumerate(classes):
                if cls_id not in dynamic_class_ids:
                    continue
                if i >= len(m):
                    break
                cur = m[i] > 0.5
                if cur.shape != (h, w):
                    cur = cv2.resize(cur.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                mask |= cur
            return mask

        xyxy = boxes.xyxy.detach().cpu().numpy()
        for i, cls_id in enumerate(classes):
            if cls_id not in dynamic_class_ids:
                continue
            x0, y0, x1, y1 = xyxy[i]
            ix0 = max(0, min(w, int(round(x0))))
            iy0 = max(0, min(h, int(round(y0))))
            ix1 = max(0, min(w, int(round(x1))))
            iy1 = max(0, min(h, int(round(y1))))
            if ix1 > ix0 and iy1 > iy0:
                mask[iy0:iy1, ix0:ix1] = True
        return mask

    def _coerce_mask_to_hw_bool(self, candidate, h: int, w: int) -> np.ndarray | None:
        if candidate is None:
            return None

        if isinstance(candidate, torch.Tensor):
            arr = candidate.detach().float().cpu().numpy()
        else:
            arr = np.asarray(candidate)

        if arr.size == 0:
            return None

        if arr.ndim == 4:
            # [B, N, H, W] or [B, H, W, C]
            arr = arr[0]
        if arr.ndim == 3:
            # [N, H, W] => union, [H, W, C] => first channel
            if arr.shape[0] <= 8 and arr.shape[1] >= 16 and arr.shape[2] >= 16:
                arr = np.any(arr > 0.5, axis=0)
            elif arr.shape[-1] <= 4:
                arr = arr[..., 0]
            else:
                arr = arr[0]

        if arr.ndim != 2:
            return None

        if arr.shape != (h, w):
            arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        if arr.dtype != np.bool_:
            arr = arr > 0.5
        return arr.astype(bool)

    def _extract_mask_like(self, obj, h: int, w: int) -> np.ndarray | None:
        direct = self._coerce_mask_to_hw_bool(obj, h, w)
        if direct is not None:
            return direct

        if isinstance(obj, Mapping):
            for key in ("mask", "masks", "pred_mask", "pred_masks", "segmentation", "segmentations"):
                if key in obj:
                    cur = self._coerce_mask_to_hw_bool(obj.get(key), h, w)
                    if cur is not None:
                        return cur

        if isinstance(obj, (list, tuple)):
            for item in obj:
                cur = self._extract_mask_like(item, h, w)
                if cur is not None:
                    return cur
        return None

    def _infer_floor_mask_with_sam2(
        self,
        sam2_model,
        frame_bgr: np.ndarray,
        sam2_prompt: str,
    ) -> np.ndarray | None:
        if sam2_model is None:
            return None

        h, w = frame_bgr.shape[:2]
        rgb_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = None
        try:
            from PIL import Image

            pil_img = Image.fromarray(rgb_u8)
        except Exception:
            pil_img = None

        call_targets = [sam2_model]
        if isinstance(sam2_model, Mapping):
            for key in ("model", "predictor", "sam_model", "sam2_model", "generator"):
                inner = sam2_model.get(key)
                if inner is not None:
                    call_targets.append(inner)

        call_patterns = [
            lambda fn: fn(image=rgb_u8, text_prompt=sam2_prompt),
            lambda fn: fn(image=rgb_u8, prompt=sam2_prompt),
            lambda fn: fn(image=rgb_u8, text=sam2_prompt),
            lambda fn: fn(rgb_u8, sam2_prompt),
            lambda fn: fn(image=pil_img, text_prompt=sam2_prompt) if pil_img is not None else None,
            lambda fn: fn(image=pil_img, prompt=sam2_prompt) if pil_img is not None else None,
            lambda fn: fn(pil_img, sam2_prompt) if pil_img is not None else None,
        ]

        for target in call_targets:
            methods = []
            if callable(target):
                methods.append(target)
            for method_name in ("predict", "segment", "generate", "infer", "__call__"):
                m = getattr(target, method_name, None)
                if callable(m):
                    methods.append(m)

            seen = set()
            unique_methods = []
            for m in methods:
                key = id(m)
                if key not in seen:
                    unique_methods.append(m)
                    seen.add(key)

            for method in unique_methods:
                for pattern in call_patterns:
                    try:
                        out = pattern(method)
                    except Exception:
                        continue
                    if out is None:
                        continue
                    mask = self._extract_mask_like(out, h=h, w=w)
                    if mask is not None and int(mask.sum()) > 0:
                        return mask
        return None

    def _score_scene_kling(
        self,
        scene_images: torch.Tensor,
        fps: float,
        depth_estimator,
        yolo_model,
        sample_fps: float,
        max_frames: int,
        invert_depth: bool,
        yolo_conf: float,
        yolo_imgsz: int,
        weights,
        sam2_model=None,
        sam2_prompt: str = "floor",
        sam2_min_mask_ratio: float = 0.01,
        sam2_max_mask_ratio: float = 0.95,
    ) -> float:
        from video_analysis_utils.kling_metrics import (
            CameraMotionMetric,
            DepthVarianceMetric,
            FrameContext,
            LightingStabilityMetric,
            OcclusionRiskMetric,
            TextureStabilityMetric,
            compute_floor_mask,
        )
        from video_analysis_utils.motion_zoom import estimate_pair_transform, pair_motion_signal
        from video_analysis_utils.plane_analysis import compute_plane_metrics, extract_planes_from_depth, update_plane_tracks

        dynamic_class_ids = {0, 1, 2, 3, 5, 7, 16, 17, 18, 19, 21, 22, 23, 24}
        n = int(scene_images.shape[0])
        if n == 0:
            return 0.0

        step = max(1, int(round(float(fps) / max(float(sample_fps), 0.1))))
        sample_ids = list(range(0, n, step))
        if max_frames > 0 and len(sample_ids) > int(max_frames):
            stride = max(1, int(np.ceil(len(sample_ids) / float(max_frames))))
            sample_ids = sample_ids[::stride]

        camera_metric = CameraMotionMetric()
        texture_metric = TextureStabilityMetric()
        occlusion_metric = OcclusionRiskMetric()
        flatness_metric = DepthVarianceMetric()
        lighting_metric = LightingStabilityMetric()
        plane_tracks = []
        residual_thresholds = []

        prev_gray = None
        sampled = 0
        h = int(scene_images.shape[1])
        w = int(scene_images.shape[2])

        for sid in sample_ids:
            frame_rgb = scene_images[int(sid)]
            frame_bgr = self._tensor_frame_to_bgr(frame_rgb)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            motion_signal = None
            if prev_gray is not None:
                params = estimate_pair_transform(prev_gray, gray)
                if params is not None:
                    motion_signal = pair_motion_signal(params, width=w, height=h)
            prev_gray = gray

            depth = depth_estimator.predict_depth(frame_bgr)
            frame_planes, residual_thr = extract_planes_from_depth(depth)
            update_plane_tracks(plane_tracks, frame_planes, frame_index=int(sid))
            residual_thresholds.append(residual_thr)

            floor_mask = None
            if sam2_model is not None:
                floor_mask = self._infer_floor_mask_with_sam2(
                    sam2_model=sam2_model,
                    frame_bgr=frame_bgr,
                    sam2_prompt=str(sam2_prompt),
                )
                if floor_mask is not None:
                    ratio = float(floor_mask.mean())
                    if ratio < float(sam2_min_mask_ratio) or ratio > float(sam2_max_mask_ratio):
                        floor_mask = None
            if floor_mask is None:
                floor_mask = compute_floor_mask(depth=depth, invert_depth=bool(invert_depth))
            yolo_results = yolo_model.predict(frame_bgr, conf=float(yolo_conf), imgsz=int(yolo_imgsz), verbose=False)
            yolo_result = yolo_results[0]
            dynamic_mask = self._extract_dynamic_mask_local(yolo_result, frame_shape=(h, w), dynamic_class_ids=dynamic_class_ids)

            ctx = FrameContext(
                frame_bgr=frame_bgr,
                gray=gray,
                depth=depth,
                floor_mask=floor_mask,
                dynamic_mask=dynamic_mask,
                motion_signal=motion_signal,
            )
            camera_metric.update(ctx)
            texture_metric.update(ctx)
            occlusion_metric.update(ctx)
            flatness_metric.update(ctx)
            lighting_metric.update(ctx)
            sampled += 1

        if sampled == 0:
            return 0.0

        global_residual_thr = float(sum(residual_thresholds) / len(residual_thresholds)) if residual_thresholds else 0.02
        _, plane_score = compute_plane_metrics(plane_tracks, residual_threshold=global_residual_thr)
        camera_out = camera_metric.finalize()
        texture_out = texture_metric.finalize()
        occlusion_out = occlusion_metric.finalize()
        flatness_out = flatness_metric.finalize()
        lighting_out = lighting_metric.finalize()

        score = (
            float(weights.w1_plane) * float(plane_score)
            + float(weights.w2_camera_motion) * float(camera_out["camera_motion_score"])
            + float(weights.w3_occlusion) * float(occlusion_out["occlusion_score"])
            + float(weights.w4_flatness) * float(flatness_out["flatness_score"])
            + float(weights.w5_lighting) * float(lighting_out["lighting_stability"])
            + float(weights.w6_texture) * float(texture_out["texture_score"])
        )
        return float(np.clip(score, 0.0, 1.0))

    def run(
        self,
        images,
        audio,
        video_info,
        cut_threshold,
        min_scene_len_frames,
        depth_model,
        depth_device,
        yolo_model,
        yolo_conf,
        yolo_imgsz,
        sample_fps,
        max_frames,
        invert_depth,
        local_files_only,
        cache_dir,
        w1_plane,
        w2_camera_motion,
        w3_occlusion,
        w4_flatness,
        w5_lighting,
        w6_texture,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be IMAGE tensor [B,H,W,C]")

        print("[KlingSelector AUDIO] input audio type:", type(audio))
        if isinstance(audio, Mapping):
            try:
                print("[KlingSelector AUDIO] input audio keys:", list(audio.keys()))
            except Exception:
                print("[KlingSelector AUDIO] input audio keys: <unavailable>")
            wf = audio.get("waveform", None)
            sr = audio.get("sample_rate", None)
            print("[KlingSelector AUDIO] input sample_rate:", sr)
            if isinstance(wf, torch.Tensor):
                print("[KlingSelector AUDIO] input waveform shape:", tuple(wf.shape), "dtype:", wf.dtype)
            else:
                print("[KlingSelector AUDIO] input waveform type:", type(wf))

        from ultralytics import YOLO
        from video_analysis_utils import DepthEstimator
        from video_analysis_utils.kling_pipeline import KlingWeights

        total = int(images.shape[0])
        if total <= 0:
            dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (dummy, audio, 24)

        fps = 24.0
        if isinstance(video_info, dict):
            fps = float(video_info.get("loaded_fps") or video_info.get("source_fps") or 24.0)
        fps = max(0.1, fps)
        fps_estimate = int(round(fps))

        scenes = self._detect_scenes_from_images(
            images=images,
            cut_threshold=float(cut_threshold),
            min_scene_len_frames=int(min_scene_len_frames),
        )

        depth_estimator = DepthEstimator(
            model_name=str(depth_model),
            device=str(depth_device),
            local_files_only=bool(local_files_only),
            cache_dir=str(cache_dir).strip() if str(cache_dir).strip() else None,
        )
        yolo = YOLO(str(yolo_model))
        weights = KlingWeights(
            w1_plane=float(w1_plane),
            w2_camera_motion=float(w2_camera_motion),
            w3_occlusion=float(w3_occlusion),
            w4_flatness=float(w4_flatness),
            w5_lighting=float(w5_lighting),
            w6_texture=float(w6_texture),
        )

        scene_scores = []
        for idx, (start, end) in enumerate(scenes, start=1):
            chunk = images[start:end]
            score = self._score_scene_kling(
                scene_images=chunk,
                fps=fps,
                depth_estimator=depth_estimator,
                yolo_model=yolo,
                sample_fps=float(sample_fps),
                max_frames=int(max_frames),
                invert_depth=bool(invert_depth),
                yolo_conf=float(yolo_conf),
                yolo_imgsz=int(yolo_imgsz),
                weights=weights,
            )
            scene_scores.append(
                {
                    "scene_index": int(idx),
                    "start_frame": int(start),
                    "end_frame_exclusive": int(end),
                    "kling_stability": float(score),
                }
            )

        best = max(scene_scores, key=lambda x: x["kling_stability"]) if scene_scores else None
        if not best:
            out_images = images[:1]
            out_audio = audio
            print("[KlingSelector AUDIO] no best scene found, audio pass-through.")
        else:
            s = int(best["start_frame"])
            e = int(best["end_frame_exclusive"])
            out_images = images[s:e] if e > s else images[:1]
            start_sec = float(s) / float(fps)
            end_sec = float(e) / float(fps)
            print(
                "[KlingSelector AUDIO] best scene:",
                f"index={best['scene_index']}",
                f"start_frame={s}",
                f"end_frame_exclusive={e}",
            )
            out_audio = self._slice_audio_by_time(audio, start_sec=start_sec, end_sec=end_sec)

        return (out_images, out_audio, fps_estimate)


class SceneSelectorUpload(SceneSelectorKling):
    @classmethod
    def _video_candidates(cls) -> list[str]:
        if folder_paths is None:
            return [""]

        # Prefer VHS-style video list if available.
        try:
            files = folder_paths.get_filename_list("video")
            files = [str(f) for f in files if str(f)]
            if files:
                return sorted(files)
        except Exception:
            pass

        try:
            input_dir = Path(folder_paths.get_input_directory())
            exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
            files = [p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            if files:
                return sorted(files)
        except Exception:
            pass
        return [""]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    cls._video_candidates(),
                    {
                        "video_upload": True,
                        "vhs_path_extensions": ["mp4", "mov", "mkv", "webm", "avi", "m4v"],
                    },
                ),
                "depth_model": ("STRING", {"default": "LiheYoung/depth-anything-small-hf"}),
                "depth_device": (["auto", "cpu", "cuda", "mps"],),
                "yolo_model": ("STRING", {"default": "yolov8m-seg.pt"}),
                "yolo_conf": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01}),
                "yolo_imgsz": ("INT", {"default": 640, "min": 256, "max": 2048}),
                "sample_fps": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "max_decode_frames_per_scene": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cache_dir": ("STRING", {"default": ""}),
                "w1_plane": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w2_camera_motion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w3_occlusion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w4_flatness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w5_lighting": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w6_texture": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("IMAGE", "audio", "fps_estimate")
    FUNCTION = "run"
    CATEGORY = "video/scenedetect"

    def _resolve_video_path(self, video: str) -> Path:
        p = Path(str(video))
        if p.exists():
            return p.resolve()

        if folder_paths is not None:
            try:
                return Path(folder_paths.get_annotated_filepath(str(video))).resolve()
            except Exception:
                pass
            try:
                return (Path(folder_paths.get_input_directory()) / str(video)).resolve()
            except Exception:
                pass

        return p.resolve()

    def _video_meta(self, video_path: Path) -> tuple[float, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return (max(0.1, fps), max(0, total))

    def _read_scene_tensor(
        self,
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

    def _read_audio_segment(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        sample_rate: int = 44100,
    ):
        start_sec = max(0.0, float(start_sec))
        end_sec = max(start_sec + 1e-3, float(end_sec))
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-to",
            f"{end_sec:.6f}",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(int(sample_rate)),
            "-f",
            "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        raw = proc.stdout if proc.returncode == 0 else b""
        if not raw:
            waveform = torch.zeros((1, 2, 1), dtype=torch.float32)
            return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}

        audio_np = np.frombuffer(raw, dtype=np.float32)
        if audio_np.size < 2:
            waveform = torch.zeros((1, 2, 1), dtype=torch.float32)
            return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}

        n = (audio_np.size // 2) * 2
        audio_np = audio_np[:n].reshape(-1, 2)  # [T, C]
        waveform = torch.from_numpy(audio_np.T.copy()).unsqueeze(0).to(dtype=torch.float32)  # [1, C, T]
        return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}

    def run(
        self,
        video,
        depth_model,
        depth_device,
        yolo_model,
        yolo_conf,
        yolo_imgsz,
        sample_fps,
        max_frames,
        max_decode_frames_per_scene,
        invert_depth,
        local_files_only,
        cache_dir,
        w1_plane,
        w2_camera_motion,
        w3_occlusion,
        w4_flatness,
        w5_lighting,
        w6_texture,
        video_path="",
    ):
        chosen = str(video_path).strip() if isinstance(video_path, str) else ""
        resolved = self._resolve_video_path(chosen if chosen else str(video))
        if not resolved.exists():
            raise FileNotFoundError(f"video not found: {resolved}")

        from ultralytics import YOLO
        from scene_split_fusion import detect_scenes
        from video_analysis_utils import DepthEstimator
        from video_analysis_utils.kling_pipeline import KlingWeights

        fps, total_frames = self._video_meta(resolved)
        fps_estimate = int(round(fps))

        scenes, _ = detect_scenes(str(resolved))
        if not scenes:
            duration = float(total_frames) / float(fps) if total_frames > 0 else 0.0
            scenes = [(0.0, max(duration, 0.1))]

        depth_estimator = DepthEstimator(
            model_name=str(depth_model),
            device=str(depth_device),
            local_files_only=bool(local_files_only),
            cache_dir=str(cache_dir).strip() if str(cache_dir).strip() else None,
        )
        yolo = YOLO(str(yolo_model))
        weights = KlingWeights(
            w1_plane=float(w1_plane),
            w2_camera_motion=float(w2_camera_motion),
            w3_occlusion=float(w3_occlusion),
            w4_flatness=float(w4_flatness),
            w5_lighting=float(w5_lighting),
            w6_texture=float(w6_texture),
        )

        best_score = -1.0
        best_images = None
        best_start_sec = 0.0
        best_end_sec = 0.0
        for start_sec, end_sec in scenes:
            scene_images = self._read_scene_tensor(
                video_path=resolved,
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                fps=fps,
                total_frames=total_frames,
                max_decode_frames_per_scene=int(max_decode_frames_per_scene),
            )
            score = self._score_scene_kling(
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
            )
            if score > best_score:
                best_score = float(score)
                best_images = scene_images
                best_start_sec = float(start_sec)
                best_end_sec = float(end_sec)

        if best_images is None:
            best_images = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            best_start_sec = 0.0
            best_end_sec = max(0.1, float(total_frames) / float(fps) if total_frames > 0 else 0.1)

        out_audio = self._read_audio_segment(
            video_path=resolved,
            start_sec=best_start_sec,
            end_sec=best_end_sec,
            sample_rate=44100,
        )
        return (best_images, out_audio, fps_estimate)


NODE_CLASS_MAPPINGS = {
    "SceneSelectorKling": SceneSelectorKling,
    "SceneSelectorUpload": SceneSelectorUpload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneSelectorKling": "Scene Selector (Without Upload)",
    "SceneSelectorUpload": "Scene Selector (Upload)",
}
