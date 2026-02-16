from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
import torch


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
        if x.ndim == 3:
            if x.shape[2] >= x.shape[1]:
                y = x[:, :, start_idx:end_idx]
            else:
                y = x[:, start_idx:end_idx, :]
        elif x.ndim == 2:
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


NODE_CLASS_MAPPINGS = {
    "SceneSelectorKling": SceneSelectorKling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneSelectorKling": "Scene Selector (Kling Best Scene)",
}
