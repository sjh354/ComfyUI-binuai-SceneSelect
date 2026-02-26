from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import torch


def normalize_choice(value, choices: list[str], fallback: str) -> str:
    s = str(value).strip()
    if s in choices:
        return s
    if fallback in choices:
        return fallback
    return choices[0] if choices else s


def compact_score_meta(score_meta: Mapping | dict | None) -> dict:
    if not isinstance(score_meta, Mapping):
        return {}
    out = {}
    keep_keys = (
        "score",
        "sampled_frames",
        "scene_total_frames",
        "sampling_step",
        "sampled_frame_indices",
        "components",
        "metrics",
        "mask_stats",
        "motion_signal_stats",
        "frame_debug_samples",
    )
    for k in keep_keys:
        if k in score_meta:
            out[k] = score_meta.get(k)
    sam2_stats = score_meta.get("sam2_stats")
    if isinstance(sam2_stats, Mapping):
        out["sam2_stats"] = {
            k: v
            for k, v in sam2_stats.items()
            if k not in {"sam2_prompt", "sam2_min_mask_ratio", "sam2_max_mask_ratio"}
        }
    return out


def _tensor_frame_to_bgr(frame: torch.Tensor) -> np.ndarray:
    arr = frame.detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def _extract_dynamic_mask_local(result, frame_shape: tuple[int, int], dynamic_class_ids: set[int]) -> np.ndarray:
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


def _coerce_mask_to_hw_bool(candidate, h: int, w: int) -> np.ndarray | None:
    if candidate is None:
        return None

    if isinstance(candidate, torch.Tensor):
        arr = candidate.detach().float().cpu().numpy()
    else:
        arr = np.asarray(candidate)

    if arr.size == 0:
        return None

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
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


def _extract_mask_like(obj, h: int, w: int) -> np.ndarray | None:
    direct = _coerce_mask_to_hw_bool(obj, h, w)
    if direct is not None:
        return direct

    if isinstance(obj, Mapping):
        for key in ("mask", "masks", "pred_mask", "pred_masks", "segmentation", "segmentations"):
            if key in obj:
                cur = _coerce_mask_to_hw_bool(obj.get(key), h, w)
                if cur is not None:
                    return cur

    if isinstance(obj, (list, tuple)):
        for item in obj:
            cur = _extract_mask_like(item, h, w)
            if cur is not None:
                return cur
    return None


def _infer_floor_mask_with_sam2(
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
                mask = _extract_mask_like(out, h=h, w=w)
                if mask is not None and int(mask.sum()) > 0:
                    return mask
    return None


def analyze_scene_chunk(
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
    occlusion_percentile: float = 85.0,
    sam2_model=None,
    sam2_prompt: str = "floor",
    sam2_min_mask_ratio: float = 0.01,
    sam2_max_mask_ratio: float = 0.95,
) -> dict:
    from video_analysis_utils.kling_metrics import (
        CameraMotionMetric,
        DepthVarianceMetric,
        FrameContext,
        LightingStabilityMetric,
        OcclusionRiskMetric,
        TextureStabilityMetric,
        compute_floor_mask,
    )
    from video_analysis_utils.geometry_metrics import depth_metrics, derive_hints
    from video_analysis_utils.motion_zoom import estimate_pair_transform, pair_motion_signal
    from video_analysis_utils.plane_analysis import compute_plane_metrics, extract_planes_from_depth, update_plane_tracks

    dynamic_class_ids = {0, 1, 2, 3, 5, 7, 16, 17, 18, 19, 21, 22, 23, 24}
    n = int(scene_images.shape[0])
    if n == 0:
        return {"score": 0.0, "reason": "empty_scene_images"}

    step = max(1, int(round(float(fps) / max(float(sample_fps), 0.1))))
    sample_ids = list(range(0, n, step))
    if max_frames > 0 and len(sample_ids) > int(max_frames):
        stride = max(1, int(np.ceil(len(sample_ids) / float(max_frames))))
        sample_ids = sample_ids[::stride]

    camera_metric = CameraMotionMetric()
    texture_metric = TextureStabilityMetric()
    occlusion_metric = OcclusionRiskMetric(occlusion_percentile=float(occlusion_percentile))
    flatness_metric = DepthVarianceMetric()
    lighting_metric = LightingStabilityMetric()
    plane_tracks = []
    residual_thresholds = []
    floor_ratios = []
    dynamic_ratios = []
    motion_values = []
    depth_geometry_series = []
    sam2_used_count = 0
    sam2_fallback_count = 0
    sam2_empty_or_error_count = 0
    sam2_ratio_reject_count = 0
    frame_debug = []

    prev_gray = None
    sampled = 0
    h = int(scene_images.shape[1])
    w = int(scene_images.shape[2])

    for sid in sample_ids:
        frame_rgb = scene_images[int(sid)]
        frame_bgr = _tensor_frame_to_bgr(frame_rgb)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        motion_signal = None
        if prev_gray is not None:
            params = estimate_pair_transform(prev_gray, gray)
            if params is not None:
                motion_signal = pair_motion_signal(params, width=w, height=h)
                motion_values.append(float(motion_signal))
        prev_gray = gray

        depth = depth_estimator.predict_depth(frame_bgr)
        geo = depth_metrics(depth=depth, near_q=0.15, far_q=0.85, invert_depth=bool(invert_depth))
        depth_geometry_series.append(
            {
                "top_mean": float(geo.get("top_mean", 0.0)),
                "mid_mean": float(geo.get("mid_mean", 0.0)),
                "bottom_mean": float(geo.get("bottom_mean", 0.0)),
                "vertical_gradient": float(geo.get("vertical_gradient", 0.0)),
                "near_ratio": float(geo.get("near_ratio", 0.0)),
                "far_ratio": float(geo.get("far_ratio", 0.0)),
            }
        )
        frame_planes, residual_thr = extract_planes_from_depth(depth)
        update_plane_tracks(plane_tracks, frame_planes, frame_index=int(sid))
        residual_thresholds.append(residual_thr)

        floor_mask = None
        floor_source = "depth_fallback"
        if sam2_model is not None:
            floor_mask = _infer_floor_mask_with_sam2(
                sam2_model=sam2_model,
                frame_bgr=frame_bgr,
                sam2_prompt=str(sam2_prompt),
            )
            if floor_mask is not None:
                ratio = float(floor_mask.mean())
                if ratio < float(sam2_min_mask_ratio) or ratio > float(sam2_max_mask_ratio):
                    floor_mask = None
                    sam2_ratio_reject_count += 1
                else:
                    floor_source = "sam2"
                    sam2_used_count += 1
            else:
                sam2_empty_or_error_count += 1
        if floor_mask is None:
            floor_mask = compute_floor_mask(depth=depth, invert_depth=bool(invert_depth))
            if sam2_model is not None:
                sam2_fallback_count += 1

        yolo_results = yolo_model.predict(frame_bgr, conf=float(yolo_conf), imgsz=int(yolo_imgsz), verbose=False)
        yolo_result = yolo_results[0]
        dynamic_mask = _extract_dynamic_mask_local(yolo_result, frame_shape=(h, w), dynamic_class_ids=dynamic_class_ids)
        floor_ratio = float(floor_mask.mean()) if floor_mask.size else 0.0
        dynamic_ratio = float(dynamic_mask.mean()) if dynamic_mask.size else 0.0
        floor_ratios.append(floor_ratio)
        dynamic_ratios.append(dynamic_ratio)

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
        if len(frame_debug) < 48:
            frame_debug.append(
                {
                    "scene_frame_index": int(sid),
                    "floor_source": floor_source,
                    "floor_ratio": floor_ratio,
                    "dynamic_ratio": dynamic_ratio,
                    "motion_signal": float(motion_signal) if motion_signal is not None else None,
                    "residual_threshold": float(residual_thr),
                }
            )

    if sampled == 0:
        return {"score": 0.0, "reason": "no_sampled_frames"}

    global_residual_thr = float(sum(residual_thresholds) / len(residual_thresholds)) if residual_thresholds else 0.02
    plane_count, plane_score = compute_plane_metrics(plane_tracks, residual_threshold=global_residual_thr)
    camera_out = camera_metric.finalize()
    texture_out = texture_metric.finalize()
    occlusion_out = occlusion_metric.finalize()
    flatness_out = flatness_metric.finalize()
    lighting_out = lighting_metric.finalize()

    comp_plane = float(weights.w1_plane) * float(plane_score)
    comp_camera = float(weights.w2_camera_motion) * float(camera_out["camera_motion_score"])
    comp_occlusion = float(weights.w3_occlusion) * float(occlusion_out["occlusion_score"])
    comp_flatness = float(weights.w4_flatness) * float(flatness_out["flatness_score"])
    comp_lighting = float(weights.w5_lighting) * float(lighting_out["lighting_stability"])
    comp_texture = float(weights.w6_texture) * float(texture_out["texture_score"])
    score = float(np.clip(comp_plane + comp_camera + comp_occlusion + comp_flatness + comp_lighting + comp_texture, 0.0, 1.0))

    if depth_geometry_series:
        top_mean = float(np.mean([x["top_mean"] for x in depth_geometry_series]))
        mid_mean = float(np.mean([x["mid_mean"] for x in depth_geometry_series]))
        bottom_mean = float(np.mean([x["bottom_mean"] for x in depth_geometry_series]))
        vertical_gradient = float(np.mean([x["vertical_gradient"] for x in depth_geometry_series]))
        near_ratio = float(np.mean([x["near_ratio"] for x in depth_geometry_series]))
        far_ratio = float(np.mean([x["far_ratio"] for x in depth_geometry_series]))
        hints = derive_hints(
            {
                "top_mean": {"mean": top_mean},
                "mid_mean": {"mean": mid_mean},
                "bottom_mean": {"mean": bottom_mean},
                "vertical_gradient": {"mean": vertical_gradient},
            }
        )
        camera_height_hint = str(hints.get("camera_height_hint", "eye"))
        pitch_hint = str(hints.get("pitch_hint", "level"))
    else:
        top_mean = 0.0
        mid_mean = 0.0
        bottom_mean = 0.0
        vertical_gradient = 0.0
        near_ratio = 0.0
        far_ratio = 0.0
        camera_height_hint = "eye"
        pitch_hint = "level"

    return {
        "score": score,
        "sampled_frames": int(sampled),
        "scene_total_frames": int(n),
        "sampling_step": int(step),
        "sampled_frame_indices": [int(x) for x in sample_ids],
        "weights": {
            "w1_plane": float(weights.w1_plane),
            "w2_camera_motion": float(weights.w2_camera_motion),
            "w3_occlusion": float(weights.w3_occlusion),
            "w4_flatness": float(weights.w4_flatness),
            "w5_lighting": float(weights.w5_lighting),
            "w6_texture": float(weights.w6_texture),
        },
        "components": {
            "plane_component": comp_plane,
            "camera_motion_component": comp_camera,
            "occlusion_component": comp_occlusion,
            "flatness_component": comp_flatness,
            "lighting_component": comp_lighting,
            "texture_component": comp_texture,
        },
        "metrics": {
            "plane_count": int(plane_count),
            "plane_score": float(plane_score),
            "global_residual_threshold": float(global_residual_thr),
            "camera_motion": camera_out,
            "texture": texture_out,
            "occlusion": occlusion_out,
            "flatness": flatness_out,
            "lighting": lighting_out,
            "top_mean": float(top_mean),
            "mid_mean": float(mid_mean),
            "bottom_mean": float(bottom_mean),
            "vertical_gradient": float(vertical_gradient),
            "near_ratio": float(near_ratio),
            "far_ratio": float(far_ratio),
            "camera_height_hint": str(camera_height_hint),
            "pitch_hint": str(pitch_hint),
        },
        "mask_stats": {
            "floor_ratio_mean": float(np.mean(floor_ratios)) if floor_ratios else 0.0,
            "floor_ratio_min": float(np.min(floor_ratios)) if floor_ratios else 0.0,
            "floor_ratio_max": float(np.max(floor_ratios)) if floor_ratios else 0.0,
            "dynamic_ratio_mean": float(np.mean(dynamic_ratios)) if dynamic_ratios else 0.0,
            "dynamic_ratio_min": float(np.min(dynamic_ratios)) if dynamic_ratios else 0.0,
            "dynamic_ratio_max": float(np.max(dynamic_ratios)) if dynamic_ratios else 0.0,
        },
        "sam2_stats": {
            "sam2_enabled": bool(sam2_model is not None),
            "sam2_used_count": int(sam2_used_count),
            "sam2_fallback_count": int(sam2_fallback_count),
            "sam2_empty_or_error_count": int(sam2_empty_or_error_count),
            "sam2_ratio_reject_count": int(sam2_ratio_reject_count),
            "sam2_prompt": str(sam2_prompt),
            "sam2_min_mask_ratio": float(sam2_min_mask_ratio),
            "sam2_max_mask_ratio": float(sam2_max_mask_ratio),
        },
        "motion_signal_stats": {
            "count": int(len(motion_values)),
            "mean": float(np.mean(motion_values)) if motion_values else 0.0,
            "std": float(np.std(motion_values)) if motion_values else 0.0,
            "min": float(np.min(motion_values)) if motion_values else 0.0,
            "max": float(np.max(motion_values)) if motion_values else 0.0,
        },
        "frame_debug_samples": frame_debug,
    }
