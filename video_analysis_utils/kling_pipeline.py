from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

import cv2
import numpy as np

from .depth_model import DepthEstimator
from .frame_signals import iter_sampled_frames_with_signals
from .kling_metrics import (
    CameraMotionMetric,
    DepthVarianceMetric,
    FrameContext,
    LightingStabilityMetric,
    OcclusionRiskMetric,
    TextureStabilityMetric,
    compute_floor_mask,
)
from .plane_analysis import compute_plane_metrics, extract_planes_from_depth, update_plane_tracks
from .video_io import open_video_meta


@dataclass
class KlingWeights:
    w1_plane: float = 0.3
    w2_camera_motion: float = 0.2
    w3_occlusion: float = 0.2
    w4_flatness: float = 0.1
    w5_lighting: float = 0.1
    w6_texture: float = 0.1


def build_plane_masks(depth: np.ndarray, frame_planes: list[dict], residual_threshold: float) -> list[np.ndarray]:
    if not frame_planes:
        return []

    h, w = depth.shape
    z = depth.astype(np.float32)

    ys, xs = np.indices((h, w), dtype=np.float32)
    fx = float(max(h, w))
    fy = fx
    cx = w * 0.5
    cy = h * 0.5

    x = (xs - cx) * z / max(fx, 1e-6)
    y = (ys - cy) * z / max(fy, 1e-6)

    thr = float(max(residual_threshold * 1.5, 0.01))
    masks: list[np.ndarray] = []
    for p in frame_planes:
        n0 = float(p["normal_x"])
        n1 = float(p["normal_y"])
        n2 = float(p["normal_z"])
        d = float(p["d"])
        dist = np.abs(n0 * x + n1 * y + n2 * z + d)
        masks.append(dist < thr)
    return masks


def select_floor_planes(
    frame_planes: list[dict],
    plane_masks: list[np.ndarray],
    floor_candidate_mask: np.ndarray,
    height: int,
    pitch_deg: float,
) -> tuple[list[dict], list[np.ndarray]]:
    if not frame_planes or not plane_masks:
        return [], []

    y_mid = int(height * 0.5)

    pitch_down = max(0.0, float(pitch_deg)) / 20.0
    pitch_up = max(0.0, -float(pitch_deg)) / 20.0

    ny_vs_nz_coeff = float(np.clip(0.35 - 0.10 * pitch_down + 0.25 * pitch_up, 0.20, 0.60))
    min_overlap = float(np.clip(0.35 - 0.10 * pitch_down + 0.25 * pitch_up, 0.25, 0.70))
    min_bottom_ratio = float(np.clip(0.62 - 0.10 * pitch_down + 0.15 * pitch_up, 0.50, 0.82))
    min_centroid_ratio = float(np.clip(0.56 - 0.08 * pitch_down + 0.10 * pitch_up, 0.42, 0.76))

    candidates: list[dict] = []
    for p, m in zip(frame_planes, plane_masks):
        area = int(m.sum())
        if area < 64:
            continue

        ys, _ = np.where(m)
        if ys.size == 0:
            continue

        centroid_y = float(np.mean(ys))
        bottom_ratio = float(np.mean(ys >= y_mid))
        overlap_ratio = float(np.logical_and(m, floor_candidate_mask).sum() / max(area, 1))

        nx = abs(float(p["normal_x"]))
        ny = abs(float(p["normal_y"]))
        nz = abs(float(p["normal_z"]))
        candidates.append(
            {
                "plane": p,
                "mask": m,
                "area": area,
                "centroid_y": centroid_y,
                "bottom_ratio": bottom_ratio,
                "overlap_ratio": overlap_ratio,
                "nx": nx,
                "ny": ny,
                "nz": nz,
            }
        )

    if not candidates:
        return [], []

    def _filter(th_ny_vs_nz: float, th_overlap: float, th_bottom: float, th_centroid: float) -> tuple[list[dict], list[np.ndarray]]:
        floor_planes: list[dict] = []
        floor_masks: list[np.ndarray] = []
        for c in candidates:
            is_more_horizontal_than_wall = bool(c["ny"] >= c["nx"] and c["ny"] >= c["nz"] * th_ny_vs_nz)
            is_bottom_plane = bool(c["centroid_y"] >= height * th_centroid and c["bottom_ratio"] >= th_bottom)
            is_overlap_floor_roi = bool(c["overlap_ratio"] >= th_overlap)
            if is_more_horizontal_than_wall and is_bottom_plane and is_overlap_floor_roi:
                floor_planes.append(c["plane"])
                floor_masks.append(c["mask"])
        return floor_planes, floor_masks

    # 1) strict pass
    floor_planes, floor_masks = _filter(
        th_ny_vs_nz=ny_vs_nz_coeff,
        th_overlap=min_overlap,
        th_bottom=min_bottom_ratio,
        th_centroid=min_centroid_ratio,
    )
    if floor_planes:
        return floor_planes, floor_masks

    # 2) relaxed pass (avoid total miss)
    floor_planes, floor_masks = _filter(
        th_ny_vs_nz=max(0.15, ny_vs_nz_coeff - 0.12),
        th_overlap=max(0.12, min_overlap - 0.18),
        th_bottom=max(0.42, min_bottom_ratio - 0.20),
        th_centroid=max(0.38, min_centroid_ratio - 0.12),
    )
    if floor_planes:
        return floor_planes, floor_masks

    # 3) hard fallback: choose best single candidate by overlap + bottom + verticality.
    scored = sorted(
        candidates,
        key=lambda c: (
            0.55 * c["overlap_ratio"]
            + 0.25 * c["bottom_ratio"]
            + 0.20 * (c["ny"] / max(c["nx"] + c["nz"], 1e-6))
            + 0.05 * min(1.0, c["area"] / 50000.0)
        ),
        reverse=True,
    )
    best = scored[0]
    return [best["plane"]], [best["mask"]]


def estimate_pitch_from_floor_mask(floor_candidate_mask: np.ndarray, height: int) -> float:
    ys, _ = np.where(floor_candidate_mask)
    if ys.size < 64:
        return 0.0

    top_y = float(np.percentile(ys, 10))
    top_ratio = top_y / max(float(height), 1.0)

    # Higher visible floor (smaller top_ratio) => camera pitched down.
    pitch_deg = (0.62 - top_ratio) * 80.0
    return float(np.clip(pitch_deg, -20.0, 20.0))


def annotate_plane_normals(
    vis: np.ndarray,
    frame_planes: list[dict],
    plane_masks: list[np.ndarray],
    floor_plane_ids: Set[int],
) -> None:
    for i, (p, m) in enumerate(zip(frame_planes, plane_masks)):
        ys, xs = np.where(m)
        if ys.size < 16:
            continue

        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        nx = float(p["normal_x"])
        ny = float(p["normal_y"])
        nz = float(p["normal_z"])

        is_floor = id(p) in floor_plane_ids
        color = (0, 255, 0) if is_floor else (180, 180, 180)
        tag = "F" if is_floor else "P"
        txt = f"{tag}{i} n=({nx:+.2f},{ny:+.2f},{nz:+.2f})"
        cv2.putText(vis, txt, (max(8, cx - 90), max(18, cy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def extract_dynamic_mask(result, frame_shape: tuple[int, int], dynamic_class_ids: Set[int]) -> np.ndarray:
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


def analyze_kling_stability(
    video_path: Path,
    depth_estimator: DepthEstimator,
    yolo_model: Any,
    sample_fps: float,
    max_frames: int,
    invert_depth: bool,
    weights: Optional[KlingWeights] = None,
    yolo_conf: float = 0.35,
    yolo_imgsz: int = 640,
    dynamic_class_ids: Optional[Set[int]] = None,
    show_window: bool = False,
    window_name: str = "Kling Stability Analysis",
) -> Dict[str, object]:
    if weights is None:
        weights = KlingWeights()
    if dynamic_class_ids is None:
        # person, bicycle, car, motorcycle, bus, train, truck, animal-ish common classes
        dynamic_class_ids = {0, 1, 2, 3, 5, 7, 16, 17, 18, 19, 21, 22, 23, 24}

    cap, meta = open_video_meta(video_path)
    src_fps = float(meta["source_fps"])
    width = int(meta["width"])
    height = int(meta["height"])

    plane_tracks = []
    residual_thresholds = []

    camera_metric = CameraMotionMetric()
    texture_metric = TextureStabilityMetric()
    occlusion_metric = OcclusionRiskMetric()
    flatness_metric = DepthVarianceMetric()
    lighting_metric = LightingStabilityMetric()

    sampled_count = 0

    for idx, _, frame, gray, _, motion_signal in iter_sampled_frames_with_signals(
        cap=cap,
        source_fps=src_fps,
        sample_fps=sample_fps,
        max_frames=max_frames,
        width=width,
        height=height,
    ):
        sampled_count += 1

        depth = depth_estimator.predict_depth(frame)
        floor_mask = compute_floor_mask(depth=depth, invert_depth=invert_depth)
        pitch_deg = estimate_pitch_from_floor_mask(floor_candidate_mask=floor_mask, height=height)

        frame_planes, residual_thr = extract_planes_from_depth(depth)
        frame_plane_masks = build_plane_masks(depth=depth, frame_planes=frame_planes, residual_threshold=residual_thr)
        floor_frame_planes, floor_plane_masks = select_floor_planes(
            frame_planes=frame_planes,
            plane_masks=frame_plane_masks,
            floor_candidate_mask=floor_mask,
            height=height,
            pitch_deg=pitch_deg,
        )

        update_plane_tracks(plane_tracks, floor_frame_planes, frame_index=idx)
        residual_thresholds.append(residual_thr)

        yolo_results = yolo_model.predict(frame, conf=yolo_conf, imgsz=yolo_imgsz, verbose=False)
        yolo_result = yolo_results[0]
        dynamic_mask = extract_dynamic_mask(yolo_result, frame_shape=(height, width), dynamic_class_ids=dynamic_class_ids)

        ctx = FrameContext(
            frame_bgr=frame,
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

        if show_window:
            vis = yolo_result.plot()

            if floor_plane_masks:
                plane_overlay = np.zeros_like(vis, dtype=np.uint8)
                for p_mask in floor_plane_masks:
                    plane_overlay[p_mask] = (0, 0, 255)
                vis = cv2.addWeighted(vis, 1.0, plane_overlay, 0.35, 0.0)

            floor_plane_ids = {id(p) for p in floor_frame_planes}
            annotate_plane_normals(
                vis=vis,
                frame_planes=frame_planes,
                plane_masks=frame_plane_masks,
                floor_plane_ids=floor_plane_ids,
            )

            floor_u8 = (floor_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(floor_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

            score_text = (
                f"frame={idx} motion={0.0 if motion_signal is None else float(motion_signal):.3f} "
                f"pitch={pitch_deg:+.1f} floor={int(floor_mask.sum())} dyn={int(dynamic_mask.sum())} "
                f"planes={len(floor_frame_planes)}"
            )
            cv2.putText(vis, score_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 240, 20), 2, cv2.LINE_AA)
            cv2.imshow(window_name, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if sampled_count == 0:
        raise RuntimeError("No sampled frame was analyzed.")

    global_residual_thr = float(sum(residual_thresholds) / len(residual_thresholds)) if residual_thresholds else 0.02
    plane_count, plane_score = compute_plane_metrics(plane_tracks, residual_threshold=global_residual_thr)

    camera_out = camera_metric.finalize()
    texture_out = texture_metric.finalize()
    occlusion_out = occlusion_metric.finalize()
    flatness_out = flatness_metric.finalize()
    lighting_out = lighting_metric.finalize()

    kling_stability = (
        weights.w1_plane * float(plane_score)
        + weights.w2_camera_motion * camera_out["camera_motion_score"]
        + weights.w3_occlusion * occlusion_out["occlusion_score"]
        + weights.w4_flatness * flatness_out["flatness_score"]
        + weights.w5_lighting * lighting_out["lighting_stability"]
        + weights.w6_texture * texture_out["texture_score"]
    )
    kling_stability = float(np.clip(kling_stability, 0.0, 1.0))

    return {
        "plane_count": int(plane_count),
        "plane_score": float(plane_score),
        **camera_out,
        **texture_out,
        **occlusion_out,
        **flatness_out,
        **lighting_out,
        "kling_stability": kling_stability,
        "weights": {
            "w1_plane": weights.w1_plane,
            "w2_camera_motion": weights.w2_camera_motion,
            "w3_occlusion": weights.w3_occlusion,
            "w4_flatness": weights.w4_flatness,
            "w5_lighting": weights.w5_lighting,
            "w6_texture": weights.w6_texture,
        },
        "sampled_frames": sampled_count,
        "sample_fps": sample_fps,
    }
