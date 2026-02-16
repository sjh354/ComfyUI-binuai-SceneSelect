from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .motion_zoom import compute_motion_strength


@dataclass
class FrameContext:
    frame_bgr: np.ndarray
    gray: np.ndarray
    depth: np.ndarray
    floor_mask: np.ndarray
    dynamic_mask: np.ndarray
    motion_signal: Optional[float]


class BaseMetric:
    def update(self, ctx: FrameContext) -> None:
        raise NotImplementedError

    def finalize(self) -> Dict[str, float]:
        raise NotImplementedError


class CameraMotionMetric(BaseMetric):
    def __init__(self) -> None:
        self.motion_signals: List[float] = []

    def update(self, ctx: FrameContext) -> None:
        if ctx.motion_signal is not None and np.isfinite(ctx.motion_signal):
            self.motion_signals.append(float(ctx.motion_signal))

    def finalize(self) -> Dict[str, float]:
        motion_strength = compute_motion_strength(self.motion_signals)
        camera_motion_score = float(np.clip(1.0 - motion_strength, 0.0, 1.0))
        return {
            "motion_strength": float(motion_strength),
            "camera_motion_score": camera_motion_score,
        }


class TextureStabilityMetric(BaseMetric):
    def __init__(self) -> None:
        self.high_freq_ratios: List[float] = []

    def update(self, ctx: FrameContext) -> None:
        mask = ctx.floor_mask
        if mask.sum() < 32:
            return

        grad_x = cv2.Sobel(ctx.gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(ctx.gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        vals = grad_mag[mask]
        if vals.size < 32:
            return

        thr = float(np.percentile(vals, 75))
        high_freq_ratio = float(np.mean(vals > thr))
        self.high_freq_ratios.append(high_freq_ratio)

    def finalize(self) -> Dict[str, float]:
        if not self.high_freq_ratios:
            return {
                "high_frequency_ratio": 1.0,
                "texture_score": 0.0,
            }

        ratio = float(np.median(np.asarray(self.high_freq_ratios, dtype=np.float32)))
        texture_score = float(np.clip(1.0 - ratio, 0.0, 1.0))
        return {
            "high_frequency_ratio": ratio,
            "texture_score": texture_score,
        }


class OcclusionRiskMetric(BaseMetric):
    def __init__(self) -> None:
        self.overlap_ratios: List[float] = []

    def update(self, ctx: FrameContext) -> None:
        floor = ctx.floor_mask
        if floor.sum() < 32:
            return

        h, w = floor.shape
        candidate = np.zeros_like(floor, dtype=bool)
        x0, x1 = int(w * 0.35), int(w * 0.65)
        y0, y1 = int(h * 0.60), int(h * 0.92)
        candidate[y0:y1, x0:x1] = True
        candidate &= floor

        denom = int(candidate.sum())
        if denom < 32:
            return

        overlap = int(np.logical_and(candidate, ctx.dynamic_mask).sum())
        self.overlap_ratios.append(float(overlap / max(denom, 1)))

    def finalize(self) -> Dict[str, float]:
        if not self.overlap_ratios:
            return {
                "dynamic_object_overlap_ratio": 1.0,
                "occlusion_score": 0.0,
            }

        overlap_ratio = float(np.percentile(np.asarray(self.overlap_ratios, dtype=np.float32), 85))
        occlusion_score = float(np.clip(1.0 - overlap_ratio, 0.0, 1.0))
        return {
            "dynamic_object_overlap_ratio": overlap_ratio,
            "occlusion_score": occlusion_score,
        }


class DepthVarianceMetric(BaseMetric):
    def __init__(self) -> None:
        self.depth_variances: List[float] = []

    def update(self, ctx: FrameContext) -> None:
        vals = ctx.depth[ctx.floor_mask]
        if vals.size < 32:
            return

        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        iqr = max(q3 - q1, 1e-6)
        self.depth_variances.append(iqr)

    def finalize(self) -> Dict[str, float]:
        if not self.depth_variances:
            return {
                "normalized_depth_variance": 1.0,
                "flatness_score": 0.0,
            }

        iqr = float(np.median(np.asarray(self.depth_variances, dtype=np.float32)))
        normalized = float(np.clip(iqr / 0.20, 0.0, 1.0))
        flatness_score = float(np.clip(1.0 - normalized, 0.0, 1.0))
        return {
            "normalized_depth_variance": normalized,
            "flatness_score": flatness_score,
        }


class LightingStabilityMetric(BaseMetric):
    def __init__(self) -> None:
        self.luminance_series: List[float] = []

    def update(self, ctx: FrameContext) -> None:
        vals = ctx.gray[ctx.floor_mask]
        if vals.size < 32:
            return
        self.luminance_series.append(float(np.mean(vals) / 255.0))

    def finalize(self) -> Dict[str, float]:
        if len(self.luminance_series) < 2:
            return {
                "temporal_luminance_variation": 1.0,
                "lighting_stability": 0.0,
            }

        arr = np.asarray(self.luminance_series, dtype=np.float32)
        diffs = np.diff(arr)
        variation = float(np.std(diffs))
        normalized = float(np.clip(variation / 0.08, 0.0, 1.0))
        lighting_stability = float(np.clip(1.0 - normalized, 0.0, 1.0))
        return {
            "temporal_luminance_variation": normalized,
            "lighting_stability": lighting_stability,
        }


def compute_floor_mask(depth: np.ndarray, invert_depth: bool) -> np.ndarray:
    h, w = depth.shape
    if invert_depth:
        d = 1.0 - depth
    else:
        d = depth

    bottom = np.zeros((h, w), dtype=bool)
    bottom[int(h * 0.55) :, :] = True

    d_blur = cv2.GaussianBlur(d.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(d_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    g_vals = grad[bottom]
    if g_vals.size < 32:
        return bottom

    low_grad_thr = float(np.percentile(g_vals, 60))
    depth_bottom_vals = d[bottom]
    depth_thr = float(np.percentile(depth_bottom_vals, 35))

    mask = bottom & (grad <= low_grad_thr) & (d >= depth_thr)

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_u8 > 0
