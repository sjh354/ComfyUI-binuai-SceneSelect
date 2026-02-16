from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .common import safe_float


def region_means(depth: np.ndarray) -> Dict[str, float]:
    h = depth.shape[0]
    top = depth[: max(1, h // 3), :]
    mid = depth[h // 3 : max(h // 3 + 1, (2 * h) // 3), :]
    bottom = depth[max((2 * h) // 3, 0) :, :]
    return {
        "top_mean": safe_float(top.mean()),
        "mid_mean": safe_float(mid.mean()),
        "bottom_mean": safe_float(bottom.mean()),
    }


def depth_metrics(depth: np.ndarray, near_q: float, far_q: float, invert_depth: bool) -> Dict[str, float]:
    d = 1.0 - depth if invert_depth else depth

    m = region_means(d)
    top_mean = m["top_mean"]
    mid_mean = m["mid_mean"]
    bottom_mean = m["bottom_mean"]

    vertical_gradient = safe_float(bottom_mean - top_mean)

    q_near = safe_float(np.quantile(d, near_q))
    q_far = safe_float(np.quantile(d, far_q))
    near_ratio = safe_float((d >= q_far).mean())
    far_ratio = safe_float((d <= q_near).mean())

    center_x = d[:, d.shape[1] // 4 : (3 * d.shape[1]) // 4]
    side_l = d[:, : d.shape[1] // 4]
    side_r = d[:, (3 * d.shape[1]) // 4 :]

    center_mean = safe_float(center_x.mean()) if center_x.size else 0.0
    side_mean = safe_float(np.concatenate([side_l.ravel(), side_r.ravel()]).mean()) if side_l.size and side_r.size else 0.0

    return {
        "top_mean": top_mean,
        "mid_mean": mid_mean,
        "bottom_mean": bottom_mean,
        "vertical_gradient": vertical_gradient,
        "near_ratio": near_ratio,
        "far_ratio": far_ratio,
        "center_focus_depth": safe_float(center_mean - side_mean),
        "depth_spatial_std": safe_float(np.std(d)),
    }


def structure_metrics(frame_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edge_density = safe_float((edges > 0).mean())

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=max(20, int(min(gray.shape[:2]) * 0.06)),
        maxLineGap=max(5, int(min(gray.shape[:2]) * 0.02)),
    )
    line_count = float(len(lines)) if lines is not None else 0.0

    lap_var = safe_float(cv2.Laplacian(gray, cv2.CV_32F).var() / 1000.0)

    left = gray[:, : gray.shape[1] // 2]
    right = gray[:, gray.shape[1] - (gray.shape[1] // 2) :]
    right = cv2.flip(right, 1)
    min_h = min(left.shape[0], right.shape[0])
    min_w = min(left.shape[1], right.shape[1])
    symmetry = 1.0 - (safe_float(np.mean(np.abs(left[:min_h, :min_w] - right[:min_h, :min_w]))) / 255.0)

    return {
        "edge_density": edge_density,
        "line_count": line_count,
        "texture_laplacian_var": lap_var,
        "symmetry_score": safe_float(np.clip(symmetry, 0.0, 1.0)),
    }


def derive_hints(agg: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    vertical_gradient = agg["vertical_gradient"]["mean"]
    bottom_vs_mid = agg["bottom_mean"]["mean"] - agg["mid_mean"]["mean"]
    top_vs_mid = agg["top_mean"]["mean"] - agg["mid_mean"]["mean"]

    if vertical_gradient > 0.08:
        pitch_hint = "up"
    elif vertical_gradient < -0.08:
        pitch_hint = "down"
    else:
        pitch_hint = "level"

    if bottom_vs_mid > 0.10 and vertical_gradient > 0.10:
        camera_height_hint = "low"
    elif top_vs_mid > 0.10 and vertical_gradient < -0.10:
        camera_height_hint = "high"
    else:
        camera_height_hint = "eye"

    return {
        "camera_height_hint": camera_height_hint,
        "pitch_hint": pitch_hint,
    }
