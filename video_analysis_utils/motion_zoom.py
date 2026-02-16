from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def estimate_pair_transform(prev_gray: np.ndarray, gray: np.ndarray) -> Optional[Dict[str, float]]:
    pts0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if pts0 is None or len(pts0) < 40:
        return None

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        pts0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts1 is None or st is None:
        return None

    st = st.reshape(-1).astype(bool)
    src = pts0.reshape(-1, 2)[st]
    dst = pts1.reshape(-1, 2)[st]
    if len(src) < 30:
        return None

    m, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if m is None or inliers is None:
        return None

    inlier_count = int(inliers.sum())
    if inlier_count < 24:
        return None

    a = m[:2, :2]
    det = float(np.linalg.det(a))
    if det <= 0.0:
        return None

    scale = float(np.sqrt(det))
    if not np.isfinite(scale) or scale < 0.7 or scale > 1.4:
        return None

    r = a / scale
    theta = float(np.arctan2(r[1, 0], r[0, 0]))
    tx = float(m[0, 2])
    ty = float(m[1, 2])

    return {
        "log_scale": float(np.log(scale)),
        "tx": tx,
        "ty": ty,
        "theta": theta,
        "inlier_count": float(inlier_count),
        "inlier_ratio": float(inlier_count / max(len(src), 1)),
    }


def compute_zoom_metrics(zoom_logs: List[float]) -> Tuple[bool, float]:
    if not zoom_logs:
        return False, 0.0

    arr = np.asarray(zoom_logs, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr = arr[np.abs(arr) < 0.2]
    if len(arr) < 2:
        return False, 0.0

    tiny_thr = 0.0025
    arr = np.where(np.abs(arr) >= tiny_thr, arr, 0.0)

    segments: List[np.ndarray] = []
    cur: List[float] = []
    cur_sign = 0
    for v in arr:
        if v == 0.0:
            if cur:
                segments.append(np.asarray(cur, dtype=np.float32))
                cur = []
                cur_sign = 0
            continue
        s = 1 if v > 0 else -1
        if cur_sign == 0 or s == cur_sign:
            cur.append(float(v))
            cur_sign = s
        else:
            segments.append(np.asarray(cur, dtype=np.float32))
            cur = [float(v)]
            cur_sign = s
    if cur:
        segments.append(np.asarray(cur, dtype=np.float32))

    if not segments:
        return False, 0.0

    seg_strengths = [float(abs(np.sum(seg))) for seg in segments]
    seg_lengths = [int(len(seg)) for seg in segments]
    max_idx = int(np.argmax(seg_strengths))

    max_log_strength = seg_strengths[max_idx]
    max_seg_len = seg_lengths[max_idx]
    zoom_detected = bool(max_log_strength > 0.03 and max_seg_len >= 2)

    zoom_strength = float(np.expm1(max_log_strength))
    zoom_strength = float(np.clip(zoom_strength, 0.0, 1.0))
    return zoom_detected, zoom_strength


def compute_motion_strength(motion_signals: List[float]) -> float:
    if not motion_signals:
        return 0.0

    arr = np.asarray(motion_signals, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0

    arr = np.clip(arr, 0.0, 1.5)
    p90 = float(np.percentile(arr, 90))
    p60 = float(np.percentile(arr, 60))

    # Favor sustained stronger motion, not isolated spikes.
    strength = 0.65 * p90 + 0.35 * p60
    return float(np.clip(strength, 0.0, 1.0))


def pair_motion_signal(params: Dict[str, float], width: int, height: int) -> float:
    diag = float(np.hypot(width, height))
    t_norm = float(np.hypot(params["tx"], params["ty"]) / max(diag, 1e-6))
    r_norm = float(min(abs(params["theta"]) / 0.20, 1.0))

    # Translation + rotation, intentionally excluding scale (zoom) term.
    signal = 2.0 * t_norm + 0.5 * r_norm
    return float(np.clip(signal, 0.0, 1.5))
