from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def run_camera_jump(
    video_path: str,
    fps: float,
    threshold: float = 0.70,
    max_width: int = 640,
    min_matches: int = 60
) -> List[Boundary]:
    """
    ORB + Homography 기반 카메라 점프 탐지.
    - 정합 실패/인라이어 비율 급감 또는 변환 파라미터 급변을 score로 사용
    """
    from . import Boundary

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        return []

    H0, W0 = prev.shape[:2]
    scale = 1.0
    if W0 > max_width:
        scale = max_width / float(W0)
    prev_small = cv2.resize(prev, (int(W0 * scale), int(H0 * scale))) if scale != 1.0 else prev
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)

    scores = []

    while True:
        ok, cur = cap.read()
        if not ok or cur is None:
            break
        cur_small = cv2.resize(cur, (int(W0 * scale), int(H0 * scale))) if scale != 1.0 else cur
        cur_gray = cv2.cvtColor(cur_small, cv2.COLOR_BGR2GRAY)

        kp2, des2 = orb.detectAndCompute(cur_gray, None)
        score = 0.0

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            score = 1.0
        else:
            matches = bf.match(des1, des2)
            if len(matches) < min_matches:
                score = 1.0
            else:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
                if H is None or mask is None:
                    score = 1.0
                else:
                    inliers = int(mask.ravel().sum())
                    inlier_ratio = float(inliers) / float(len(matches) + 1e-6)

                    sx = float(np.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0]))
                    sy = float(np.sqrt(H[0, 1] * H[0, 1] + H[1, 1] * H[1, 1]))
                    s = max(1e-6, (sx + sy) * 0.5)
                    scale_jump = abs(np.log(s))

                    tx = float(H[0, 2]) / scale
                    ty = float(H[1, 2]) / scale
                    trans_norm = np.hypot(tx, ty) / float(max(W0, H0) + 1e-6)

                    score = (
                        (1.0 - inlier_ratio) * 0.5
                        + min(scale_jump / 0.12, 1.0) * 0.3
                        + min(trans_norm / 0.08, 1.0) * 0.2
                    )

        scores.append(score)

        kp1, des1 = kp2, des2

    cap.release()

    p_raw = np.array(scores, dtype=np.float32)
    if p_raw.size == 0:
        return []

    # Suppress single-frame spikes by requiring neighborhood support
    # around a candidate (or a very strong raw peak).
    p = _moving_average_1d(p_raw, window=5)
    idx = np.where(p >= threshold)[0]
    boundaries: List[Boundary] = []
    if len(idx) == 0:
        return boundaries

    support_radius = 2
    support_min_count = 2
    near_thr = float(threshold) * 0.85
    very_strong_thr = min(1.0, float(threshold) + 0.20)

    valid_idx = []
    for i in idx.tolist():
        a0 = max(0, int(i) - support_radius)
        b0 = min(int(p_raw.size), int(i) + support_radius + 1)
        support_count = int(np.sum(p_raw[a0:b0] >= near_thr))
        if support_count >= support_min_count or float(p_raw[int(i)]) >= very_strong_thr:
            valid_idx.append(int(i))

    if not valid_idx:
        return boundaries

    groups = []
    start = valid_idx[0]
    prev_i = valid_idx[0]
    for k in valid_idx[1:]:
        if k == prev_i + 1:
            prev_i = k
        else:
            groups.append((start, prev_i))
            start = k
            prev_i = k
    groups.append((start, prev_i))

    for a, b in groups:
        peak_rel = int(np.argmax(p[a:b + 1]))
        peak_idx = a + peak_rel
        t = peak_idx / fps
        score = float(p[peak_idx])
        boundaries.append(Boundary(t=t, score=score, source="camera", kind="cut"))
    return boundaries
