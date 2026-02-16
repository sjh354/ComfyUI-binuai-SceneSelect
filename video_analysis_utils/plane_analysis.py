from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-8
RNG = np.random.default_rng(42)


def make_point_cloud(depth: np.ndarray, stride: int = 4, max_points: int = 6000) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    z = depth[gy, gx].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.01)
    gy = gy[valid]
    gx = gx[valid]
    z = z[valid]

    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    fx = float(max(h, w))
    fy = fx
    cx = w * 0.5
    cy = h * 0.5

    x = (gx.astype(np.float32) - cx) * z / fx
    y = (gy.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    uv = np.stack([gx.astype(np.float32), gy.astype(np.float32)], axis=1)

    if len(points) > max_points:
        idx = RNG.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        uv = uv[idx]
    return points, uv


def fit_plane_from_points(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + EPS)
    d = -float(np.dot(normal, centroid))
    return normal.astype(np.float32), d


def plane_distances(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.abs(points @ normal + d)


def extract_planes_from_depth(
    depth: np.ndarray,
    min_inlier_ratio: float = 0.05,
    max_planes: int = 4,
    ransac_iters: int = 100,
) -> Tuple[List[Dict[str, float]], float]:
    h, w = depth.shape
    points, uv = make_point_cloud(depth, stride=4, max_points=6000)
    if len(points) < 200:
        return [], 0.02

    residual_threshold = float(np.clip(0.007 + 0.03 * np.std(depth), 0.008, 0.03))
    remaining = np.arange(len(points))
    planes: List[Dict[str, float]] = []

    for _ in range(max_planes):
        if len(remaining) < 120:
            break

        cur_pts = points[remaining]
        best_inliers_local = None
        best_count = 0

        for _ in range(ransac_iters):
            if len(cur_pts) < 3:
                break
            pick = RNG.choice(len(cur_pts), size=3, replace=False)
            p3 = cur_pts[pick]
            v1 = p3[1] - p3[0]
            v2 = p3[2] - p3[0]
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-6:
                continue
            n = n / norm_n
            d = -float(np.dot(n, p3[0]))
            dist = plane_distances(cur_pts, n, d)
            inliers_local = np.where(dist < residual_threshold)[0]
            if len(inliers_local) > best_count:
                best_count = len(inliers_local)
                best_inliers_local = inliers_local

        if best_inliers_local is None:
            break

        inlier_pts = cur_pts[best_inliers_local]
        inlier_ratio = float(len(inlier_pts) / (len(cur_pts) + EPS))
        if inlier_ratio < min_inlier_ratio:
            break

        normal, d = fit_plane_from_points(inlier_pts)
        residual_error = float(np.mean(plane_distances(inlier_pts, normal, d)))

        inlier_global_idx = remaining[best_inliers_local]
        inlier_uv = uv[inlier_global_idx]

        min_u, min_v = np.min(inlier_uv, axis=0)
        max_u, max_v = np.max(inlier_uv, axis=0)
        bbox_area_ratio = float(((max_u - min_u + 1.0) * (max_v - min_v + 1.0)) / (w * h + EPS))

        if normal[2] < 0:
            normal = -normal
            d = -d

        if bbox_area_ratio > 0.03 and residual_error < residual_threshold:
            planes.append(
                {
                    "normal_x": float(normal[0]),
                    "normal_y": float(normal[1]),
                    "normal_z": float(normal[2]),
                    "d": float(d),
                    "inlier_ratio": float(len(inlier_global_idx) / (len(points) + EPS)),
                    "residual_error": residual_error,
                }
            )

        keep_mask = np.ones(len(remaining), dtype=bool)
        keep_mask[best_inliers_local] = False
        remaining = remaining[keep_mask]

    return planes, residual_threshold


def normal_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + EPS)))


def update_plane_tracks(tracks: List[Dict[str, object]], frame_planes: List[Dict[str, float]], frame_index: int) -> None:
    for p in frame_planes:
        n = np.asarray([p["normal_x"], p["normal_y"], p["normal_z"]], dtype=np.float32)
        d = float(p["d"])

        best_idx = -1
        best_score = -1.0
        for i, t in enumerate(tracks):
            last_n = t["normals"][-1]
            last_d = t["d_vals"][-1]
            cos_sim = normal_cos(last_n, n)
            d_diff = abs(last_d - d)
            gap = frame_index - t["frames"][-1]
            if cos_sim > 0.95 and d_diff < 0.08 and gap <= 3:
                score = cos_sim - 0.5 * d_diff
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_idx >= 0:
            t = tracks[best_idx]
            t["normals"].append(n)
            t["d_vals"].append(d)
            t["inlier_ratios"].append(float(p["inlier_ratio"]))
            t["residual_errors"].append(float(p["residual_error"]))
            t["frames"].append(frame_index)
        else:
            tracks.append(
                {
                    "normals": [n],
                    "d_vals": [d],
                    "inlier_ratios": [float(p["inlier_ratio"])],
                    "residual_errors": [float(p["residual_error"])],
                    "frames": [frame_index],
                }
            )


def compute_plane_metrics(
    tracks: List[Dict[str, object]],
    residual_threshold: float,
    min_temporal_stability: float = 0.7,
) -> Tuple[int, float]:
    valid_tracks = []

    for t in tracks:
        if len(t["frames"]) < 2:
            continue
        normals = t["normals"]
        cos_sims = [normal_cos(normals[i - 1], normals[i]) for i in range(1, len(normals))]
        temporal_stability = float(np.mean(cos_sims)) if cos_sims else 0.0

        inlier_ratio = float(np.mean(t["inlier_ratios"]))
        residual_error = float(np.mean(t["residual_errors"]))

        if inlier_ratio > 0.05 and residual_error < residual_threshold and temporal_stability > min_temporal_stability:
            valid_tracks.append(
                {
                    "inlier_ratio": inlier_ratio,
                    "temporal_stability": temporal_stability,
                }
            )

    if not valid_tracks:
        return 0, 0.0

    dominant = max(valid_tracks, key=lambda x: x["inlier_ratio"])
    plane_score = float(np.clip(dominant["inlier_ratio"] * dominant["temporal_stability"], 0.0, 1.0))
    return len(valid_tracks), plane_score
