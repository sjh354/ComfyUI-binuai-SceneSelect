from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import probe_fps


def run_transnetv2(
    video_path: str,
    weights_dir: str,
    threshold: float = 0.35,
    transition_threshold: float = 0.35,
    transition_min_frames: int = 1
) -> Tuple[List[Boundary], np.ndarray, float]:
    """
    TransNetV2 single-frame head의 per-frame probability를 boundary로 변환.
    반환: boundaries, per_frame_probs, fps
    """
    from . import Boundary

    from .transnetv2 import TransNetV2

    model = TransNetV2(weights_dir)
    frames, single_p, all_p = model.predict_video(video_path)
    fps = probe_fps(video_path)

    p = np.squeeze(single_p).astype(np.float32)
    p_all = np.squeeze(all_p).astype(np.float32)

    idx = np.where(p >= threshold)[0]
    boundaries: List[Boundary] = []
    if len(idx) == 0:
        return boundaries, p, fps

    groups = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            groups.append((start, prev))
            start = k
            prev = k
    groups.append((start, prev))

    idx_all = np.where(p_all >= transition_threshold)[0] if p_all.size else np.array([], dtype=np.int64)
    trans_groups = []
    if len(idx_all) > 0:
        s = idx_all[0]
        prev = idx_all[0]
        for k in idx_all[1:]:
            if k == prev + 1:
                prev = k
            else:
                trans_groups.append((s, prev))
                s = k
                prev = k
        trans_groups.append((s, prev))

    for a, b in groups:
        seg_len = b - a + 1
        peak_rel = int(np.argmax(p[a:b + 1]))
        peak_idx = a + peak_rel
        t = peak_idx / fps
        score = float(p[peak_idx])

        kind = "cut"
        trans_score = 0.0
        for ta, tb in trans_groups:
            if not (b < ta or a > tb):
                trans_score = float(np.max(p_all[ta:tb + 1])) if p_all.size else 0.0
                if (tb - ta + 1) >= int(transition_min_frames):
                    kind = "transition"
                break

        boundaries.append(
            Boundary(
                t=t,
                score=score,
                source="transnet",
                kind=kind,
                seg_len=int(seg_len),
                trans_score=float(trans_score),
            )
        )
    return boundaries, p, fps
