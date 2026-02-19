import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional


@dataclass
class Boundary:
    t: float          # time in seconds
    score: float      # confidence-like score
    source: str       # 'pyscene', 'transnet', 'audio', 'camera'
    kind: str = "cut"  # 'cut' or 'transition'
    seg_len: int = 0
    trans_score: float = 0.0


def _resolve_bin(name: str) -> str:
    # 1) explicit env override
    env_key = name.upper() + "_PATH"
    env_val = os.environ.get(env_key, "").strip()
    if env_val and Path(env_val).exists():
        return env_val

    # 2) PATH lookup
    found = shutil.which(name)
    if found:
        return found

    # 3) common macOS Homebrew/system locations
    candidates = [
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    # keep original behavior (will raise FileNotFoundError at execution)
    return name


def probe_fps(video_path: str) -> float:
    """
    ffprobe로 fps 가져오기. (TransNetV2 frames length -> time 변환에 필요)
    """
    ffprobe_bin = _resolve_bin("ffprobe")
    cmd = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    if "/" in out:
        num, den = out.split("/")
        return float(num) / float(den)
    return float(out)


def run_pyscenedetect(*args, **kwargs):
    from .pyscenedetect import run_pyscenedetect as _impl
    return _impl(*args, **kwargs)


def run_transnetv2(*args, **kwargs):
    from .transnetv2_detector import run_transnetv2 as _impl
    return _impl(*args, **kwargs)


def run_audio_onset(*args, **kwargs):
    from .audio_onset import run_audio_onset as _impl
    return _impl(*args, **kwargs)


def run_camera_jump(*args, **kwargs):
    from .camera_jump import run_camera_jump as _impl
    return _impl(*args, **kwargs)


def probe_duration(video_path: str) -> float:
    ffprobe_bin = _resolve_bin("ffprobe")
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(out)


def _default_transnet_weights_dir() -> str:
    return str(Path(__file__).resolve().parent / "transnetv2" / "transnetv2-weights")


def nms_fuse_boundaries(boundaries: List[Boundary], merge_window_sec: float = 0.25) -> List[Boundary]:
    """
    서로 가까운 boundary를 하나로 merge (NMS-like).
    """
    if not boundaries:
        return []

    boundaries = sorted(boundaries, key=lambda b: b.t)
    fused = [boundaries[0]]

    for b in boundaries[1:]:
        last = fused[-1]
        if abs(b.t - last.t) <= merge_window_sec:
            if b.score > last.score:
                fused[-1] = b
        else:
            fused.append(b)
    return fused


def _group_boundaries_for_nms(boundaries: List[Boundary], merge_window_sec: float) -> List[List[Boundary]]:
    """
    merge_window_sec 내에 있는 boundary들을 하나의 그룹으로 묶는다.
    """
    if not boundaries:
        return []
    sorted_b = sorted(boundaries, key=lambda b: b.t)
    groups: List[List[Boundary]] = [[sorted_b[0]]]
    for b in sorted_b[1:]:
        last = groups[-1][-1]
        if abs(b.t - last.t) <= merge_window_sec:
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


def build_scenes_from_boundaries(boundaries: List[Boundary], video_duration_sec: float) -> List[Tuple[float, float]]:
    """
    boundary 시간들을 scene 구간으로 변환.
    """
    ts = [b.t for b in sorted(boundaries, key=lambda x: x.t)]
    ts = [t for t in ts if 0.0 < t < video_duration_sec]
    ts = sorted(set(ts))

    scenes = []
    prev = 0.0
    for t in ts:
        scenes.append((prev, t))
        prev = t
    scenes.append((prev, video_duration_sec))
    return scenes


def render_scene_numbers(
    video_path: str,
    scenes: List[Tuple[float, float]],
    output_path: str = "output.mp4",
    x: int = 20,
    y: int = 20,
    fontsize: int = 36,
):
    """
    ffmpeg drawtext로 씬 번호를 좌상단에 오버레이.
    """
    if not scenes:
        raise ValueError("No scenes to render.")

    filters = []
    for i, (start, end) in enumerate(scenes, start=1):
        enable_expr = f"between(t\\,{start:.6f}\\,{end:.6f})"
        text = f"Scene {i}"
        filters.append(
            "drawtext="
            f"text='{text}':"
            f"x={x}:y={y}:"
            f"fontsize={fontsize}:"
            "fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8:"
            f"enable='{enable_expr}'"
        )

    filter_complex = ",".join(filters)
    base_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", filter_complex,
    ]
    primary = base_cmd + [
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "copy",
        output_path
    ]
    fallback = base_cmd + [
        "-c:v", "libopenh264", "-b:v", "4M", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]
    try:
        subprocess.run(primary, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(fallback, check=True)


def detect_scenes(
    video_path: str,
    transnet_weights_dir: Optional[str] = None,
    pyscene_threshold: float = 27.0,
    pyscene_min_scene_len_frames: int = 8,
    transnet_threshold: float = 0.35,
    transnet_transition_threshold: float = 0.35,
    transnet_transition_min_frames: int = 1,
    audio_tmp_wav: str = "__tmp_audio.wav",
    audio_sr: int = 22050,
    camera_threshold: float = 0.70,
    camera_max_width: int = 640,
    camera_min_matches: int = 60,
    camera_mavg_window: int = 5,
    merge_window_sec: float = 0.30,
    weights: Optional[Dict[str, float]] = None,
    print_cut_reasons: bool = True,
) -> Tuple[List[Tuple[float, float]], List[Boundary]]:
    if not transnet_weights_dir:
        transnet_weights_dir = _default_transnet_weights_dir()

    duration = probe_duration(video_path)
    fps = probe_fps(video_path)

    try:
        from .pyscenedetect import run_pyscenedetect
        b_pys = run_pyscenedetect(
            video_path,
            threshold=pyscene_threshold,
            min_scene_len_frames=pyscene_min_scene_len_frames
        )
    except Exception:
        b_pys = []

    try:
        from .transnetv2_detector import run_transnetv2
        b_tn, _, _ = run_transnetv2(
            video_path,
            transnet_weights_dir,
            threshold=transnet_threshold,
            transition_threshold=transnet_transition_threshold,
            transition_min_frames=transnet_transition_min_frames
        )
    except Exception:
        b_tn = []

    try:
        from .audio_onset import run_audio_onset
        b_aud = run_audio_onset(video_path, tmp_wav=audio_tmp_wav, sr=audio_sr)
        if os.path.exists(audio_tmp_wav):
            os.remove(audio_tmp_wav)
    except Exception:
        b_aud = []

    try:
        from .camera_jump import run_camera_jump
        b_cam = run_camera_jump(
            video_path,
            fps=fps,
            threshold=camera_threshold,
            max_width=camera_max_width,
            min_matches=camera_min_matches,
            score_smoothing_window=int(camera_mavg_window),
        )
    except Exception:
        b_cam = []

    if weights is None:
        weights = {
            "transnet": 0.55,
            "audio": 0.0,
            "pyscene": 0.05,
            "camera": 0.25,
        }

    all_b: List[Boundary] = []
    all_b.extend(b_tn)
    all_b.extend(b_aud)
    all_b.extend(b_pys)
    all_b.extend(b_cam)

    grouped = _group_boundaries_for_nms(all_b, merge_window_sec=merge_window_sec)
    fused: List[Boundary] = []
    nms_reason_map: Dict[Tuple[float, str, str], str] = {}
    for g in grouped:
        winner = max(g, key=lambda x: x.score)
        fused.append(winner)
        key = (winner.t, winner.source, winner.kind)
        src_summary = ", ".join(f"{x.source}:{x.score:.3f}@{x.t:.3f}s" for x in sorted(g, key=lambda y: y.score, reverse=True))
        if len(g) > 1:
            nms_reason_map[key] = (
                f"merged {len(g)} nearby candidates within {merge_window_sec:.2f}s; "
                f"selected highest score from [{src_summary}]"
            )
        else:
            nms_reason_map[key] = "single candidate in merge window"

    filtered = []
    accepted_reasons: Dict[Tuple[float, str, str], str] = {}
    for b in fused:
        if b.source == "audio":
            continue
        if b.source == "camera":
            if b.score >= 0.30:
                filtered.append(b)
                accepted_reasons[(b.t, b.source, b.kind)] = (
                    f"camera final gate passed (score {b.score:.3f} >= 0.300)"
                )
        elif b.score >= 0.35:
            filtered.append(b)
            accepted_reasons[(b.t, b.source, b.kind)] = (
                f"{b.source} final gate passed (score {b.score:.3f} >= 0.350)"
            )

    for b in filtered:
        raw_score = b.score
        w = float(weights.get(b.source, 1.0))
        b.score = raw_score * w
        if print_cut_reasons:
            key = (b.t, b.source, b.kind)
            base_reason = accepted_reasons.get(key, "passed filtering")
            nms_reason = nms_reason_map.get(key, "")
            kind_extra = ""
            if b.source == "transnet" and b.kind == "transition":
                kind_extra = f", transition_score={b.trans_score:.3f}, seg_len={b.seg_len}"
            print(
                "[detect_scenes][CUT]",
                f"t={b.t:.3f}s",
                f"source={b.source}",
                f"kind={b.kind}",
                f"raw={raw_score:.3f}",
                f"weight={w:.3f}",
                f"weighted={b.score:.3f}",
                f"reason={base_reason}; {nms_reason}{kind_extra}",
            )

    scenes = build_scenes_from_boundaries(filtered, video_duration_sec=duration)
    return scenes, filtered


def run_and_write(
    video_path: str,
    transnet_weights_dir: Optional[str] = None,
    output_txt: str = "output.txt",
    cuts_path: str = "cuts.json",
    output_video: str = "output.mp4",
    **detect_kwargs,
) -> List[Tuple[float, float]]:
    if not transnet_weights_dir:
        transnet_weights_dir = _default_transnet_weights_dir()

    scenes, filtered = detect_scenes(video_path, transnet_weights_dir, **detect_kwargs)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=== Boundaries (fused & filtered) ===\n")
        for b in sorted(filtered, key=lambda x: x.t):
            if b.source == "transnet":
                f.write(
                    f"{b.t:7.3f}s  score={b.score:0.3f}  source={b.source}  "
                    f"kind={b.kind}  seg_len={b.seg_len}  trans_score={b.trans_score:0.3f}\n"
                )
            else:
                f.write(f"{b.t:7.3f}s  score={b.score:0.3f}  source={b.source}  kind={b.kind}\n")

        f.write("\n=== Scenes (start_sec, end_sec) ===\n")
        for s, e in scenes:
            f.write(f"[{s:7.3f}, {e:7.3f}]  len={(e-s):0.3f}s\n")

    cuts = [[round(s, 3), round(e, 3)] for s, e in scenes]
    with open(cuts_path, "w", encoding="utf-8") as f:
        json.dump(cuts, f, ensure_ascii=False)

    render_scene_numbers(video_path, scenes, output_path=output_video)
    return scenes


__all__ = [
    "Boundary",
    "run_pyscenedetect",
    "run_transnetv2",
    "run_audio_onset",
    "run_camera_jump",
    "detect_scenes",
    "run_and_write",
]
