from __future__ import annotations

from typing import List

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def run_pyscenedetect(video_path: str, threshold: float = 27.0, min_scene_len_frames: int = 8) -> List[Boundary]:
    """
    ContentDetector 기반. 쇼츠는 변화가 많아 threshold를 약간 올리는게 FP 줄이는 데 유리할 때가 많습니다.
    """
    from . import Boundary
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames))
    scene_manager.detect_scenes(video, show_progress=False)
    scenes = scene_manager.get_scene_list()

    boundaries = []
    for i, (start, end) in enumerate(scenes):
        if i == 0:
            continue
        t = start.get_seconds()
        boundaries.append(Boundary(t=t, score=0.6, source="pyscene", kind="cut"))
    return boundaries
