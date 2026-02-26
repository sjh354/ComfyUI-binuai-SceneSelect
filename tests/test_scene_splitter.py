from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.scene_split.splitter import split_from_video


def test_split_from_video_wraps_segments():
    fake = types.ModuleType("scene_split_fusion")
    fake.detect_scenes = lambda *_args, **_kwargs: ([(1.0, 2.5), (4.0, 5.0)], {})
    sys.modules["scene_split_fusion"] = fake

    scenes, fallback = split_from_video(Path("/tmp/x.mp4"), fps=24.0, total_frames=240, camera_mavg_window=5)
    assert fallback is False
    assert len(scenes) == 2
    assert scenes[0].start_sec == 1.0
    assert scenes[0].end_sec == 2.5


def test_split_from_video_fallback_when_empty():
    fake = types.ModuleType("scene_split_fusion")
    fake.detect_scenes = lambda *_args, **_kwargs: ([], {})
    sys.modules["scene_split_fusion"] = fake

    scenes, fallback = split_from_video(Path("/tmp/x.mp4"), fps=20.0, total_frames=100, camera_mavg_window=5)
    assert fallback is True
    assert len(scenes) == 1
    assert scenes[0].start_sec == 0.0
    assert scenes[0].end_sec == 5.0
