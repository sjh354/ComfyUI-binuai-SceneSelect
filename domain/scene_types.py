from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class SceneSegment:
    start_sec: float
    end_sec: float
    confidence: Optional[float] = None
    signals_used: list[str] = field(default_factory=list)
    debug_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneScore:
    score: float
    sampled_frames: int
    scene_total_frames: int
    sampling_step: int
    sampled_frame_indices: list[int]
    components: dict[str, float]
    metrics: dict[str, Any]
    mask_stats: dict[str, float]
    motion_signal_stats: dict[str, float]
    frame_debug_samples: list[dict[str, Any]]
    sam2_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedScene:
    start_sec: float
    end_sec: float
    duration_sec: float
    selected_frame_count: int
    kling_stability: float


@dataclass
class AnalysisSummary:
    node: str
    scene_detection: dict[str, Any]
    selected_scene: dict[str, Any]
    analysis: dict[str, Any]


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    return str(obj)


def to_analysis_json(summary: AnalysisSummary | dict[str, Any]) -> str:
    payload = asdict(summary) if isinstance(summary, AnalysisSummary) else summary
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
