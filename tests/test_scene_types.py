from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binu_domain.scene_types import AnalysisSummary, to_analysis_json


def test_to_analysis_json_serializes_summary():
    summary = AnalysisSummary(
        node="SceneSelectorUpload",
        scene_detection={"scene_count": 2, "used_single_scene_fallback": False},
        selected_scene={"start_sec": 1.0, "end_sec": 2.0},
        analysis={"score": 0.8},
    )
    payload = json.loads(to_analysis_json(summary))
    assert payload["node"] == "SceneSelectorUpload"
    assert payload["analysis"]["score"] == 0.8
