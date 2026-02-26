from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.kling_analysis.pipeline import compact_score_meta, normalize_choice


def test_normalize_choice_uses_value_when_valid():
    assert normalize_choice("b", ["a", "b", "c"], "a") == "b"


def test_normalize_choice_uses_fallback_when_invalid():
    assert normalize_choice("x", ["a", "b", "c"], "b") == "b"


def test_compact_score_meta_drops_sensitive_sam2_prompt_fields():
    meta = {
        "score": 0.9,
        "components": {"a": 1},
        "sam2_stats": {
            "sam2_enabled": True,
            "sam2_prompt": "floor",
            "sam2_min_mask_ratio": 0.1,
            "sam2_max_mask_ratio": 0.9,
            "sam2_used_count": 5,
        },
    }
    out = compact_score_meta(meta)
    assert out["score"] == 0.9
    assert out["sam2_stats"]["sam2_enabled"] is True
    assert out["sam2_stats"]["sam2_used_count"] == 5
    assert "sam2_prompt" not in out["sam2_stats"]
    assert "sam2_min_mask_ratio" not in out["sam2_stats"]
    assert "sam2_max_mask_ratio" not in out["sam2_stats"]
