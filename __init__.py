from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_pkg_on_path() -> None:
    node_dir = Path(__file__).resolve().parent
    if str(node_dir) not in sys.path:
        sys.path.insert(0, str(node_dir))


_ensure_local_pkg_on_path()

from binu_nodes.prompt_node import KlingPromptFromAnalysis
from binu_nodes.sam3_loader_node import SAM3ModelLoader
from binu_nodes.scene_selector_node import SceneSelectorSAM, SceneSelectorUpload

NODE_CLASS_MAPPINGS = {
    "SAM3ModelLoader": SAM3ModelLoader,
    "KlingPromptFromAnalysis": KlingPromptFromAnalysis,
    "SceneSelectorUpload": SceneSelectorUpload,
    "SceneSelectorSAM": SceneSelectorSAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3ModelLoader": "SAM3 Model Loader",
    "KlingPromptFromAnalysis": "Kling Prompt Builder (Analysis + Caption)",
    "SceneSelectorUpload": "Scene Selector (Upload)",
    "SceneSelectorSAM": "Scene Selector (SAM)",
}
