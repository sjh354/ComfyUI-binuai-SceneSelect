from __future__ import annotations

from binu_infra.comfy_paths import sam3_model_dir
from binu_infra.model_hub import build_sam3_payload, download_sam3_checkpoint


class SAM3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cpu", "cuda", "mps"],),
                "force_download": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAM2_MODEL", "STRING")
    RETURN_NAMES = ("sam2_model", "checkpoint_path")
    FUNCTION = "run"
    CATEGORY = "binu_ai/scenedetect"

    def run(self, device, force_download):
        ckpt_path = download_sam3_checkpoint(sam3_model_dir(), bool(force_download))
        payload = build_sam3_payload(checkpoint_path=ckpt_path, device=str(device))
        return (payload, str(ckpt_path))
