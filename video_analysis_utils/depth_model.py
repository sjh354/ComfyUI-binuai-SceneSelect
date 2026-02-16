from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = self._pick_device(device)

        if self.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            print(
                "[info] MPS selected but this model may fail on unsupported ops. "
                "Switching to CPU for stable execution. "
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 and --device mps to force MPS fallback."
            )
            self.device = "cpu"

        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load depth model '{model_name}'. "
                f"If this is the first run, ensure internet access to download model weights, "
                f"or pre-download to cache and use --local-files-only. "
                f"Original error: {e}"
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pick_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def predict_depth(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        pred = outputs.predicted_depth

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        depth = pred[0].detach().float().cpu().numpy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax - dmin > 1e-6:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.clip(depth, 0.0, 1.0)
        return depth
