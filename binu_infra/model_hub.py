from __future__ import annotations

import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from binu_domain.constants import SAM3_CHECKPOINT_URL, SAM3_HF_REPO_CANDIDATES, SAM3_URL_CANDIDATES


class SAM3ModelWrapper:
    """Best-effort SAM3 wrapper exposing predict()/__call__ for downstream nodes."""

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.backend = None
        self.predictor = None
        self.init_error = ""
        self._init_predictor()

    def _init_predictor(self) -> None:
        try:
            import importlib

            sam3_mod = importlib.import_module("sam3")
        except Exception as exc:
            self.init_error = f"sam3 package import failed: {exc}"
            return

        candidates = [
            ("build_sam3_text_predictor", {"checkpoint": self.checkpoint_path, "device": self.device}),
            ("build_predictor", {"checkpoint": self.checkpoint_path, "device": self.device}),
            ("SAM3TextPredictor", {"checkpoint": self.checkpoint_path, "device": self.device}),
            ("SAM3Predictor", {"checkpoint": self.checkpoint_path, "device": self.device}),
            ("SAM3", {"checkpoint": self.checkpoint_path, "device": self.device}),
        ]

        for name, kwargs in candidates:
            ctor = getattr(sam3_mod, name, None)
            if not callable(ctor):
                continue
            try:
                self.predictor = ctor(**kwargs)
                self.backend = name
                self.init_error = ""
                return
            except Exception:
                continue

        self.init_error = "No compatible SAM3 predictor builder found in installed sam3 package."

    def _infer(self, image, prompt: str):
        if self.predictor is None:
            raise RuntimeError(
                "SAM3 predictor is not initialized. "
                f"checkpoint={self.checkpoint_path} device={self.device} reason={self.init_error}"
            )

        call_patterns = [
            lambda fn: fn(image=image, text_prompt=prompt),
            lambda fn: fn(image=image, prompt=prompt),
            lambda fn: fn(image=image, text=prompt),
            lambda fn: fn(image, prompt),
        ]

        methods = []
        if callable(self.predictor):
            methods.append(self.predictor)
        for name in ("predict", "segment", "infer", "__call__"):
            m = getattr(self.predictor, name, None)
            if callable(m):
                methods.append(m)

        last_err = None
        for method in methods:
            for pattern in call_patterns:
                try:
                    return pattern(method)
                except Exception as exc:
                    last_err = exc
                    continue

        raise RuntimeError(f"SAM3 inference call failed for all patterns: {last_err}")

    def predict(self, image=None, prompt: str = "", text_prompt: str = "", text: str = ""):
        q = str(text_prompt or prompt or text or "").strip()
        return self._infer(image=image, prompt=q)

    def __call__(self, image=None, prompt: str = "", text_prompt: str = "", text: str = ""):
        q = str(text_prompt or prompt or text or "").strip()
        return self._infer(image=image, prompt=q)


def download_sam3_checkpoint(model_dir: Path, force_download: bool) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "sam3.pt"
    if out_path.exists() and not force_download:
        return out_path

    errors = []

    try:
        from huggingface_hub import hf_hub_download

        for repo_id in SAM3_HF_REPO_CANDIDATES:
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename="sam3.pt",
                    local_dir=str(model_dir),
                )
                return Path(downloaded)
            except Exception as exc:
                errors.append(f"hf_hub_download[{repo_id}]={exc}")
    except Exception as exc:
        errors.append(f"huggingface_hub_import={exc}")

    for ckpt_url in SAM3_URL_CANDIDATES:
        try:
            urllib.request.urlretrieve(ckpt_url, str(out_path))
            return out_path
        except (HTTPError, URLError, OSError) as exc:
            errors.append(f"urlretrieve[{ckpt_url}]={exc}")
            continue

    raise RuntimeError("SAM3 checkpoint download failed. " + " | ".join(errors))


def build_sam3_payload(checkpoint_path: Path, device: str) -> dict:
    wrapped = SAM3ModelWrapper(checkpoint_path=str(checkpoint_path), device=str(device))
    return {
        "model": wrapped,
        "checkpoint_path": str(checkpoint_path),
        "model_type": "sam3",
        "checkpoint_url": SAM3_CHECKPOINT_URL,
    }
