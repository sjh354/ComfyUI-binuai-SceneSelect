from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch


def read_audio_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 44100,
):
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 1e-3, float(end_sec))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    raw = proc.stdout if proc.returncode == 0 else b""
    if not raw:
        waveform = torch.zeros((1, 2, 1), dtype=torch.float32)
        return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}

    audio_np = np.frombuffer(raw, dtype=np.float32)
    if audio_np.size < 2:
        waveform = torch.zeros((1, 2, 1), dtype=torch.float32)
        return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}

    n = (audio_np.size // 2) * 2
    audio_np = audio_np[:n].reshape(-1, 2)
    waveform = torch.from_numpy(audio_np.T.copy()).unsqueeze(0).to(dtype=torch.float32)
    return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}
