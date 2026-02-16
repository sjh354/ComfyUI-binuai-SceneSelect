from __future__ import annotations

import subprocess
from typing import List

import librosa


def extract_audio_wav(video_path: str, wav_path: str, sr: int = 22050) -> str:
    """
    ffmpeg로 mono wav 추출. (librosa 안정적으로 로드)
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "wav", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path


def run_audio_onset(video_path: str, tmp_wav: str = "tmp_audio.wav", sr: int = 22050) -> List[Boundary]:
    """
    음악 비트/강한 효과음에 따른 컷을 잡기 위한 onset 후보.
    onset 강도 기반으로 score 부여.
    """
    from . import Boundary

    extract_audio_wav(video_path, tmp_wav, sr=sr)
    y, sr = librosa.load(tmp_wav, sr=sr, mono=True)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="frames")

    if len(onset_env) == 0:
        return []
    env = onset_env.astype(float)
    env_norm = (env - env.min()) / (env.max() - env.min() + 1e-8)

    boundaries: List[Boundary] = []
    for f in onset_frames:
        t = librosa.frames_to_time(f, sr=sr)
        score = float(env_norm[min(f, len(env_norm) - 1)])
        if score >= 0.35:
            boundaries.append(Boundary(t=t, score=score, source="audio", kind="cut"))
    return boundaries
