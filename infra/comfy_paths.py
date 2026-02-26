from __future__ import annotations

from pathlib import Path

try:
    import folder_paths
except Exception:
    folder_paths = None


def sam3_model_dir() -> Path:
    if folder_paths is not None:
        try:
            return Path(folder_paths.models_dir) / "sam3"
        except Exception:
            pass
    return Path.cwd() / "models" / "sam3"


def video_candidates() -> list[str]:
    if folder_paths is None:
        return [""]

    try:
        files = folder_paths.get_filename_list("video")
        files = [str(f) for f in files if str(f)]
        if files:
            return sorted(files)
    except Exception:
        pass

    try:
        input_dir = Path(folder_paths.get_input_directory())
        exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        files = [p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if files:
            return sorted(files)
    except Exception:
        pass
    return [""]


def resolve_video_path(video: str) -> Path:
    p = Path(str(video))
    if p.exists():
        return p.resolve()

    if folder_paths is not None:
        try:
            return Path(folder_paths.get_annotated_filepath(str(video))).resolve()
        except Exception:
            pass
        try:
            return (Path(folder_paths.get_input_directory()) / str(video)).resolve()
        except Exception:
            pass

    return p.resolve()
