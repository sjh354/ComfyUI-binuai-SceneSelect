from __future__ import annotations

DEPTH_MODEL_PRESETS = [
    "LiheYoung/depth-anything-small-hf",
    "LiheYoung/depth-anything-base-hf",
    "LiheYoung/depth-anything-large-hf",
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
]

YOLO_MODEL_PRESETS = [
    "yolov8m-seg.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
]

SAM3_CHECKPOINT_URL = "https://huggingface.co/DiffusionWave/sam3/resolve/main/sam3.pt"
SAM3_HF_REPO_CANDIDATES = [
    "facebook/sam3",
    "DiffusionWave/sam3",
    "Justin331/sam3",
    "bodhicitta/sam3",
]
SAM3_URL_CANDIDATES = [
    "https://huggingface.co/DiffusionWave/sam3/resolve/main/sam3.pt",
    "https://huggingface.co/Justin331/sam3/resolve/main/sam3.pt",
    "https://huggingface.co/bodhicitta/sam3/resolve/main/sam3.pt",
    "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt",
]
