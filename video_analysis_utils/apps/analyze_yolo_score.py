#!/usr/bin/env python3
"""YOLO tracking-based video object quality scorer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import cv2


def compute_video_scores(
    num_ids: int,
    per_id_confidence: Dict[int, Dict[str, float]],
    count_scale: float = 10.0,
    alpha: float = 0.8,
) -> Dict[str, float]:
    object_count_score = 1.0 - math.exp(-num_ids / count_scale)

    if not per_id_confidence:
        confidence_quality_score = 0.0
    else:
        weighted_sum = 0.0
        weight_sum = 0.0
        for stat in per_id_confidence.values():
            quality = alpha * stat["avg_confidence"] + (1.0 - alpha) * stat["max_confidence"]
            weight = math.log1p(stat["samples"])
            weighted_sum += quality * weight
            weight_sum += weight
        confidence_quality_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    return {
        "object_count_score": object_count_score,
        "confidence_quality_score": confidence_quality_score,
    }


def run_yolo_tracking(
    video_path: Path,
    model_path: str,
    tracker: str,
    show_window: bool,
    conf: float,
    iou: float,
    imgsz: int,
) -> Dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    seen_ids = set()
    frame_idx = 0
    per_id_conf_stats = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        r0 = results[0]

        if (
            r0.boxes is not None
            and getattr(r0.boxes, "id", None) is not None
            and getattr(r0.boxes, "conf", None) is not None
        ):
            ids = r0.boxes.id.int().tolist()
            confs = r0.boxes.conf.tolist()
            for obj_id, obj_conf in zip(ids, confs):
                seen_ids.add(obj_id)
                if obj_id not in per_id_conf_stats:
                    per_id_conf_stats[obj_id] = {
                        "sum_conf": 0.0,
                        "count": 0,
                        "max_conf": float("-inf"),
                    }
                stat = per_id_conf_stats[obj_id]
                stat["sum_conf"] += obj_conf
                stat["count"] += 1
                stat["max_conf"] = max(stat["max_conf"], obj_conf)

        if show_window:
            annotated = r0.plot()
            cv2.imshow("YOLO Track (ID)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    id_list = sorted(seen_ids)
    per_id_confidence = {}
    for obj_id in sorted(per_id_conf_stats):
        stat = per_id_conf_stats[obj_id]
        per_id_confidence[obj_id] = {
            "avg_confidence": stat["sum_conf"] / stat["count"],
            "max_confidence": stat["max_conf"],
            "samples": stat["count"],
        }

    video_scores = compute_video_scores(
        num_ids=len(seen_ids),
        per_id_confidence=per_id_confidence,
    )

    return {
        "num_ids": len(seen_ids),
        "ids": id_list,
        "frame_count": frame_idx,
        "per_id_confidence": per_id_confidence,
        "object_count_score": video_scores["object_count_score"],
        "confidence_quality_score": video_scores["confidence_quality_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO tracking based video score analyzer")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    result = run_yolo_tracking(
        video_path=args.video,
        model_path=args.model,
        tracker=args.tracker,
        show_window=args.show_window,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
