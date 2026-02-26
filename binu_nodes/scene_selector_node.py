from __future__ import annotations

from binu_domain.constants import DEPTH_MODEL_PRESETS, YOLO_MODEL_PRESETS
from binu_domain.scene_selection import select_best_scene_from_video
from binu_infra.comfy_paths import resolve_video_path, video_candidates


class _SceneSelectorBase:
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "audio", "fps_estimate", "analysis_json")
    FUNCTION = "run"
    CATEGORY = "binu_ai/scenedetect"

    @classmethod
    def _video_candidates(cls) -> list[str]:
        return video_candidates()

    def _run_video_selector(
        self,
        *,
        node_name: str,
        video,
        depth_model,
        depth_device,
        yolo_model,
        yolo_conf,
        yolo_imgsz,
        sample_fps,
        max_frames,
        max_decode_frames_per_scene,
        camera_mavg_window,
        invert_depth,
        local_files_only,
        cache_dir,
        w1_plane,
        w2_camera_motion,
        w3_occlusion,
        w4_flatness,
        w5_lighting,
        w6_texture,
        occlusion_percentile,
        video_path="",
        sam2_model=None,
        sam2_prompt="floor, ground, road",
        sam2_min_mask_ratio=0.01,
        sam2_max_mask_ratio=0.90,
    ):
        chosen = str(video_path).strip() if isinstance(video_path, str) else ""
        resolved = resolve_video_path(chosen if chosen else str(video))
        if not resolved.exists():
            raise FileNotFoundError(f"video not found: {resolved}")

        from video_analysis_utils.kling_pipeline import KlingWeights

        weights = KlingWeights(
            w1_plane=float(w1_plane),
            w2_camera_motion=float(w2_camera_motion),
            w3_occlusion=float(w3_occlusion),
            w4_flatness=float(w4_flatness),
            w5_lighting=float(w5_lighting),
            w6_texture=float(w6_texture),
        )

        return select_best_scene_from_video(
            node_name=node_name,
            video_path=resolved,
            depth_model=depth_model,
            depth_device=depth_device,
            yolo_model=yolo_model,
            yolo_conf=float(yolo_conf),
            yolo_imgsz=int(yolo_imgsz),
            sample_fps=float(sample_fps),
            max_frames=int(max_frames),
            max_decode_frames_per_scene=int(max_decode_frames_per_scene),
            camera_mavg_window=int(camera_mavg_window),
            invert_depth=bool(invert_depth),
            local_files_only=bool(local_files_only),
            cache_dir=str(cache_dir),
            depth_model_presets=DEPTH_MODEL_PRESETS,
            yolo_model_presets=YOLO_MODEL_PRESETS,
            weights=weights,
            occlusion_percentile=float(occlusion_percentile),
            sam2_model=sam2_model,
            sam2_prompt=str(sam2_prompt),
            sam2_min_mask_ratio=float(sam2_min_mask_ratio),
            sam2_max_mask_ratio=float(sam2_max_mask_ratio),
        )


class SceneSelectorUpload(_SceneSelectorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    cls._video_candidates(),
                    {
                        "video_upload": True,
                        "vhs_path_extensions": ["mp4", "mov", "mkv", "webm", "avi", "m4v"],
                    },
                ),
                "depth_model": (DEPTH_MODEL_PRESETS,),
                "depth_device": (["auto", "cpu", "cuda", "mps"],),
                "yolo_model": (YOLO_MODEL_PRESETS,),
                "yolo_conf": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01}),
                "yolo_imgsz": ("INT", {"default": 640, "min": 256, "max": 2048}),
                "sample_fps": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "max_decode_frames_per_scene": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "camera_mavg_window": ("INT", {"default": 5, "min": 1, "max": 31}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cache_dir": ("STRING", {"default": ""}),
                "w1_plane": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w2_camera_motion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w3_occlusion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w4_flatness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w5_lighting": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w6_texture": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "occlusion_percentile": ("FLOAT", {"default": 85.0, "min": 50.0, "max": 99.0, "step": 1.0}),
            },
            "optional": {
                "video_path": ("STRING", {"default": ""}),
            },
        }

    def run(
        self,
        video,
        depth_model,
        depth_device,
        yolo_model,
        yolo_conf,
        yolo_imgsz,
        sample_fps,
        max_frames,
        max_decode_frames_per_scene,
        camera_mavg_window,
        invert_depth,
        local_files_only,
        cache_dir,
        w1_plane,
        w2_camera_motion,
        w3_occlusion,
        w4_flatness,
        w5_lighting,
        w6_texture,
        occlusion_percentile,
        video_path="",
    ):
        return self._run_video_selector(
            node_name="SceneSelectorUpload",
            video=video,
            depth_model=depth_model,
            depth_device=depth_device,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            yolo_imgsz=yolo_imgsz,
            sample_fps=sample_fps,
            max_frames=max_frames,
            max_decode_frames_per_scene=max_decode_frames_per_scene,
            camera_mavg_window=camera_mavg_window,
            invert_depth=invert_depth,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            w1_plane=w1_plane,
            w2_camera_motion=w2_camera_motion,
            w3_occlusion=w3_occlusion,
            w4_flatness=w4_flatness,
            w5_lighting=w5_lighting,
            w6_texture=w6_texture,
            occlusion_percentile=occlusion_percentile,
            video_path=video_path,
        )


class SceneSelectorSAM(SceneSelectorUpload):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    cls._video_candidates(),
                    {
                        "video_upload": True,
                        "vhs_path_extensions": ["mp4", "mov", "mkv", "webm", "avi", "m4v"],
                    },
                ),
                "sam2_model": ("SAM2_MODEL", {"forceInput": True}),
                "sam2_prompt": ("STRING", {"default": "floor, ground, road"}),
                "sam2_min_mask_ratio": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "sam2_max_mask_ratio": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.001}),
                "depth_model": (DEPTH_MODEL_PRESETS,),
                "depth_device": (["auto", "cpu", "cuda", "mps"],),
                "yolo_model": (YOLO_MODEL_PRESETS,),
                "yolo_conf": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01}),
                "yolo_imgsz": ("INT", {"default": 640, "min": 256, "max": 2048}),
                "sample_fps": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "max_decode_frames_per_scene": ("INT", {"default": 0, "min": 0, "max": 20000}),
                "camera_mavg_window": ("INT", {"default": 5, "min": 1, "max": 31}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "cache_dir": ("STRING", {"default": ""}),
                "w1_plane": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w2_camera_motion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w3_occlusion": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w4_flatness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w5_lighting": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "w6_texture": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "occlusion_percentile": ("FLOAT", {"default": 85.0, "min": 50.0, "max": 99.0, "step": 1.0}),
            },
            "optional": {
                "video_path": ("STRING", {"default": ""}),
            },
        }

    def run(
        self,
        video,
        sam2_model,
        sam2_prompt,
        sam2_min_mask_ratio,
        sam2_max_mask_ratio,
        depth_model,
        depth_device,
        yolo_model,
        yolo_conf,
        yolo_imgsz,
        sample_fps,
        max_frames,
        max_decode_frames_per_scene,
        camera_mavg_window,
        invert_depth,
        local_files_only,
        cache_dir,
        w1_plane,
        w2_camera_motion,
        w3_occlusion,
        w4_flatness,
        w5_lighting,
        w6_texture,
        occlusion_percentile,
        video_path="",
    ):
        return self._run_video_selector(
            node_name="SceneSelectorSAM",
            video=video,
            depth_model=depth_model,
            depth_device=depth_device,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            yolo_imgsz=yolo_imgsz,
            sample_fps=sample_fps,
            max_frames=max_frames,
            max_decode_frames_per_scene=max_decode_frames_per_scene,
            camera_mavg_window=camera_mavg_window,
            invert_depth=invert_depth,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            w1_plane=w1_plane,
            w2_camera_motion=w2_camera_motion,
            w3_occlusion=w3_occlusion,
            w4_flatness=w4_flatness,
            w5_lighting=w5_lighting,
            w6_texture=w6_texture,
            occlusion_percentile=occlusion_percentile,
            video_path=video_path,
            sam2_model=sam2_model,
            sam2_prompt=sam2_prompt,
            sam2_min_mask_ratio=sam2_min_mask_ratio,
            sam2_max_mask_ratio=sam2_max_mask_ratio,
        )
