from __future__ import annotations

import json
from collections.abc import Mapping


class KlingPromptFromAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "reference_caption": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "binu_ai/scenedetect"

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _risk_label(v: float, good_hi: float = 0.7, warn_lo: float = 0.4) -> str:
        if v < warn_lo:
            return "high risk"
        if v < good_hi:
            return "moderate risk"
        return "low risk"

    def run(self, analysis_json, reference_caption):
        parsed = {}
        if isinstance(analysis_json, Mapping):
            parsed = dict(analysis_json)
        else:
            raw = str(analysis_json or "").strip()
            if raw:
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, Mapping):
                        parsed = dict(obj)
                except Exception:
                    parsed = {}

        analysis = parsed.get("analysis", {}) if isinstance(parsed, Mapping) else {}
        metrics = analysis.get("metrics", {}) if isinstance(analysis, Mapping) else {}
        selected_scene = parsed.get("selected_scene", {}) if isinstance(parsed, Mapping) else {}

        score = self._to_float(analysis.get("score", 0.0))
        cam_score = self._to_float(metrics.get("camera_motion", {}).get("camera_motion_score", 0.0))
        occ_score = self._to_float(metrics.get("occlusion", {}).get("occlusion_score", 0.0))
        flat_score = self._to_float(metrics.get("flatness", {}).get("flatness_score", 0.0))
        light_score = self._to_float(metrics.get("lighting", {}).get("lighting_stability", 0.0))
        tex_score = self._to_float(metrics.get("texture", {}).get("texture_score", 0.0))
        plane_score = self._to_float(metrics.get("plane_score", 0.0))
        plane_count = int(self._to_float(metrics.get("plane_count", 0)))
        pitch_hint = str(metrics.get("pitch_hint", "level"))
        height_hint = str(metrics.get("camera_height_hint", "eye"))
        scene_start = self._to_float(selected_scene.get("start_sec", 0.0))
        scene_end = self._to_float(selected_scene.get("end_sec", 0.0))

        caption = str(reference_caption or "").strip() or "an object"
        caption_phrase = caption.rstrip(" .")
        object_ref = "the target object"

        base_prompt = (
            f"Insert {caption_phrase} from [@Image] into [@Video].\n\n"
            f"Target object: {caption_phrase}\n\n"
            "Placement:\n"
            f"- MUST anchor {object_ref} to one fixed floor position for the full clip.\n"
            f"- MUST preserve the same footprint, orientation, and scale of {object_ref} in every frame.\n\n"
            "Appearance:\n"
            f"- MUST match color, tone, and material appearance of {object_ref} from [@Image].\n"
            f"- MUST keep natural perspective and contact shadow of {object_ref} on the floor.\n\n"
            "Temporal consistency:\n"
            f"- NEVER translate, rotate, resize, warp, or jitter {object_ref} over time.\n"
            f"- NEVER introduce flicker in {object_ref} edges, color, or shading.\n\n"
            "Occlusion behavior:\n"
            f"- NEVER let moving foreground objects erase, cut, or deform {object_ref}.\n"
            f"- Keep {object_ref} geometrically stable even during crossings."
        )

        risk_lines = []
        if occ_score < 0.40:
            risk_lines.append(
                f"- High occlusion risk (occlusion_safety={occ_score:.2f}): prioritize integrity under foreground crossings."
            )
        if cam_score < 0.70:
            risk_lines.append(
                f"- Camera motion risk (camera_motion={cam_score:.2f}): enforce strict floor-lock and zero drift."
            )
        if flat_score < 0.40:
            risk_lines.append(
                f"- Flatness risk (flatness={flat_score:.2f}): avoid floating/sinking; maintain firm ground contact."
            )
        if light_score < 0.50:
            risk_lines.append(
                f"- Lighting risk (lighting_stability={light_score:.2f}): keep shadow direction/intensity temporally coherent."
            )
        if tex_score < 0.50:
            risk_lines.append(
                f"- Texture risk (texture_stability={tex_score:.2f}): suppress edge shimmer and local jitter on textured ground."
            )
        if plane_score < 0.35 or plane_count <= 1:
            risk_lines.append(
                f"- Floor-plane confidence risk (plane_score={plane_score:.2f}, plane_count={plane_count}): keep placement conservative and locked to dominant lower-plane cues."
            )

        metrics_brief = [
            "Scene analysis brief:",
            f"- Overall stability: {self._risk_label(score)} ({score:.2f}).",
            f"- Camera motion: {self._risk_label(cam_score)} ({cam_score:.2f}); lower means drift pressure is higher.",
            f"- Occlusion safety: {self._risk_label(occ_score)} ({occ_score:.2f}); lower means crossings are likely.",
            f"- Floor flatness: {self._risk_label(flat_score)} ({flat_score:.2f}); lower means floating/sinking risk.",
            f"- Lighting stability: {self._risk_label(light_score)} ({light_score:.2f}); lower means shadow inconsistency risk.",
            f"- Texture stability: {self._risk_label(tex_score)} ({tex_score:.2f}); lower means edge jitter risk.",
            (
                f"- Floor-plane confidence: {self._risk_label(plane_score)} ({plane_score:.2f}, planes={plane_count}); "
                "lower means anchoring is less reliable."
            ),
            f"- Camera orientation hint: pitch={pitch_hint}, height={height_hint}.",
            f"- Selected scene window: {scene_start:.3f}s-{scene_end:.3f}s.",
        ]
        metrics_summary = "\n".join(metrics_brief)

        if risk_lines:
            prompt = base_prompt + "\n\nRisk directives:\n" + "\n".join(risk_lines) + "\n\n" + metrics_summary
        else:
            prompt = base_prompt + "\n\nNo major risk flags detected.\n\n" + metrics_summary
        return (prompt,)
