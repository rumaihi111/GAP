#!/usr/bin/env python3
import os, json, base64, tempfile
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# Try to import AnimateDiff (optional). We’ll fall back to per-frame generation if missing.
def _maybe_import_animatediff():
    try:
        ad_root = Path(__file__).resolve().parents[1] / "AnimateDiff"
        if ad_root.exists():
            import sys
            sys.path.append(str(ad_root))
        from animatediff.pipelines.pipeline_animatediff import AnimateDiffPipeline  # type: ignore
        return AnimateDiffPipeline
    except Exception:
        return None

AnimateDiffPipeline = _maybe_import_animatediff()

# Models (SD 1.5 path is much lighter than SDXL)
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "runwayml/stable-diffusion-v1-5")
CONTROLNET_ID = os.environ.get("CONTROLNET_ID", "lllyasviel/controlnet-depth")
MOTION_MODULE_PATH = os.environ.get("MOTION_MODULE_PATH", "/app/models/Motion_Module/mm_sd_v15_v2.ckpt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

_cached = {"pipe": None, "mode": None}

def _load_pipelines():
    if _cached["pipe"] is not None:
        return _cached["pipe"], _cached["mode"]

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=DTYPE)
    if AnimateDiffPipeline is not None and Path(MOTION_MODULE_PATH).exists():
        try:
            pipe = AnimateDiffPipeline.from_pretrained(
                BASE_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=DTYPE,
            )
            if DEVICE == "cuda":
                pipe = pipe.to("cuda")
            pipe.load_motion_module(MOTION_MODULE_PATH)
            _cached.update(pipe=pipe, mode="animatediff")
            return pipe, "animatediff"
        except Exception as e:
            print(f"[Animatediff] init failed, falling back: {e}")

    # Fallback: per-frame ControlNet pipeline
    from diffusers import StableDiffusionControlNetPipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=DTYPE,
    )
    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
    _cached.update(pipe=pipe, mode="sd15")
    return pipe, "sd15"

def _decode_manifest(manifest_b64: str) -> Dict[str, Any]:
    data = base64.b64decode(manifest_b64)
    return json.loads(data)

def _load_depth_and_refs(manifest: Dict[str, Any], frames: int):
    """Load depth/reference images from manifest (supports both file paths and base64)."""
    import io
    angles = sorted(manifest["views"].keys(), key=lambda a: int(a))
    
    depths = []
    refs = []
    for angle in angles:
        view = manifest["views"][angle]
        
        # Load depth (try base64 first, then path)
        if "depth_b64" in view:
            depth_data = base64.b64decode(view["depth_b64"])
            depths.append(Image.open(io.BytesIO(depth_data)).convert("RGB"))
        elif "depth" in view:
            depths.append(Image.open(view["depth"]).convert("RGB"))
        else:
            raise ValueError(f"View {angle} missing depth or depth_b64")
        
        # Load reference (optional, try base64 first, then path)
        if "reference_b64" in view:
            ref_data = base64.b64decode(view["reference_b64"])
            refs.append(Image.open(io.BytesIO(ref_data)).convert("RGB"))
        elif "reference" in view:
            refs.append(Image.open(view["reference"]).convert("RGB"))
        else:
            # Use depth as reference if not provided
            refs.append(depths[-1])
    
    # Repeat to reach target frames
    rep = (frames + len(depths) - 1) // len(depths)
    depths = (depths * rep)[:frames]
    refs = (refs * rep)[:frames]
    return depths, refs

def _save_gif(frames: List[Image.Image], out_path: Path, fps: int = 8):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      {
        "manifest_b64": "<base64 json of multiview_manifest.json>",
        "prompt": "text",
        "frames": 24,
        "steps": 25,
        "guidance": 7.5,
        "controlnet_scale": 0.6,
        "fps": 8,
        "format": "gif"  # or "mp4" (gif implemented here)
      }
    """
    try:
        manifest_b64 = event["input"]["manifest_b64"]
        prompt = event["input"].get("prompt", "studio product rotation, clean white background")
        frames = int(event["input"].get("frames", 24))
        steps = int(event["input"].get("steps", 25))
        guidance = float(event["input"].get("guidance", 7.5))
        cn_scale = float(event["input"].get("controlnet_scale", 0.6))
        fps = int(event["input"].get("fps", 8))
        fmt = event["input"].get("format", "gif")

        manifest = _decode_manifest(manifest_b64)
        depths, refs = _load_depth_and_refs(manifest, frames)

        pipe, mode = _load_pipelines()

        if mode == "animatediff":
            result = pipe(
                prompt=prompt,
                controlnet_conditioning_frames=depths,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=cn_scale,
            )
            vid_frames = result.frames[0]
        else:
            # Per-frame fallback; ignores refs to keep it simple
            vid_frames = []
            for d in depths:
                out = pipe(
                    prompt=prompt,
                    image=d,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    controlnet_conditioning_scale=cn_scale,
                )
                vid_frames.append(out.images[0])

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            if fmt == "gif":
                out_path = td / "video.gif"
                _save_gif(vid_frames, out_path, fps=fps)
                data = out_path.read_bytes()
                return {"status": "ok", "mode": mode, "format": "gif", "video_b64": base64.b64encode(data).decode()}
            else:
                # mp4 encoding not implemented here (needs imageio-ffmpeg)
                return {"status": "error", "message": "mp4 not implemented; use format=gif"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Allow running this file directly in a serverless container without
# relying on an external module entrypoint. When invoked as
# `python /app/handler.py`, this will register the handler with
# RunPod's serverless runtime and begin processing requests.
if __name__ == "__main__":
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except Exception as _e:
        # Surface the error to container logs for faster debugging
        print(f"[serverless] failed to start: {_e}")
        raise