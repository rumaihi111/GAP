#!/usr/bin/env python3
import argparse, json, sys, os
from pathlib import Path
from PIL import Image
import torch

def _ensure_ad_import():
    # Add local AnimateDiff repo to sys.path if present
    ad_repo = Path(__file__).resolve().parents[1] / "AnimateDiff"
    if ad_repo.exists():
        sys.path.append(str(ad_repo))

_ensure_ad_import()

# Version checks to catch the cached_download mismatch early
def _check_versions():
    try:
        import diffusers, huggingface_hub
        from packaging import version
        dv = version.parse(diffusers.__version__)
        hv = version.parse(huggingface_hub.__version__)
        print(f"[versions] diffusers={dv}, huggingface_hub={hv}")
        # diffusers < 0.24 may rely on cached_download with newer hub; require >=0.29 for safety
        if dv < version.parse("0.29.0"):
            print("Warning: diffusers < 0.29.0 may be incompatible. Run: pip install 'diffusers==0.29.1'")
        if hv < version.parse("0.22.0"):
            print("Warning: huggingface_hub < 0.22.0 is quite old. Run: pip install 'huggingface_hub==0.23.5'")
    except Exception as e:
        print(f"[versions] check failed: {e}")

_check_versions()

# Ensure Hugging Face cache uses a large volume if available
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# Diffusers (ControlNet + base pipeline)
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# Try multiple AnimateDiff import paths
AnimateDiffPipeline = None
try:
    # Common path in the repo
    from animatediff.pipelines.pipeline_animatediff import AnimateDiffPipeline as _ADP
    AnimateDiffPipeline = _ADP
except Exception:
    try:
        from animatediff.pipelines import AnimateDiffPipeline as _ADP
        AnimateDiffPipeline = _ADP
    except Exception:
        # Not available; fallback later
        pass

def load_manifest(path: Path):
    with open(path) as f:
        return json.load(f)

def build_pipeline(motion_module_path: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load ControlNet (depth)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/controlnet-depth",
        torch_dtype=dtype
    )

    if AnimateDiffPipeline is not None:
        try:
            pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=dtype
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            # Load motion module
            if not Path(motion_module_path).exists():
                raise FileNotFoundError(f"Motion module missing: {motion_module_path}")
            pipe.load_motion_module(motion_module_path)
            print("Using AnimateDiff pipeline.")
            return pipe, "animatediff"
        except Exception as e:
            print(f"AnimateDiff init failed: {e}")
            print("Falling back to single-frame pipeline.")

    # Fallback: single-frame SD 1.5 ControlNet
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    print("Using SD 1.5 ControlNet fallback.")
    return pipe, "sd15"

def prepare_depth_frames(manifest, target_frames: int):
    angles_sorted = sorted(manifest["views"].keys(), key=lambda a: int(a))
    depth_images = [Image.open(manifest["views"][a]["depth"]).convert("RGB") for a in angles_sorted]
    repeats = (target_frames + len(depth_images) - 1) // len(depth_images)
    frames = (depth_images * repeats)[:target_frames]
    return frames

def prepare_reference_images(manifest):
    angles_sorted = sorted(manifest["views"].keys(), key=lambda a: int(a))
    refs = [Image.open(manifest["views"][a]["reference"]).convert("RGB") for a in angles_sorted]
    return refs

def save_gif(frames, out_path: Path, fps: int):
    duration_ms = int(1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to multiview_manifest.json")
    ap.add_argument("--motion-module", default="models/Motion_Module/mm_sd_v15_v2.ckpt")
    ap.add_argument("--prompt", default="studio product rotation, clean white background, soft shadow")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="output/videos/supplement_mv_turntable.gif")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--controlnet-scale", type=float, default=0.6)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    depth_frames = prepare_depth_frames(manifest, args.frames)
    reference_images = prepare_reference_images(manifest)

    pipe, pipe_type = build_pipeline(args.motion_module)

    if pipe_type == "animatediff":
        # Depending on AnimateDiff fork, arguments may differ; start with this
        result = pipe(
            prompt=args.prompt,
            controlnet_conditioning_frames=depth_frames,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.controlnet_scale,
        )
        frames = result.frames[0]
    else:
        # Fallback: one frame per angle
        frames = []
        for depth_img, ref_img in zip(depth_frames, reference_images):
            out = pipe(
                prompt=args.prompt,
                image=depth_img,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                controlnet_conditioning_scale=args.controlnet_scale,
            )
            frames.append(out.images[0])

    save_gif(frames, Path(args.out), fps=args.fps)
    print(f"✅ Video saved ({pipe_type} mode): {args.out}")

if __name__ == "__main__":
    main()