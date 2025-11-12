#!/usr/bin/env python3
import os
import json
import base64
from pathlib import Path

import runpod

# Set API key from environment
runpod.api_key = os.environ.get("RUNPOD_API_KEY")

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "gap-animatediff-video-sd15")
MANIFEST_PATH = os.environ.get("GAP_MANIFEST", "output/gaps/supplement_mv/multiview_manifest.json")
OUT_PATH = os.environ.get("GAP_VIDEO_OUT", "output/videos/supplement_mv_turntable.gif")

PROMPT = os.environ.get("GAP_PROMPT", "studio product rotation, clean white background, soft shadow")
FRAMES = int(os.environ.get("GAP_FRAMES", "24"))
STEPS = int(os.environ.get("GAP_STEPS", "25"))
GUIDANCE = float(os.environ.get("GAP_GUIDANCE", "7.5"))
CN_SCALE = float(os.environ.get("GAP_CN_SCALE", "0.6"))
FPS = int(os.environ.get("GAP_FPS", "8"))
FORMAT = os.environ.get("GAP_FORMAT", "gif")


def main():
    rp = runpod.Endpoint(ENDPOINT_ID)
    manifest_bytes = Path(MANIFEST_PATH).read_bytes()
    job = {
        "input": {
            "manifest_b64": base64.b64encode(manifest_bytes).decode(),
            "prompt": PROMPT,
            "frames": FRAMES,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "controlnet_scale": CN_SCALE,
            "fps": FPS,
            "format": FORMAT,
        }
    }
    print("Submitting job to endpoint:", ENDPOINT_ID)
    job = rp.run(job)
    print("Job ID:", job.job_id)
    print("Waiting for job to complete...")
    # Wait for job to complete and get output
    output = job.output()
    if output is None:
        print("Job returned None. Check endpoint configuration:")
        print("  1. Ensure the endpoint has a Docker image with the handler.")
        print("  2. Check RunPod logs at https://www.runpod.io/console/serverless")
        print("  3. Verify handler path is set to: diffusion/runpod_animatediff_handler.py")
        print("  4. Verify handler function name is: handler")
        return
    if not isinstance(output, dict):
        print("Unexpected response:", output)
        return
    if output.get("status") != "ok":
        print("Job error:", json.dumps(output, indent=2))
        return

    video_b64 = output.get("video_b64")
    if not video_b64:
        print("No video_b64 in response")
        return

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(video_b64))
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()
