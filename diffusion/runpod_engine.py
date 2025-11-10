"""
RunPod serverless engine for SDXL + Multi-ControlNet
Uses your existing RunPod setup from TripoSR
"""
import runpod
import os
import base64
import io
from PIL import Image
from pathlib import Path
import time
import json

class RunPodDiffusionEngine:
    """SDXL + ControlNet via RunPod serverless"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize RunPod client
        
        Args:
            api_key: RunPod API key (or uses RUNPOD_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('RUNPOD_API_KEY')
        if not self.api_key:
            raise ValueError("RunPod API key required. Set RUNPOD_API_KEY env var or pass api_key")
        
        runpod.api_key = self.api_key
        
        # Your RunPod endpoint ID (we'll create this)
        self.endpoint_id = os.environ.get('RUNPOD_ENDPOINT_ID', 'gap-sdxl-controlnet')
        
        print(f"✅ RunPod engine initialized")
        print(f"   Endpoint: {self.endpoint_id}")
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def base64_to_image(self, b64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_data))
    
    def generate_frame(
        self,
        prompt: str,
        depth_map: Image.Image,
        normal_map: Image.Image,
        mask: Image.Image = None,
        reference_image: Image.Image = None,
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: list = [0.8, 0.6],
        seed: int = None
    ) -> Image.Image:
        """
        Generate single frame using RunPod
        
        Args:
            prompt: Text prompt
            depth_map: Depth conditioning image
            normal_map: Normal conditioning image
            mask: Object mask (optional)
            reference_image: For IP-Adapter (optional)
            num_inference_steps: Sampling steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: [depth_weight, normal_weight]
            seed: Random seed for reproducibility
        
        Returns:
            Generated PIL Image
        """
        # Prepare payload
        payload = {
            "input": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "depth_image": self.image_to_base64(depth_map),
                "normal_image": self.image_to_base64(normal_map),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_scale": controlnet_conditioning_scale,
            }
        }
        
        if mask is not None:
            payload["input"]["mask"] = self.image_to_base64(mask)
        
        if reference_image is not None:
            payload["input"]["reference_image"] = self.image_to_base64(reference_image)
        
        if seed is not None:
            payload["input"]["seed"] = seed
        
        # Send to RunPod
        print(f"📤 Sending request to RunPod...")
        
        try:
            endpoint = runpod.Endpoint(self.endpoint_id)
            job = endpoint.run(payload)
            
            # Poll for result
            print(f"⏳ Job ID: {job.job_id}")
            result = job.output(timeout=300)  # 5 min timeout
            
            # Decode result
            if result and "image" in result:
                image = self.base64_to_image(result["image"])
                print(f"✅ Image generated successfully")
                return image
            else:
                raise RuntimeError(f"Unexpected response: {result}")
                
        except Exception as e:
            print(f"❌ RunPod error: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: list,
        depth_maps: list,
        normal_maps: list,
        **kwargs
    ) -> list:
        """
        Generate multiple frames in parallel
        
        Args:
            prompts: List of prompts
            depth_maps: List of depth images
            normal_maps: List of normal images
            **kwargs: Additional args passed to generate_frame
        
        Returns:
            List of generated PIL Images
        """
        print(f"🔄 Generating {len(prompts)} frames in parallel...")
        
        jobs = []
        endpoint = runpod.Endpoint(self.endpoint_id)
        
        # Submit all jobs
        for i, (prompt, depth, normal) in enumerate(zip(prompts, depth_maps, normal_maps)):
            payload = {
                "input": {
                    "prompt": prompt,
                    "depth_image": self.image_to_base64(depth),
                    "normal_image": self.image_to_base64(normal),
                    **kwargs
                }
            }
            
            job = endpoint.run(payload)
            jobs.append((i, job))
            print(f"   Submitted job {i+1}/{len(prompts)}")
        
        # Collect results
        results = [None] * len(jobs)
        for idx, job in jobs:
            result = job.output(timeout=300)
            if result and "image" in result:
                results[idx] = self.base64_to_image(result["image"])
                print(f"   ✅ Frame {idx+1} complete")
        
        return results
    
    def estimate_cost(self, num_frames: int, seconds_per_frame: float = 3.0) -> float:
        """
        Estimate RunPod cost
        
        Args:
            num_frames: Number of frames to generate
            seconds_per_frame: Estimated time per frame
        
        Returns:
            Estimated cost in USD
        """
        # RunPod A100 pricing: ~$2.00/hr = $0.00056/sec
        cost_per_second = 0.00056
        total_seconds = num_frames * seconds_per_frame
        return total_seconds * cost_per_second


def get_runpod_engine() -> RunPodDiffusionEngine:
    """Factory function to get configured RunPod engine"""
    # Try to load API key from TripoSR config if it exists
    triposr_env = Path("/workspaces/TripoSR/.env")
    if triposr_env.exists():
        with open(triposr_env) as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY"):
                    key = line.split("=")[1].strip()
                    os.environ["RUNPOD_API_KEY"] = key
                    break
    
    return RunPodDiffusionEngine()