"""
RunPod Serverless Handler for GAP Diffusion
Runs SDXL + Multi-ControlNet + IP-Adapter on GPU
"""
import runpod
import torch
import base64
import io
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler
)
from ip_adapter import IPAdapterXL

# Global pipeline (loaded once on cold start)
pipe = None

def load_models():
    """Load models once on cold start"""
    global pipe
    
    print("🔥 Loading models on cold start...")
    
    # Load ControlNets
    print("📥 Loading ControlNet-Depth...")
    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    )
    
    print("📥 Loading ControlNet-Normal...")
    controlnet_normal = ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0",  # Using as placeholder for normal
        torch_dtype=torch.float16
    )
    
    # Load SDXL pipeline with Multi-ControlNet
    print("📥 Loading SDXL base model...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[controlnet_depth, controlnet_normal],
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # Load IP-Adapter for reference image
    print("📥 Loading IP-Adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin"
    )
    pipe.set_ip_adapter_scale(0.8)  # Strong reference image influence
    
    # Optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    # Use DPM++ 2M scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    
    print("✅ Models loaded successfully")
    return pipe

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return img

def encode_image_to_base64(image):
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def handler(event):
    """
    RunPod handler function
    
    Expected input:
    {
        "input": {
            "depth_image": "base64_encoded_png",
            "normal_image": "base64_encoded_png",
            "reference_image": "base64_encoded_png",  # NEW: Canonical photo
            "prompt": "a supplement bottle...",
            "negative_prompt": "blurry, distorted...",
            "seed": 42,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": [1.2, 1.0],  # Stronger
            "ip_adapter_scale": 0.8  # Reference image weight
        }
    }
    
    Returns:
    {
        "image": "base64_encoded_png",
        "seed": 42
    }
    """
    global pipe
    
    # Load models on first run
    if pipe is None:
        load_models()
    
    try:
        input_data = event["input"]
        
        # Decode conditioning images
        depth_img = decode_base64_image(input_data["depth_image"])
        normal_img = decode_base64_image(input_data["normal_image"])
        reference_img = decode_base64_image(input_data.get("reference_image"))
        
        # Preserve aspect ratio from reference
        target_width = reference_img.width
        target_height = reference_img.height
        
        # Resize to nearest multiple of 8 (required by SDXL)
        target_width = (target_width // 8) * 8
        target_height = (target_height // 8) * 8
        
        # Resize conditioning images to match
        depth_img = depth_img.resize((target_width, target_height), Image.LANCZOS)
        normal_img = normal_img.resize((target_width, target_height), Image.LANCZOS)
        reference_img = reference_img.resize((target_width, target_height), Image.LANCZOS)
        
        # Get parameters
        prompt = input_data.get("prompt", "professional product photography")
        negative_prompt = input_data.get(
            "negative_prompt", 
            "blurry, distorted, low quality, different object, deformed, wrong product, text, watermark"
        )
        seed = input_data.get("seed", 42)
        num_steps = input_data.get("num_inference_steps", 50)
        guidance = input_data.get("guidance_scale", 7.5)
        cn_scales = input_data.get("controlnet_conditioning_scale", [1.2, 1.0])
        ip_scale = input_data.get("ip_adapter_scale", 0.8)
        
        # Set IP-Adapter scale
        pipe.set_ip_adapter_scale(ip_scale)
        
        # Setup generator
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate image with IP-Adapter + ControlNet
        print(f"🎨 Generating {target_width}x{target_height} with seed {seed}...")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[depth_img, normal_img],
            ip_adapter_image=reference_img,  # KEY: Reference image
            controlnet_conditioning_scale=cn_scales,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
            height=target_height,
            width=target_width
        ).images[0]
        
        # Encode result
        result_base64 = encode_image_to_base64(result)
        
        return {
            "image": result_base64,
            "seed": seed,
            "width": target_width,
            "height": target_height
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
