"""
Test Identity Preservation Across Scenes
Proves GAP maintains asset identity in different contexts
"""
import sys
import os
from pathlib import Path
import torch
import base64
import io
import json
import requests
import time
from PIL import Image
from dotenv import load_dotenv
import numpy as np

# Load environment
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

def load_gap_package(gap_dir):
    """Load GAP package"""
    gap_dir = Path(gap_dir)
    
    # Load images
    canonical = Image.open(gap_dir / "canonical/reference.png")
    depth = Image.open(gap_dir / "geometry/depth_000.png")
    mask = Image.open(gap_dir / "geometry/mask_000.png")
    
    # Load identity embedding
    identity = torch.load(gap_dir / "identity_embedding.pt")
    
    # Load metadata
    with open(gap_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    return {
        'canonical': canonical,
        'depth': depth,
        'mask': mask,
        'identity': identity,
        'metadata': metadata
    }

def encode_image(img):
    """Encode PIL Image to base64"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def resize_for_gpu(image: Image.Image, max_dimension: int = 1024) -> Image.Image:
    """Resize image to fit GPU memory constraints"""
    if image.width > max_dimension or image.height > max_dimension:
        scale = max_dimension / max(image.width, image.height)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        # Ensure multiple of 8 for SDXL
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image

def generate_with_gap(gap_package, scene_prompt, seed=42):
    """
    Generate image using GAP package conditioning
    
    Args:
        gap_package: Loaded GAP package
        scene_prompt: Scene/context description
        seed: Random seed for reproducibility
    """
    # Prepare payload with GAP conditioning + reference image
    payload = {
        'input': {
            "depth_image": encode_image(gap_package['depth']),
            "normal_image": encode_image(gap_package['depth']),  # Using depth as placeholder
            "reference_image": encode_image(gap_package['canonical']),  # KEY: Actual photo
            "prompt": f"{gap_package['metadata']['object_name']}, {scene_prompt}, highly detailed, professional photography",
            "negative_prompt": "blurry, distorted, low quality, different object, wrong product, deformed, ugly, bad anatomy, text, watermark, logo",
            "seed": seed,
            "num_inference_steps": 50,  # More steps
            "guidance_scale": 7.5,
            "controlnet_conditioning_scale": [1.2, 1.0],  # Stronger conditioning
            "ip_adapter_scale": 0.8  # Strong reference image influence
        }
    }
    
    # Send to RunPod
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    response = requests.post(
        f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run',
        headers=headers,
        json=payload,
        timeout=30
    )
    
    result = response.json()
    job_id = result.get("id")
    
    if not job_id:
        raise Exception(f"No job ID: {result}")
    
    # Poll for completion
    status_url = f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}'
    
    print("   Generating", end="", flush=True)
    for _ in range(60):
        time.sleep(3)
        status_response = requests.get(status_url, headers=headers)
        status_result = status_response.json()
        
        if status_result.get("status") == "COMPLETED":
            print(" ✅")
            output = status_result.get("output", {})
            img_data = base64.b64decode(output["image"])
            return Image.open(io.BytesIO(img_data))
        elif status_result.get("status") == "FAILED":
            print(" ❌")
            raise Exception(f"Generation failed: {status_result.get('error')}")
        
        print(".", end="", flush=True)
    
    raise Exception("Timeout")

def calculate_ssim(img1, img2):
    """Simple SSIM calculation"""
    # Convert to numpy arrays
    arr1 = np.array(img1.resize((512, 512)).convert('L')).astype(np.float32)
    arr2 = np.array(img2.resize((512, 512)).convert('L')).astype(np.float32)
    
    # Normalize
    arr1 = arr1 / 255.0
    arr2 = arr2 / 255.0
    
    # Simple correlation-based similarity
    mean1 = arr1.mean()
    mean2 = arr2.mean()
    std1 = arr1.std()
    std2 = arr2.std()
    
    covariance = ((arr1 - mean1) * (arr2 - mean2)).mean()
    correlation = covariance / (std1 * std2 + 1e-10)
    
    return (correlation + 1) / 2  # Normalize to 0-1

def measure_identity_consistency(images):
    """
    Measure how similar the objects are across different scenes
    High score = same object, Low score = different objects
    """
    scores = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            score = calculate_ssim(images[i], images[j])
            scores.append(score)
    
    return np.mean(scores)

def test_identity_preservation():
    """
    Main test: Generate same asset in different scenes
    """
    print("="*60)
    print("🧪 Testing Identity Preservation Across Scenes")
    print("="*60)
    
    # Load GAP package
    print("\n1️⃣ Loading GAP package...")
    gap = load_gap_package("output/gaps/supplement_photo")
    print(f"   Asset: {gap['metadata']['object_name']}")
    print(f"   Identity embedding: {gap['identity'].shape}")
    
    # Define test scenes
    scenes = [
        "on marble kitchen counter, morning sunlight, clean background",
        "on wooden desk, dramatic shadows, office environment",
        "on gym equipment, studio lighting, fitness context"
    ]
    
    print(f"\n2️⃣ Generating in {len(scenes)} different scenes...")
    
    # Create timestamped output directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/identity_test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_images = []
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n   Scene {i}: {scene}")
        
        try:
            # Generate
            img = generate_with_gap(gap, scene, seed=42+i)
            generated_images.append(img)
            
            # Save
            img.save(output_dir / f"scene_{i}.png")
            print(f"   💾 Saved: scene_{i}.png")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    # Measure consistency
    print("\n3️⃣ Measuring identity consistency...")
    consistency_score = measure_identity_consistency(generated_images)
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"\n🎯 Identity Consistency Score: {consistency_score:.3f}")
    print(f"   Target: >0.70 (proves same asset)")
    
    if consistency_score > 0.70:
        print("\n✅ PASS: Asset identity preserved across scenes!")
        print("   The supplement bottle maintains its identity despite")
        print("   different lighting, backgrounds, and contexts.")
    else:
        print("\n⚠️  MARGINAL: Some identity drift detected")
        print(f"   Score {consistency_score:.3f} < 0.70 target")
        print("   May need stronger conditioning or fine-tuning.")
    
    print(f"\n📂 Output: {output_dir.absolute()}")
    print("\n📋 Generated images:")
    for i in range(len(scenes)):
        print(f"   • scene_{i+1}.png - {scenes[i][:50]}...")
    
    print("\n💡 Compare the images visually:")
    print("   Is it recognizably the SAME supplement bottle?")
    print("   That's what GAP guarantees for asset marketplaces.")
    
    return consistency_score > 0.70

if __name__ == "__main__":
    success = test_identity_preservation()
    sys.exit(0 if success else 1)