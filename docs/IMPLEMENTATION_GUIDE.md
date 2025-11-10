# GAP Implementation Guide

## 🎯 GAP Package Specification

### Required Files

Every GAP package must contain:

```
gap_name/
├── metadata.json          # Conditioning specs (see below)
├── subject.usd           # Original high-poly mesh
├── mesh_low.usd          # Decimated mesh for rendering
├── albedo.png            # Base texture (optional)
├── canonical/            # Reference view (0° azimuth)
│   ├── rgb.png          # Textured render
│   ├── depth.exr        # 16-bit float, camera space
│   ├── normal.png       # View-space normals [0-1]
│   ├── mask.png         # Binary mask (no antialiasing)
│   └── ref_embed.bin    # IP-Adapter CLIP embedding
└── multiview/           # 4-view conditioning
    ├── 000/             # 0° (front)
    ├── 090/             # 90° (right)
    ├── 180/             # 180° (back)
    └── 270/             # 270° (left)
        ├── rgb.png
        ├── depth.exr
        ├── normal.png
        ├── mask.png
        └── ref_embed.bin
```

### metadata.json Schema

```json
{
  "asset_id": "chair_001",
  "source_file": "office_chair_01_1k.glb",
  "created": "2024-01-15T10:30:00Z",
  "units": "meters",
  "axes": {
    "up": "Y",
    "front": "Z"
  },
  "bbox": [0.52, 0.94, 0.49],
  "views": ["000", "090", "180", "270"],
  "conditioning": {
    "depth_space": "camera",
    "depth_format": "exr16",
    "depth_range": [0.5, 3.0],
    "normal_space": "view",
    "mask_mode": "binary"
  },
  "render_config": {
    "resolution": 1024,
    "samples": 128,
    "camera_distance": 2.0,
    "fov": 50.0
  }
}
```

### Validate Your GAP

```bash
python tools/validate_gap.py output/gaps/chair_001
```

---

## Quick Start Commands

### 1. Install Dependencies (5 minutes)

```bash
cd /workspaces/GAP

# System packages (headless Blender if needed)
sudo apt-get update
sudo apt-get install -y blender xvfb

# Python packages
pip install -r requirements.txt

# Verify
python tests/test_installation.py
```

### 2. Download Test Assets (2 minutes)

```bash
python tests/download_assets.py

# Should download:
# - office_chair_01_1k.glb (~5 MB)
# - retro_tv_01_1k.glb (~3 MB)
# - reference.png - Simple RGB reference for IP-Adapter test
```

### 3. Test Diffusion Pipeline (15 minutes)

```bash
# Test with dummy priors + embeddings + IP-Adapter
python tests/test_diffusion_dummy.py --with-ip-adapter

# Expected output:
# - Generated 8 test frames
# - Cost: ~$0.01-0.03 (depends on RunPod pricing)
# - Validates RunPod endpoint works
# - Validates IP-Adapter embedding works
```

**What this tests:**
- ✅ SDXL base model loads
- ✅ Multi-ControlNet (depth + normal) works
- ✅ IP-Adapter receives embeddings correctly
- ✅ RunPod endpoint responds with valid images
- ✅ Deterministic seeds produce consistent results

**Engine Configuration (defaults that work well):**
```python
num_inference_steps = 40
controlnet_conditioning_scale = [0.9, 0.8]  # depth, normal
ip_adapter_scale = 0.8
guidance_scale = 4.5
control_guidance_start = 0.0  # Full control across ALL denoise steps
control_guidance_end = 1.0    # No tapering off
```

**Memory Optimizations:**
```python
pipe.to(torch.float16)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
```

### 4. Build GAP Creator

```bash
# Create GAP from test asset with multiview support
python tools/gap_creator.py \
    --input assets/office_chair_01_1k.glb \
    --output output/gaps/chair_001 \
    --views 000,090,180,270 \
    --res 1024

# Output structure (revised):
# chair_001/
#   subject.usd
#   mesh_low.usd
#   albedo.png
#   metadata.json                # Complete specification
#   canonical/
#     rgb.png
#     depth.exr                  # 16-bit float, camera space
#     normal.png                 # view-space, unit mapped to [0,1]
#     mask.png                   # binary (0/255), no antialiasing
#     ref_embed.bin              # IP-Adapter/CLIP embedding
#   multiview/
#     000/{rgb.png,depth.exr,normal.png,mask.png,ref_embed.bin}
#     090/{...}
#     180/{...}
#     270/{...}
```

**Key Format Specifications:**
- **Depth**: EXR 16-bit float, camera space, normalized per-frame
- **Normal**: View-space normals, XYZ[-1,1] remapped to RGB[0,1]
- **Mask**: Binary (0/255), no antialiasing, eroded 1px to avoid edge artifacts
- **Embedding**: IP-Adapter CLIP embedding from canonical RGB

### 5. Generate Full Orbit (20 minutes)

```bash
# Use GAP to generate 72-frame orbit with deterministic seeds
python tests/demo_full_orbit.py \
    --gap output/gaps/chair_001 \
    --frames 72 \
    --output output/renders/chair_001_orbit \
    --use_ip_adapter 1 \
    --ref_mode multiview_switch \
    --seed 42

# Outputs:
# - 72 PNG frames (frame_0000.png to frame_0071.png)
# - Compiled MP4 video (orbit.mp4)
# - Side-by-side comparison (comparison.mp4)
```

**Reproducibility:**
GAP uses deterministic seeds for fair comparison:
```bash
# First pass with seed 42
python tests/demo_full_orbit.py --gap output/gaps/chair_001 --seed 42

# Second pass with seed 123 (for robustness check)
python tests/demo_full_orbit.py --gap output/gaps/chair_001 --seed 123
```

This ensures ablation studies are valid:
- Same seeds → only conditioning changes affect output
- Different results = real improvement, not randomness

**Scheduler Options:**
```bash
# Use DPM++ 2M scheduler for cleaner results
python tests/demo_full_orbit.py \
    --gap output/gaps/chair_001 \
    --scheduler dpmpp_2m
```

### 6. Evaluate Quality

```bash
python tools/evaluate.py \
    --generated output/renders/chair_001_orbit \
    --priors output/gaps/chair_001/multiview \
    --canonical output/gaps/chair_001/canonical/rgb.png \
    --metrics mask_iou edge_overlap ssim_patch clip \
    --patch-bbox 420,360,120,120  # x,y,w,h of distinctive texture region
```

**Metrics Computed:**

1. **Mask IoU** (Geometry Silhouette)
   - Compares generated mask vs USD-rendered mask
   - Uses GAP's own mask (not SAM post-segmentation)
   - Target: ≥0.80 average, ≥0.70 minimum

2. **Edge Overlap** (Geometry Detail)
   - Compares Canny edges: generated RGB vs depth gradients
   - Measures fine geometry preservation
   - Target: ≥0.70 average

3. **SSIM Patch** (Texture Consistency)
   - Tracks distinctive texture patch across 72 frames
   - Specify patch bbox with `--patch-bbox x,y,w,h`
   - Target: ≥0.65 median

4. **CLIP Similarity** (Semantic Consistency)
   - Embeddings similarity: generated vs canonical
   - Measures object identity preservation
   - Target: ≥0.85 average

**Output Files:**
- `metrics.csv` - Per-frame scores
- `metrics_summary.json` - Aggregated statistics
- `plots/mask_iou_curve.png` - IoU over orbit
- `plots/edge_overlap_curve.png` - Edge similarity over orbit
- `plots/ssim_patch_curve.png` - Texture consistency over orbit
- `plots/clip_similarity_curve.png` - Semantic identity over orbit

**Important:** Evaluation uses GAP's rendered masks directly (not external segmentation like SAM) to avoid introducing additional noise into metrics.

---

## File Structure Guide

### What Each Directory Contains

```
GAP/
├── assets/              # Your input 3D models
│   ├── test/           # Downloaded test assets
│   │   ├── office_chair_01_1k.glb
│   │   ├── retro_tv_01_1k.glb
│   │   └── reference.png    # RGB reference for IP-Adapter
│   └── custom/         # Add your own models here
│
├── tools/              # Core GAP creation tools
│   ├── gap_creator.py      # Main: GLB → GAP
│   ├── validate_gap.py     # Validate GAP package structure
│   ├── make_embedding.py   # Generate IP-Adapter embeddings
│   ├── evaluate.py         # Metrics computation
│   └── utils.py            # Shared utilities
│
├── blender_scripts/    # Headless rendering
│   ├── render_gbuffer.py   # RGB, depth, normal, mask
│   └── setup_compositor.py # Blender compositor setup
│
├── diffusion/          # SDXL + ControlNet pipeline
│   ├── engine.py           # SDXL wrapper
│   ├── runpod_engine.py    # RunPod serverless backend
│   ├── gap_conditioner.py  # GAP → conditioning
│   └── ip_adapter_wrapper.py
│
├── docker/             # RunPod deployment
│   ├── Dockerfile          # Container definition
│   └── handler.py          # RunPod serverless handler
│
├── tests/              # Demo & validation
│   ├── test_installation.py
│   ├── download_assets.py
│   ├── test_diffusion_dummy.py
│   └── demo_full_orbit.py
│
└── output/             # Generated results
    ├── gaps/               # Created GAP packages
    ├── renders/            # Generated orbits
    └── metrics/            # Evaluation results
```

---

## Common Tasks

### Task 1: Add a New Test Asset

```bash
# 1. Download GLB/USD file to assets/
wget <url> -O assets/my_object.glb

# 2. Create GAP with multiview support
python tools/gap_creator.py \
    --input assets/my_object.glb \
    --output output/gaps/my_object_001 \
    --views 000,090,180,270

# 3. Validate GAP package
python tools/validate_gap.py output/gaps/my_object_001

# 4. Generate orbit
python tests/demo_full_orbit.py \
    --gap output/gaps/my_object_001 \
    --seed 42
```

### Task 2: Regenerate Embeddings

```bash
# If you update canonical RGB, regenerate embedding
python tools/make_embedding.py \
    --input output/gaps/chair_001/canonical/rgb.png \
    --output output/gaps/chair_001/canonical/ref_embed.bin

# Batch regenerate for all views
for view in 000 090 180 270; do
    python tools/make_embedding.py \
        --input output/gaps/chair_001/multiview/$view/rgb.png \
        --output output/gaps/chair_001/multiview/$view/ref_embed.bin
done
```

### Task 2: Adjust Diffusion Parameters

Edit `diffusion/engine.py`:

```python
# Increase conditioning strength
controlnet_conditioning_scale=[0.9, 0.8]  # Was [0.8, 0.6]

# Force full ControlNet scheduling (no tapering)
control_guidance_start=0.0  # Active from first step
control_guidance_end=1.0    # Active until last step

# More inference steps (slower but better quality)
num_inference_steps=50  # Was 40

# Stronger IP-Adapter influence
ip_adapter_scale=0.8  # Was 0.5

# Adjust guidance scale
guidance_scale=4.5  # Lower = more creative, higher = more faithful
```

### Task 3: Change Orbit Resolution

Edit `tests/demo_full_orbit.py`:

```python
# More frames for smoother video
NUM_FRAMES = 120  # Was 72

# Higher resolution
RESOLUTION = 2048  # Was 1024
```

### Task 4: Batch Process Multiple Assets

```bash
# Process all GLBs in assets/custom/
for glb in assets/custom/*.glb; do
    name=$(basename "$glb" .glb)
    echo "Processing $name..."
    
    # Create GAP
    python tools/gap_creator.py \
        --input "$glb" \
        --output "output/gaps/$name" \
        --views 000,090,180,270
    
    # Validate
    python tools/validate_gap.py "output/gaps/$name"
    
    # Generate orbit
    python tests/demo_full_orbit.py \
        --gap "output/gaps/$name" \
        --seed 42
done
```

### Task 5: Run Ablation Study

```bash
# Test different conditioning combinations
# A: Depth only
python tests/demo_full_orbit.py \
    --gap output/gaps/chair_001 \
    --conditioning depth \
    --output output/renders/chair_001_depth_only

# B: Depth + Normal
python tests/demo_full_orbit.py \
    --gap output/gaps/chair_001 \
    --conditioning depth,normal \
    --output output/renders/chair_001_depth_normal

# C: Depth + Normal + IP-Adapter (full GAP)
python tests/demo_full_orbit.py \
    --gap output/gaps/chair_001 \
    --conditioning depth,normal,ip_adapter \
    --output output/renders/chair_001_full

# Compare results
python tools/evaluate.py \
    --compare \
    --runs output/renders/chair_001_depth_only \
           output/renders/chair_001_depth_normal \
           output/renders/chair_001_full
```

---

## Troubleshooting

### Issue: Blender not found

```bash
# Check installation
which blender

# If not found, install (with xvfb for headless)
sudo apt-get update
sudo apt-get install -y blender xvfb

# Verify
blender --version

# For headless rendering
xvfb-run blender --background --python script.py
```

### Issue: GAP validation fails

```bash
# Check package structure
python tools/validate_gap.py output/gaps/chair_001

# Common issues:
# - Missing ref_embed.bin: Run tools/make_embedding.py
# - Wrong depth format: Check metadata.json depth_format = "exr16"
# - Mask has antialiasing: Disable AA in Blender render settings
# - Normal space incorrect: Ensure view-space normals in Blender
```

### Issue: CUDA out of memory

```python
# In diffusion/engine.py, add memory optimizations:
pipe.to(torch.float16)  # Use half precision
pipe.enable_model_cpu_offload()  # Offload to CPU when not in use
pipe.enable_vae_slicing()  # Process VAE in slices
pipe.enable_attention_slicing(1)  # Reduce memory for attention
```

### Issue: Depth normalization incorrect

```python
# Check metadata.json has depth_range recorded
# Verify depth is normalized to [0, 1] in EXR
import OpenEXR, Imath
exr = OpenEXR.InputFile("depth.exr")
# Values should be between 0.0 and 1.0
```

### Issue: Normal map looks wrong

```bash
# Verify normals are in VIEW space (not world space)
# Check metadata.json: "normal_space": "view"
# In Blender: Use "Normal" pass from Render Layers (not custom)
# Verify XYZ[-1,1] remapped to RGB[0,1]: RGB = (XYZ + 1.0) / 2.0
```

### Issue: Models downloading too slow

```python
# Set Hugging Face cache to faster storage
import os
os.environ['HF_HOME'] = '/tmp/huggingface'
```

### Issue: RunPod endpoint timeout

```python
# In handler.py, increase timeout
# Or process fewer frames per request
BATCH_SIZE = 8  # Instead of 72
```

---

## Performance Tips

### Speed Optimizations

1. **Use xformers**
   ```python
   pipe.enable_xformers_memory_efficient_attention()
   ```

2. **Batch processing**
   ```python
   # Generate multiple frames at once
   images = pipe(prompt, image=controls, num_images_per_prompt=4)
   ```

3. **Lower resolution during testing**
   ```python
   RESOLUTION = 512  # Instead of 1024
   ```

### Quality Improvements

1. **More inference steps**
   ```python
   num_inference_steps=50  # Instead of 40
   ```

2. **Stronger conditioning**
   ```python
   controlnet_conditioning_scale=[0.9, 0.8]  # Instead of [0.9, 0.8]
   ```

3. **Full ControlNet scheduling**
   ```python
   control_guidance_start=0.0  # No tapering
   control_guidance_end=1.0
   ```

4. **Better scheduler**
   ```python
   from diffusers import DPMSolverMultistepScheduler
   pipe.scheduler = DPMSolverMultistepScheduler.from_config(
       pipe.scheduler.config
   )
   ```

---

## Data Format Specifications

### Depth Maps

**Format:** EXR 16-bit float
**Color Space:** Camera space
**Range:** [0, 1] (normalized per-frame)
**Storage:** `depth.exr`

**Blender Export:**
```python
# In compositor, use Normalize node
depth_raw = render_layers.outputs['Depth']
normalize = tree.nodes.new('CompositorNodeNormalize')
tree.links.new(depth_raw, normalize.inputs[0])

# Output to EXR 16-bit
output.format.file_format = 'OPEN_EXR'
output.format.color_depth = '16'
```

**Metadata Recording:**
```json
{
  "depth_space": "camera",
  "depth_format": "exr16",
  "depth_range": [0.0, 1.0],
  "depth_min_meters": 0.5,
  "depth_max_meters": 3.0
}
```

### Normal Maps

**Format:** PNG 8-bit RGB
**Color Space:** View space (camera-relative)
**Encoding:** XYZ[-1,1] remapped to RGB[0,1]
**Formula:** `RGB = (XYZ + 1.0) / 2.0`
**Storage:** `normal.png`

**Blender Export:**
```python
# Use built-in Normal pass (already in view space)
scene.view_layers[0].use_pass_normal = True
normal = render_layers.outputs['Normal']

# No additional transforms needed
output.format.file_format = 'PNG'
```

**Metadata:**
```json
{
  "normal_space": "view",
  "normal_format": "PNG_8bit_RGB",
  "normal_encoding": "RGB_01_from_XYZ_neg1_1",
  "normal_convention": "camera_look_negZ_up_Y"
}
```

### Masks

**Format:** PNG 8-bit grayscale
**Encoding:** Binary (0 or 255)
**Antialiasing:** Disabled
**Post-processing:** Eroded 1px to avoid edge artifacts
**Storage:** `mask.png`

**Blender Export:**
```python
# Disable antialiasing
scene.render.filter_size = 0.01

# Use ID Mask node
id_mask = tree.nodes.new('CompositorNodeIDMask')
id_mask.use_antialiasing = False

# Threshold to binary
threshold = tree.nodes.new('CompositorNodeMath')
threshold.operation = 'GREATER_THAN'
threshold.inputs[1].default_value = 0.5
```

**Post-processing:**
```python
import cv2
mask = cv2.imread("mask.png", 0)
mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
cv2.imwrite("mask.png", mask)
```

**Metadata:**
```json
{
  "mask_format": "PNG_8bit_grayscale",
  "mask_encoding": "binary_255_0",
  "mask_antialiasing": false,
  "mask_eroded_px": 1
}
```

### IP-Adapter Embeddings

**Format:** PyTorch tensor (`.bin` file)
**Model:** h94/IP-Adapter CLIP Vision
**Dimensions:** [1, 768] for CLIP ViT-L/14
**Storage:** `ref_embed.bin`

**Generation:**
```python
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
).to("cuda")

processor = CLIPImageProcessor()
image = Image.open("rgb.png")
inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

with torch.no_grad():
    embedding = image_encoder(**inputs).image_embeds

torch.save(embedding.cpu(), "ref_embed.bin")
```

**Metadata:**
```json
{
  "embedding_model": "h94/IP-Adapter",
  "embedding_dim": 768,
  "embedding_dtype": "float16"
}
```

---

## Seed Discipline & Reproducibility

### Per-Frame Deterministic Seeds

For fair ablation studies, use deterministic seeds:

```python
def get_frame_seed(global_seed: int, frame_idx: int) -> int:
    """Generate deterministic per-frame seed"""
    return global_seed + frame_idx * 1000

# Usage
for frame_idx in range(72):
    seed = get_frame_seed(42, frame_idx)  # 42, 1042, 2042, ...
    generate_frame(..., seed=seed)
```

### Two-Pass Validation

```bash
# Pass 1: Global seed 42
python tests/demo_full_orbit.py --gap output/gaps/chair_001 --seed 42

# Pass 2: Global seed 123 (robustness check)
python tests/demo_full_orbit.py --gap output/gaps/chair_001 --seed 123

# Both orbits should have:
# - Same geometry (ControlNet dominates)
# - Same texture consistency (IP-Adapter dominates)
# - Different fine details (seed variation)
```

### Seed Strategy for Ablation

```bash
# Keep seed FIXED across ablation runs
SEED=42

# Test A: Depth only
python tests/demo_full_orbit.py --conditioning depth --seed $SEED

# Test B: Depth + Normal
python tests/demo_full_orbit.py --conditioning depth,normal --seed $SEED

# Test C: Full GAP
python tests/demo_full_orbit.py --conditioning depth,normal,ip_adapter --seed $SEED

# Now differences are from conditioning, not randomness
```

---

## Tool Reference

### validate_gap.py

Validates GAP package structure and metadata:

```bash
python tools/validate_gap.py output/gaps/chair_001
```

**Checks:**
- ✅ All required files present
- ✅ metadata.json schema valid
- ✅ Depth format correct (EXR 16-bit)
- ✅ Normal space correct (view space)
- ✅ Mask format correct (binary, no AA)
- ✅ Embeddings present for all views

### make_embedding.py

Generate IP-Adapter CLIP embeddings from RGB images:

```bash
python tools/make_embedding.py \
    --input output/gaps/chair_001/canonical/rgb.png \
    --output output/gaps/chair_001/canonical/ref_embed.bin \
    --model h94/IP-Adapter
```

**Options:**
- `--input`: Input RGB image (PNG/JPG)
- `--output`: Output embedding file (.bin)
- `--model`: IP-Adapter model (default: h94/IP-Adapter)

### evaluate.py

Compute metrics for generated orbits:

```bash
python tools/evaluate.py \
    --generated output/renders/chair_001_orbit \
    --priors output/gaps/chair_001/multiview \
    --canonical output/gaps/chair_001/canonical/rgb.png \
    --metrics mask_iou edge_overlap ssim_patch clip \
    --patch-bbox 420,360,120,120
```

**Metrics:**
- `mask_iou`: Silhouette IoU (geometry)
- `edge_overlap`: Edge similarity (geometry detail)
- `ssim_patch`: Texture patch consistency
- `clip`: Semantic identity preservation

**Outputs:**
- `metrics.csv`: Per-frame scores
- `metrics_summary.json`: Aggregate statistics
- `plots/*.png`: Visualization curves
