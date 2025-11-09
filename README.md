# GAP — Generative Asset Package

**Derivative project from TripoSR - focused on proving diffusion conditioning preserves 3D asset identity**

## Quick Context

This project validates that:
1. Structured 3D assets (USD/GLB) can be packaged as GAPs
2. GAP conditioning (depth/normal/mask/reference) preserves geometry + texture
3. Across 360° diffusion-generated orbits

## Success Criteria
- Silhouette IoU ≥ 0.80 average
- Texture SSIM ≥ 0.65 median  
- Visual consistency across orbit

## Development Approach
**Diffusion-first:** Validate SDXL + ControlNet pipeline before building GAP tooling

## Stack
- SDXL + Multi-ControlNet (huggingface/diffusers)
- IP-Adapter (tencent-ailab)
- Blender Python API (G-buffers)
- Trimesh (3D processing)
- LPIPS + CLIP (metrics)

## Structure
```
GAP/
├── assets/       # Test GLB/USD files
├── tools/        # GAP creator, evaluator
├── diffusion/    # SDXL + ControlNet wrapper
├── docker/       # RunPod endpoint
├── tests/        # Demo scripts
└── output/       # Generated results
```

## Getting Started

```bash
cd /workspaces/GAP

# Install dependencies
python tests/install_dependencies.py

# Download test assets
python tests/download_assets.py

# Run demo
python tests/demo_fasttrack.py
```

See `docs/CONTEXT.md` for full conversation history.
