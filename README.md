# GAP — Generative Asset Package

**Goal:** Prove that structured 3D assets (USD/GLB) can be packaged as GAPs to preserve geometry + texture identity across diffusion-generated multi-view frames.

## Project Structure

```
GAP/
├── assets/          # Input 3D assets (GLB/USD)
├── tools/           # GAP creation & evaluation
├── diffusion/       # SDXL + ControlNet pipeline
├── docker/          # RunPod serverless endpoint
├── tests/           # Demo scripts
└── output/          # Generated GAPs, renders, metrics
```

## Quick Start

```bash
# 1. Install dependencies
cd /workspaces/GAP
python tests/install_dependencies.py

# 2. Test installation
python tests/test_installation.py

# 3. Download test assets
python tests/download_assets.py

# 4. Run fast-track demo
python tests/demo_fasttrack.py
```

## Success Criteria

- ✅ Silhouette IoU ≥ 0.80 average
- ✅ Texture SSIM ≥ 0.65 median
- ✅ Visual consistency across 360° orbit

