# Project Context

## Origin
Split from TripoSR project to focus on GAP validation.

## Key Architectural Decisions

### 1. Diffusion-First Development
- Validate SDXL + Multi-ControlNet works before building GAP tools
- Use dummy priors initially (synthetic depth/normal maps)
- Incrementally add real GAP conditioning

### 2. Fast-Track Using Existing Projects
- **diffusers**: Native multi-ControlNet support
- **IP-Adapter**: Pre-built appearance preservation
- **Blender**: Headless G-buffer rendering
- **Trimesh**: Universal 3D format loader

### 3. Evaluation Strategy
- IoU for geometry preservation
- SSIM for texture consistency
- CLIP for semantic similarity
- Ablation: depth-only → +normal → +IP-Adapter

## Timeline
- Phase 1: Diffusion setup (today)
- Phase 2: GAP creator (next)
- Phase 3: Full integration
- Phase 4: Evaluation

## Assets
Using CC0 models from Poly Haven and Kenney for testing.
