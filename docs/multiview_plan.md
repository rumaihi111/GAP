# GAP Multi-View Fidelity Plan

## 1. Mission

Deliver photo → multi-view → video generation with consistent product fidelity across all camera angles, outperforming single-image diffusion baselines (e.g., Runway Gen-3).

---

## 2. Current Capabilities (Working Today)

- **Photo → GAP package**
  - depth-anything-v2 for depth maps
  - `rembg` for foreground masks
  - CLIP-based 517-dim identity embedding
- **Diffusion generation**
  - SDXL + ControlNet (depth + canny)
  - IP-Adapter for appearance guidance
  - RunPod serverless deployment (48 GB GPU)
- **Identity testing**
  - `tests/test_identity_preservation.py` (3-scene image validation)

---

## 3. Not in Use / Out of Scope

- TripoSR mesh reconstruction (separate repo, unused)
- Any mesh/PBR pipeline
- Midjourney/Runway for generation (only for comparison)

---

## 4. Gaps to Close

1. **Multi-view capture pipeline**
   - Need 8 photos at 45° intervals per product
   - Either manual capture (turntable + phone) or Polycam guidance
2. **Video generation**
   - Integrate AnimateDiff using existing ControlNet + IP-Adapter
   - Interpolate between the 8 GAP views to produce 24-frame videos
3. **Fidelity benchmarking**
   - LPIPS-based metric comparing generated frames to captured references
   - Runway baseline for comparative demo

---

## 5. Execution Plan

### Phase 1 – Multi-view GAP Packages (Week 1) – DONE
- Capture 8-angle photo set (`supplement_{000..315}.jpg`) ✅
- Run `tools/multiview/build_gap_multiview.py` to package all 8 angles ✅
- Build manifest at `output/gaps/supplement_mv/multiview_manifest.json` ✅

### Phase 2 – Video Generation (Week 2)
- RunPod serverless video endpoint created: `gap-animatediff-video-sd15`
  - GPU: 80 GB (RTX 6000/A100-class), container disk: 50 GB
  - Intention: SD 1.5 + ControlNet depth + AnimateDiff motion module
- Actions next:
  - [ ] Add serverless handler: `diffusion/runpod_animatediff_handler.py`
  - [ ] Point endpoint image to include AnimateDiff repo + motion module at `/app/models/Motion_Module/mm_sd_v15_v2.ckpt`
  - [ ] Redirect caches in container: `HF_HOME=/cache/hf`, `TORCH_HOME=/cache/torch`
  - [ ] Local fallback script ready: `diffusion/generate_multiview_video.py` (SD 1.5 path)
  - [ ] Submit manifest from client and save returned GIF/MP4

### Phase 3 – Validation & Demo (Week 3)
- Generate Runway comparison video
- Add test: `tests/test_video_fidelity.py`
  - Compute LPIPS per angle (ground truth vs generated)
  - Report average fidelity improvement
- Assemble demo reel + metrics table

---

## 6. Tooling Reference

| Task | Tool/Model | Notes |
|------|------------|-------|
| Depth | depth-anything-v2 | Already in pipeline |
| Masking | rembg | Already in pipeline |
| Appearance guidance | IP-Adapter / IP-Adapter-Plus | Handler update needed |
| Video interpolation | AnimateDiff | New dependency |
| Fidelity metric | LPIPS | `pip install lpips` |
| Baseline comparison | Runway Gen-3 | External |

---

## 7. Checklist

- [x] Capture multi-view photo set
- [x] Generate GAP packages + manifest for each angle
- [ ] Install AnimateDiff + motion module (serverless image)
- [x] Implement `generate_multiview_video.py` (local fallback; SD 1.5)
- [ ] Add serverless handler for video: `diffusion/runpod_animatediff_handler.py`
- [ ] Produce GAP multi-view video (serverless)
- [ ] Generate Runway comparison video
- [ ] Implement LPIPS validation test
- [ ] Document results + share demo

---

## 8. Communication

- Keep this document as the authoritative reference.
- Update checklist and add findings directly in this file during execution.
- Use `%BROWSER` command if external documentation needs to be opened.