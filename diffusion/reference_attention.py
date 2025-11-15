#!/usr/bin/env python3
"""
Custom Reference-Guided Attention Module
Injects reference image features into diffusion UNet attention layers
to preserve visual identity during generation.

Fork-friendly implementation you can modify and extend.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from PIL import Image


class ReferenceEncoder:
    """Encodes reference images into feature maps for attention injection."""
    
    def __init__(self, pipe, device="cuda"):
        self.pipe = pipe
        self.device = device
        self.ref_features = {}
        
    def encode_reference_images(self, ref_images: List[Image.Image]) -> dict:
        """
        Extract VAE features from reference images.
        These features will be injected into attention layers.
        """
        ref_tensors = []
        for img in ref_images:
            # Preprocess image
            img = img.resize((512, 512))
            tensor = self.pipe.feature_extractor(img, return_tensors="pt").pixel_values
            ref_tensors.append(tensor.to(self.device, dtype=self.pipe.dtype))
        
        # Stack and encode
        ref_batch = torch.cat(ref_tensors, dim=0)
        with torch.no_grad():
            # Get VAE latent features (these preserve visual details)
            ref_latents = self.pipe.vae.encode(ref_batch).latent_dist.sample()
            ref_latents = ref_latents * self.pipe.vae.config.scaling_factor
        
        return {"latents": ref_latents}
    
    def clear_cache(self):
        """Clear stored reference features."""
        self.ref_features.clear()


class ReferenceAttentionInjector:
    """
    Injects reference features into UNet attention layers.
    You can modify the attention mechanism here to control
    how reference identity is preserved.
    """
    
    def __init__(self, unet, ref_encoder: ReferenceEncoder):
        self.unet = unet
        self.ref_encoder = ref_encoder
        self.original_forward = {}
        self.enabled = False
        
    def enable(self, ref_images: List[Image.Image], strength: float = 0.8):
        """
        Enable reference-guided generation.
        
        Args:
            ref_images: List of reference images to preserve identity from
            strength: How strongly to preserve reference (0.0-1.0)
                     Higher = more faithful to reference
                     Lower = more creative freedom
        """
        # Encode reference images
        self.ref_features = self.ref_encoder.encode_reference_images(ref_images)
        self.ref_strength = strength
        
        # Hook into UNet attention layers
        self._inject_attention_hooks()
        self.enabled = True
        
    def disable(self):
        """Disable reference guidance and restore original UNet."""
        if self.enabled:
            self._remove_attention_hooks()
            self.ref_encoder.clear_cache()
            self.enabled = False
    
    def _inject_attention_hooks(self):
        """
        Inject reference features into attention layers.
        
        MODIFY THIS METHOD to change how reference identity is preserved.
        Current approach: Blend reference latents with generated latents
        using attention-based weighting.
        """
        def make_ref_attention_forward(original_forward, ref_latents, strength):
            """Create modified forward pass that incorporates reference."""
            def ref_forward(hidden_states, encoder_hidden_states=None, *args, **kwargs):
                # Original attention computation
                attn_output = original_forward(hidden_states, encoder_hidden_states, *args, **kwargs)
                
                # Blend with reference features
                # TODO: You can modify this blending strategy
                if ref_latents is not None and attn_output.shape[2:] == ref_latents.shape[2:]:
                    # Simple weighted blend (you can replace with learned attention)
                    ref_contribution = ref_latents.mean(dim=0, keepdim=True)
                    ref_contribution = torch.nn.functional.interpolate(
                        ref_contribution, 
                        size=attn_output.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    attn_output = (1 - strength) * attn_output + strength * ref_contribution
                
                return attn_output
            
            return ref_forward
        
        # Hook mid-block and up-blocks (where visual details are refined)
        ref_latents = self.ref_features.get("latents")
        
        for name, module in self.unet.named_modules():
            if "attn" in name and hasattr(module, "forward"):
                if name not in self.original_forward:
                    self.original_forward[name] = module.forward
                    module.forward = make_ref_attention_forward(
                        module.forward, 
                        ref_latents,
                        self.ref_strength
                    )
    
    def _remove_attention_hooks(self):
        """Restore original UNet attention."""
        for name, module in self.unet.named_modules():
            if name in self.original_forward:
                module.forward = self.original_forward[name]
        self.original_forward.clear()


def add_reference_guidance(pipe, ref_images: List[Image.Image], strength: float = 0.8):
    """
    Convenience function to add reference guidance to a pipeline.
    
    Usage in your handler:
        from diffusion.reference_attention import add_reference_guidance
        
        # Before generation:
        injector = add_reference_guidance(pipe, reference_images, strength=0.8)
        
        # Generate with reference guidance
        result = pipe(prompt=..., ...)
        
        # Clean up
        injector.disable()
    
    Args:
        pipe: AnimateDiff or SD pipeline
        ref_images: List of PIL images to use as reference
        strength: Reference preservation strength (0.0-1.0)
    
    Returns:
        ReferenceAttentionInjector instance (call .disable() when done)
    """
    encoder = ReferenceEncoder(pipe, device=pipe.device)
    injector = ReferenceAttentionInjector(pipe.unet, encoder)
    injector.enable(ref_images, strength=strength)
    return injector
