"""
Photo-to-GAP Pipeline
Converts photos into GAP packages with identity encoding
"""
import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
from transformers import pipeline, CLIPProcessor, CLIPModel
from rembg import remove

class PhotoToGAP:
    def __init__(self):
        print("Loading models...")
        
        # Depth estimation (Depth-Anything-V2)
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Identity encoding (CLIP)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if torch.cuda.is_available():
            self.clip_model = self.clip_model.cuda()
        
        print("✅ Models loaded")
    
    def estimate_depth(self, image):
        """Estimate depth map from image"""
        result = self.depth_estimator(image)
        depth = result["depth"]
        
        # Normalize to 0-255 for visualization
        depth_array = np.array(depth)
        depth_normalized = ((depth_array - depth_array.min()) / 
                           (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        
        return Image.fromarray(depth_normalized)
    
    def extract_mask(self, image):
        """Remove background, get object mask"""
        # Remove background
        output = remove(image)
        
        # Extract alpha channel as mask
        if output.mode == 'RGBA':
            mask = output.split()[-1]
        else:
            mask = Image.new('L', image.size, 255)
        
        return mask, output
    
    def encode_identity(self, image, depth, mask):
        """
        Create unique identity embedding for this asset
        Combines visual + geometric features
        """
        # CLIP visual features
        inputs = self.clip_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            visual_features = self.clip_model.get_image_features(**inputs)
        
        # Geometric features from depth
        depth_array = np.array(depth).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        # Masked depth statistics (only the object)
        masked_depth = depth_array * mask_array
        valid_depth = masked_depth[mask_array > 0.5]
        
        if len(valid_depth) > 0:
            geometric_features = torch.tensor([
                valid_depth.mean(),
                valid_depth.std(),
                valid_depth.min(),
                valid_depth.max(),
                (mask_array > 0.5).sum() / mask_array.size  # Object size ratio
            ])
        else:
            geometric_features = torch.zeros(5)
        
        # Combined identity token
        identity_embedding = torch.cat([
            visual_features.cpu().flatten(),
            geometric_features
        ])
        
        return identity_embedding
    
    def create_gap_package(self, photo_path, output_dir, object_name=None):
        """
        Create GAP package from photo
        
        Args:
            photo_path: Path to input photo
            output_dir: Where to save GAP package
            object_name: Asset name (default: filename)
        """
        photo_path = Path(photo_path)
        output_dir = Path(output_dir)
        
        if object_name is None:
            object_name = photo_path.stem
        
        print(f"\n{'='*60}")
        print(f"Creating GAP package: {object_name}")
        print(f"{'='*60}")
        
        # Load image
        print("\n1️⃣ Loading image...")
        image = Image.open(photo_path).convert('RGB')
        print(f"   Size: {image.size}")
        
        # Estimate depth
        print("\n2️⃣ Estimating depth...")
        depth = self.estimate_depth(image)
        print("   ✅ Depth map generated")
        
        # Extract mask
        print("\n3️⃣ Extracting object mask...")
        mask, image_no_bg = self.extract_mask(image)
        print("   ✅ Background removed")
        
        # Encode identity
        print("\n4️⃣ Encoding identity features...")
        identity_embedding = self.encode_identity(image, depth, mask)
        print(f"   ✅ Identity embedding: {identity_embedding.shape}")
        
        # Create output structure
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical_dir = output_dir / "canonical"
        geometry_dir = output_dir / "geometry"
        canonical_dir.mkdir(exist_ok=True)
        geometry_dir.mkdir(exist_ok=True)
        
        # Save files
        print("\n5️⃣ Saving GAP package...")
        image.save(canonical_dir / "reference.png")
        image_no_bg.save(canonical_dir / "reference_no_bg.png")
        depth.save(geometry_dir / "depth_000.png")
        mask.save(geometry_dir / "mask_000.png")
        
        # Save identity embedding
        torch.save(identity_embedding, output_dir / "identity_embedding.pt")
        
        # Save metadata
        metadata = {
            "object_name": object_name,
            "version": "1.0",
            "source": str(photo_path),
            "identity_embedding_shape": list(identity_embedding.shape),
            "has_depth": True,
            "has_mask": True,
            "image_size": list(image.size)
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ GAP Package Created!")
        print("="*60)
        print(f"\n📂 Location: {output_dir.absolute()}")
        print("\n📋 Contents:")
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                rel_path = file.relative_to(output_dir)
                print(f"   • {rel_path} ({size_kb:.1f} KB)")
        
        return output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert photo to GAP package")
    parser.add_argument("--input", required=True, help="Input photo path")
    parser.add_argument("--output", required=True, help="Output GAP directory")
    parser.add_argument("--name", help="Object name (default: filename)")
    
    args = parser.parse_args()
    
    converter = PhotoToGAP()
    converter.create_gap_package(args.input, args.output, args.name)