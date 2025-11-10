"""
Step 2: Download Test Assets
Downloads test meshes (GLB format)
"""
import urllib.request
from pathlib import Path
from tqdm import tqdm
import sys

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    assets_dir = Path("assets/test")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("📦 Downloading Test Assets (GLB Format)")
    print("=" * 60)
    print()
    
    # Use Khronos glTF Sample Models (guaranteed stable URLs, CC0/public domain)
    assets = [
        {
            "name": "BoxTextured.glb",
            "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BoxTextured/glTF-Binary/BoxTextured.glb",
            "desc": "Textured cube (simple test case)"
        },
        {
            "name": "Avocado.glb",
            "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Avocado/glTF-Binary/Avocado.glb",
            "desc": "Avocado (organic shape with texture)"
        },
        {
            "name": "DamagedHelmet.glb",
            "url": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
            "desc": "Damaged helmet (PBR materials)"
        }
    ]
    
    print("ℹ️  Source: Khronos glTF Sample Models")
    print("   License: CC0 / Public Domain")
    print()
    
    success_count = 0
    for asset in assets:
        output_path = assets_dir / asset["name"]
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ {asset['name']} already exists ({size_mb:.2f} MB)")
            success_count += 1
            continue
        
        print(f"📥 Downloading {asset['desc']}...")
        try:
            download_url(asset["url"], output_path)
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Saved: {output_path.name} ({size_mb:.2f} MB)")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print()
    print("=" * 60)
    
    if success_count == 0:
        print("❌ No assets downloaded")
        print()
        print("💡 Manual download option:")
        print("   1. Visit: https://github.com/KhronosGroup/glTF-Sample-Models")
        print("   2. Download any .glb file")
        print(f"   3. Save to: {assets_dir.absolute()}")
        return False
    
    print(f"✅ Downloaded {success_count}/{len(assets)} assets")
    print()
    print(f"📂 Location: {assets_dir.absolute()}")
    print()
    print("📋 Available meshes:")
    for file in sorted(assets_dir.glob("*.glb")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   • {file.name} ({size_mb:.2f} MB)")
    
    print()
    print("=" * 60)
    print("✅ Step 2 Complete!")
    print()
    print("Next: Step 3 - Test Diffusion Pipeline")
    print("   (Uses dummy priors, not these meshes yet)")
    print()
    print("These meshes will be used in:")
    print("   • Step 4: GAP Creator (GLB → USD + priors)")
    print("   • Step 5: Full Orbit Generation")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
