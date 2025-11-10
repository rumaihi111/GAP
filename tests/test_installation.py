# tests/test_installation.py
"""
Step 1: Test Installation
Verifies all dependencies are correctly installed
"""
import sys
from pathlib import Path

def test_python_version():
    """Python 3.10+ required"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    assert version >= (3, 10), "Python 3.10+ required"
    print("   ✅ Version OK")

def test_pytorch():
    """PyTorch with CUDA support"""
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
            print("   ✅ GPU ready")
        else:
            print("   ⚠️  CPU only (fine for testing, RunPod will use GPU)")
        print("   ✅ PyTorch OK")
    except ImportError as e:
        print(f"   ❌ PyTorch not installed: {e}")
        sys.exit(1)

def test_diffusers():
    """Diffusers library"""
    try:
        import diffusers
        print(f"🎨 Diffusers: {diffusers.__version__}")
        # Check if we can import key classes
        from diffusers import StableDiffusionXLPipeline, ControlNetModel
        print("   ✅ Diffusers OK")
    except ImportError as e:
        print(f"   ❌ Diffusers not installed: {e}")
        sys.exit(1)

def test_transformers():
    """Transformers for IP-Adapter"""
    try:
        import transformers
        print(f"🤗 Transformers: {transformers.__version__}")
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        print("   ✅ Transformers OK")
    except ImportError as e:
        print(f"   ❌ Transformers not installed: {e}")
        sys.exit(1)

def test_trimesh():
    """Trimesh for mesh loading"""
    try:
        import trimesh
        print(f"📐 Trimesh: {trimesh.__version__}")
        print("   ✅ Trimesh OK")
    except ImportError as e:
        print(f"   ❌ Trimesh not installed: {e}")
        sys.exit(1)

def test_openexr():
    """OpenEXR for depth maps"""
    try:
        import OpenEXR
        import Imath
        print(f"📊 OpenEXR: Installed")
        print("   ✅ OpenEXR OK")
    except ImportError as e:
        print(f"   ❌ OpenEXR not installed: {e}")
        print("   Install with: pip install OpenEXR")
        return False
    return True

def test_opencv():
    """OpenCV for image processing"""
    try:
        import cv2
        print(f"👁️  OpenCV: {cv2.__version__}")
        print("   ✅ OpenCV OK")
    except ImportError as e:
        print(f"   ❌ OpenCV not installed: {e}")
        print("   Install with: pip install opencv-python")
        return False
    return True

def test_blender():
    """Blender for rendering"""
    import subprocess
    try:
        result = subprocess.run(
            ["blender", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_line = result.stdout.split('\n')[0]
        print(f"🎭 Blender: {version_line}")
        print("   ✅ Blender OK")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"   ❌ Blender not found: {e}")
        print("   Install with: sudo apt-get install -y blender")
        return False
    return True

def test_other_deps():
    """Other required dependencies"""
    deps = [
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("requests", "requests"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for module_name, package_name in deps:
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} not installed")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 GAP Installation Test")
    print("=" * 60)
    print()
    
    # Track failures
    critical_failures = []
    optional_failures = []
    
    # Test each component
    try:
        test_python_version()
    except AssertionError as e:
        critical_failures.append(f"Python version: {e}")
    
    test_pytorch()
    test_diffusers()
    test_transformers()
    test_trimesh()
    
    if not test_openexr():
        critical_failures.append("OpenEXR missing")
    
    if not test_opencv():
        critical_failures.append("OpenCV missing")
    
    if not test_blender():
        critical_failures.append("Blender missing")
    
    print()
    print("📦 Other dependencies:")
    if not test_other_deps():
        critical_failures.append("Some Python packages missing")
    
    print()
    print("=" * 60)
    
    if critical_failures:
        print("❌ INSTALLATION INCOMPLETE")
        print()
        print("Missing critical dependencies:")
        for failure in critical_failures:
            print(f"   • {failure}")
        print()
        print("Run: pip install -r requirements.txt")
        print("Run: sudo apt-get install -y blender")
        sys.exit(1)
    else:
        print("✅ ALL DEPENDENCIES INSTALLED")
        print()
        print("Ready to proceed with Step 2: Download Test Assets")
        sys.exit(0)