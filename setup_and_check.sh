#!/bin/bash
set -e

echo "🚀 GAP Project Setup & Check"
echo "=============================="
echo ""

# Navigate to GAP
cd /workspaces/GAP

echo "📂 Current directory: $(pwd)"
echo ""

# Check GPU
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo "❌ No NVIDIA GPU found"
    GPU_AVAILABLE=false
fi

echo ""

# Check Python & PyTorch
echo "🐍 Checking Python environment..."
python3 --version

if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch installed"
    python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
else
    echo "⚠️  PyTorch not installed yet (will install in Step 1)"
fi

echo ""

# Check Blender
echo "🎨 Checking Blender..."
if command -v blender &> /dev/null; then
    echo "✅ Blender installed:"
    blender --version | head -n 1
else
    echo "⚠️  Blender not installed yet (will install in Step 1)"
fi

echo ""

# Check directory structure
echo "📁 Checking project structure..."
for dir in assets tools diffusion tests blender_scripts output docs; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir/"
    else
        echo "   ⚠️  $dir/ (will create)"
    fi
done

echo ""
echo "📊 Recommendation:"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "   ✅ Use local GPU for diffusion testing (fastest, free)"
else
    echo "   💡 No GPU detected. Options:"
    echo "      1. Use CPU (slower but works)"
    echo "      2. Use HuggingFace API (~$0.01 per test)"
    echo "      3. Use Replicate API (~$0.02 per test)"
fi

echo ""
echo "✅ Ready to proceed with Step 1: Install Dependencies"
echo "   Run: pip3 install -r requirements.txt"
