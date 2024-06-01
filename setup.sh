# 1. First deactivate and remove the conda environment
conda deactivate
conda remove -n gdino --all -y

# 2. Create fresh conda environment with just Python 3.10
conda create -n gdino python=3.10 -y
conda activate gdino

# 3. Remove any existing PyTorch installations to be safe
pip uninstall torch torchvision -y

# 4. Install PyTorch and torchvision with CUDA 11.8 using pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Verify CUDA is available and correct version
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 6. Clean any previous Grounding DINO installation
#cd /path/to/Adaptive-Vision-Language-Detector  # Replace with your actual path
rm -rf build/
rm -rf *.egg-info
rm -rf dist/
find . -name "*.so" -delete

# 7. Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 8. Now try installing Grounding DINO
pip install -e .
