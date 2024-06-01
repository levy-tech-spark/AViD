# 1. First check current GCC version
gcc --version

# 2. Install GCC 9
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-9 g++-9

# 3. Set GCC 9 as default for the compilation
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
                        --slave /usr/bin/gcov gcov /usr/bin/gcov-9

# 4. Verify GCC version
gcc --version
g++ --version

# 5. Clean previous build attempts
cd /path/to/Adaptive-Vision-Language-Detector  # Replace with your actual path
rm -rf build/
rm -rf *.egg-info
rm -rf dist/
find . -name "*.so" -delete

# 6. Set environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

# 7. Try installing again
pip install -e .
