export CUDACXX=/usr/local/cuda/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd submodules/cutlass
rm -rf build
mkdir -p build && cd build
# Auto-detect GPU compute capability or default to 90 (most recent)
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1 | sed 's/\.//g' || echo "90")
cmake .. -DCUTLASS_NVCC_ARCHS="$GPU_ARCH" -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
make -j 16
