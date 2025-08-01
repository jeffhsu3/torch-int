# torch-int
This repository contains integer operators on GPUs for PyTorch.

## Dependencies
- CUTLASS
- PyTorch with CUDA 12.9
- NVIDIA-Toolkit 12.9
- CUDA Driver 12.9
- gcc g++ 9.4.0 or newer
- cmake >= 3.12

## Installation
```bash
git clone --recurse-submodules https://github.com/Guangxuan-Xiao/torch-int.git
conda create -n int python=3.8
conda activate int
conda install -c anaconda gxx_linux-64=9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
source environment.sh
bash build_cutlass.sh
python setup.py install
```

## Test
```bash
python tests/test_linear_modules.py
```
