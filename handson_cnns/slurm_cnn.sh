#!/bin/bash
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --mem=100G
#SBATCH --job-name=cnn_object_detection
#SBATCH --output=cnn_object_detection.out

# Activate environment
uenv verbose cuda-12.4.0 cudnn-12.x-8.8.0
uenv miniconda3-py311
# module avail cudnn
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/_cuda/cudnn-12.x-8.9.2/lib:$LD_LIBRARY_PATH

python -c "import torch; import torchvision; print(torch.__version__, torchvision.__version__)"
CUDA_LAUNCH_BLOCKING=1 python -u cnn_train_object_detection.py