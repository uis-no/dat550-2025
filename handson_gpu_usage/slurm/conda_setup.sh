#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=1:00:00
#SBATCH --job-name=conda_setup
#SBATCH --output=conda_setup.out

uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39
conda create -n transformer_cuda12 -c pytorch pytorch torchvision torchaudio pytorch-cuda=12.1 -c nvidia -y
conda activate transformer_cuda12
pip3 install torch torchvision torchaudio
pip3 install transformers[torch]
pip3 install -r requirements.txt
