#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --job-name=imdb_train_sample
#SBATCH --output=imdb_train_sample.out

# Activate environment
uenv verbose cuda-12.4.0 cudnn-12.x-8.8.0
uenv miniconda3-py311
pip3 install torch torchvision torchaudio
pip3 install transformers[torch]
pip3 install -r requirements.txt

CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false python -u bert_train.py
