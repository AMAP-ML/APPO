#!/bin/bash


unset PYTHONHOME
unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_SHLVL
# export PATH="/home/wangxucong.wxc/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="HF_TOKEN_PLACEHOLDER"
export UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
# conda activate SkillRL # 整体的环境 
source /home/wangxucong.wxc/miniconda3/envs/arpo/bin/activate


#================== Basic Configuration ==================#
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # List of visible GPUs
export PYTHONPATH=$(pwd):$PYTHONPATH

# Disable Weights & Biases
export WANDB_DISABLED=false

# 将 HuggingFace 缓存重定向到 NAS，避免写满 /home/huangtongwen.htw（JuiceFS 200G 已满）
mkdir -p "/data/HF_TOKEN_PLACEHOLDER/datasets"
mkdir -p "/data/HF_TOKEN_PLACEHOLDER/transformers"
export HF_HOME="/data/HF_TOKEN_PLACEHOLDER"
export HF_DATASETS_CACHE="/data/HF_TOKEN_PLACEHOLDER/datasets"
export TRANSFORMERS_CACHE="/data/HF_TOKEN_PLACEHOLDER/transformers
"

#================== Training Parameter Configuration ==================#
# Distributed training configuration
NNODES=1                 # Total number of nodes
NODE_RANK=0              # Rank of the current node
PROC_PER_NODE=8          # Number of processes per node
MASTER_ADDR="127.0.0.1"  # Address of the master node
MASTER_PORT=29500        # Port of the master node

# Output directory
OUTPUT_DIR="/data/Agent/logs"
# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Path to the training script
TRAIN_SCRIPT="/home/wangxucong.wxc/AERPO/LLaMA-Factory/src/llamafactory/launcher.py"
# mkdir -p "/data/LLaMA-Factory/src/llamafactory"

# Path to the training argument configuration file
TRAIN_ARGS="/home/wangxucong.wxc/AERPO/LLaMA-Factory/arpo_train_sft/yaml/qwen.yaml" 
stat=$(date +%s)
which python
which conda
which pip

# pip install ./LLaMA-Factory
# # pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# pip install "llvmlite==0.44.0"
# pip install "deepspeed==0.16.9"
# pip install "packaging==25.0"  
# pip install "vllm==0.8.5"
# pip install -r LLaMA-Factory/requirements.txt
# pip check | sed -n 's/.*requires \([^, ]*\).*/\1/p' | sort -u  | xargs -r pip install
# pip install "lark==1.2.2"
# pip install "llguidance==0.7.30" 
# pip install "numpy==1.26.4"
# pip install LLaMA-Factory/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# pip install wandb
 
export WANDB_API_KEY="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"
# Command to launch training
# setuptools==80.10.2
export WANDB_MODE=offline
export WANDB_DIR=/data/wandb   # 你想存放的路径
mkdir -p "$WANDB_DIR"

torchrun --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node ${PROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${TRAIN_SCRIPT} \
         ${TRAIN_ARGS} 2>&1 | tee ${OUTPUT_DIR}/training.log

# Optionally enable logging redirection
# exec >> ${OUTPUT_DIR}/training.log 2>&1
