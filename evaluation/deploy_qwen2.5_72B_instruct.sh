#!/bin/bash
source /home/wangxucong.wxc/miniconda3/envs/arpo/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve /data/Agent/models/Qwen3-14B \
  --served-model-name Qwen2.5-72B-Instruct \
  --max-model-len 32768 \
  --tensor_parallel_size 4 \
  --gpu-memory-utilization 0.75 \
  --dtype auto \
  --port 8089