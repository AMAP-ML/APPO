#!/bin/bash

use_qwen3=true 
# Activate the Conda environment
source /home/wangxucong.wxc/miniconda3/envs/arpo/bin/activate

# Move to the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "cd $SCRIPT_DIR"

# Create log directory
mkdir -p logs

# Model path - same model used for all instances
MODEL_PATH="/data/Agent/models/Qwen3-8B-AEPO-DeepSearch"
MODEL_NAME="Qwen2.5-7B-Instruct"

rm /home/wangxucong.wxc/log/model4.log
rm /home/wangxucong.wxc/log/model3.log
# Launch instance 3 - using GPU 4 and 5
echo "Starting Instance 3 on GPU 4,5"
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8002 > /home/wangxucong.wxc/log/model3.log 2>&1 &
INSTANCE3_PID=$!
echo "Instance 3 deployed on port 8002 using GPU 4,5"

# Launch instance 4 - using GPU 6 and 7
echo "Starting Instance 4 on GPU 6,7"
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8003 > /home/wangxucong.wxc/log/model4.log 2>&1 &
INSTANCE4_PID=$!
echo "Instance 4 deployed on port 8003 using GPU 6,7"

# Display all running model services
echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

# Handle cleanup on termination
trap "kill $INSTANCE3_PID $INSTANCE4_PID" SIGTERM
wait $INSTANCE3_PID $INSTANCE4_PID
