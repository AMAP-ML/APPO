#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/_appo_common.sh"

# conda activate appo

PROJECT_NAME="reasoning_tasks"
CONFIG_NAME="ppo_trainer.yaml"
NNODES=1
N_GPUS_PER_NODE=8

PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=96
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=6144
TRAIN_FILES="${APPO_DIR}/rl_datasets/train_10k.parquet"
VALID_FILES="${APPO_DIR}/rl_datasets/valid.parquet"

ACTOR_MODEL_PATH="<your_sft_model_path>"
LASTING="${ACTOR_MODEL_PATH##*/}"
EXPERIMENT_NAME="APPO_7B_reasoning_${LASTING}"

ROLLOUT_N=16
INITIAL_ROLLOUTS=8
GPU_MEMORY_UTILIZATION=0.5
TOTAL_EPOCHS=2
SAVE_FREQ=20
TEST_FREQ=1
SAVE_PATH="<your_checkpoint_save_dir>/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_DIR="${SAVE_PATH}"
fi

appo_prepare_dirs
appo_run_training
