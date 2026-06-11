#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/_appo_common.sh"

# conda activate appo

PROJECT_NAME="deep_research"
CONFIG_NAME="ppo_trainer_dr.yaml"
NNODES=1
N_GPUS_PER_NODE=8

PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=2000
MAX_RESPONSE_LENGTH=10000
TRAIN_FILES="${APPO_DIR}/rl_datasets/hard_search_1k.parquet"
VALID_FILES="['${APPO_DIR}/rl_datasets/gaia_test.parquet','${APPO_DIR}/rl_datasets/hle_test.parquet']"

ACTOR_MODEL_PATH="<your_8B_sft_model_path>"
LASTING="${ACTOR_MODEL_PATH##*/}"
EXPERIMENT_NAME="APPO_8B_deepsearch_${LASTING}"

ROLLOUT_N=16
INITIAL_ROLLOUTS=8
GPU_MEMORY_UTILIZATION=0.6
TOTAL_EPOCHS=5
SAVE_FREQ=5
TEST_FREQ=5
SAVE_PATH="<your_checkpoint_save_dir>/rl/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_DIR="${SAVE_PATH}"
fi

appo_prepare_dirs
appo_run_training
