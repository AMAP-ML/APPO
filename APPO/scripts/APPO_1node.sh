SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APPO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$APPO_DIR")"
cd "$APPO_DIR"
echo "Switched to APPO directory: $APPO_DIR"

source "${CONDA_ENV_PATH:-/home/wangxucong.wxc/miniconda3/envs/arpo/bin/activate}"
# ============================ Environment Setup ============================
export CUDA_VISIBLE_DEVICES=2,4,5,6
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=WARNING
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
mkdir -p ${RAY_TMPDIR}

# APPO 使用自己的 verl_arpo_entropy（已替换为 APPO procedural branching 逻辑）
# cognitive_pipeline 需要在 PYTHONPATH 中，供 vllm_rollout_with_tools.py 导入
# 注意：APPO 的 verl_arpo_entropy 必须排在最前面，避免被 conda 环境里 pip 安装的 verl 覆盖
export PYTHONPATH="${APPO_DIR}/verl_arpo_entropy:${REPO_ROOT}:$PYTHONPATH"

# ============================ Basic Configuration ============================
PROJECT_NAME="reasoning_tasks" 

# 使用 APPO 自己的 config（从 ARPO 复制而来，路径已更新）
CONFIG_PATH="${APPO_DIR}/scripts/config"
CONFIG_NAME="ppo_trainer.yaml"

NNODES=1
N_GPUS_PER_NODE=4

# ============================ Data Configuration ============================
PROMPT_KEY="prompt"
TRAIN_BATCH_SIZE=96
PPO_MINI_BATCH_SIZE=16
MAX_PROMPT_LENGTH=2048  # 比 ARPO 多 512，为 COGNITIVE_ADDON（约 400 token）留出空间
MAX_RESPONSE_LENGTH=6144  # 比 ARPO 多 2048：补偿 cognitive addon 注入 + 认知标签本身占用的 response token

# 使用 APPO 数据集（COGNITIVE_SYSTEM_PROMPT_ADDON 在 rollout 时动态注入）
TRAIN_FILES="${APPO_DIR}/rl_datasets/train_10k.parquet"
VALID_FILES="${APPO_DIR}/rl_datasets/valid.parquet"

# ============================ Model Configuration ============================
ACTOR_MODEL_PATH="/data/Agent/models/Qwen2.5-7B-AEPO"
# 
LASTING=${ACTOR_MODEL_PATH##*/}
EXPERIMENT_NAME="APPO_7B_procedural_branching_$LASTING"
# ============================ Rollout Configuration ==========================
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
# APPO 设计：INITIAL_ROLLOUTS × BEAM_SIZE = ROLLOUT_N（branching 补满预算）
# APPO 对齐：INITIAL_ROLLOUTS=8，剩余 8 个配额由 cognitive branch 补满
ROLLOUT_N=16
INITIAL_ROLLOUTS=8
ENABLE_MULTI_TURN=True

# ============================ Rollout Tools Configuration ==========================
SEARCH_CACHE_PATH="${APPO_DIR}/search_cache/search_cache.json"
TOOL_CONFIG_PATH="${SEARCH_CACHE_PATH}"

# ============================ Reward Model Configuration ==========================
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${APPO_DIR}/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
TOTAL_EPOCHS=2
SAVE_FREQ=100
TEST_FREQ=1

# ============================ Path Configuration ============================
SAVE_PATH="/data/Agent/logs/ckpt_savedir/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
WANDB_API_KEY="${WANDB_API_KEY:-}"
SEARCH_CLASS_PATH="verl.workers.rollout.tools.search_tool.BingSearchTool"

if [ "$WANDB_API_KEY" != "" ]; then
    export WANDB_API_KEY=${WANDB_API_KEY}
    export WANDB_DIR=${SAVE_PATH}
    export WANDB_MODE=offline
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# ============================ Start Training ============================
python -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=appo \
    algorithm.reward_scale_discount=0.9 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.policy_loss.loss_mode=future_kl \
    actor_rollout_ref.actor.policy_loss.decay_rate=64.0 \
    actor_rollout_ref.actor.policy_loss.chunk_size=128 \
    actor_rollout_ref.actor.policy_loss.future_kl_clip_ratio=0.2 \
    actor_rollout_ref.actor.policy_loss.future_kl_clip_high_only=True \
    actor_rollout_ref.actor.policy_loss.future_kl_weight=1.0 \
    actor_rollout_ref.actor.policy_loss.safety_thresh=6.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.appo_dynamic_branching=True \
    actor_rollout_ref.rollout.reward_scale_discount=0.9 \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.class_path=${SEARCH_CLASS_PATH} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG_PATH} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log
