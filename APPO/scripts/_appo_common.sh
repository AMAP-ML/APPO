#!/bin/bash
# Shared path setup for APPO launch scripts.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APPO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$APPO_DIR")"
cd "$APPO_DIR"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=WARNING
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
export RAY_memory_monitor_refresh_ms=0
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_tmp}
mkdir -p "${RAY_TMPDIR}"

export PYTHONPATH="${APPO_DIR}/verl_arpo_entropy:${REPO_ROOT}:${PYTHONPATH}"

CONFIG_PATH="${APPO_DIR}/scripts/config"
ROLLOUT_NAME="vllm"
ROLLOUT_MODE="sync_with_tool"
ENABLE_MULTI_TURN=True
REWARD_MANAGER="naive"
CUSTOM_REWARD_FUNCTION_PATH="${APPO_DIR}/verl_arpo_entropy/verl/utils/reward_score/deep_research.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"
SEARCH_CLASS_PATH="verl.workers.rollout.tools.search_tool.BingSearchTool"
SEARCH_CACHE_PATH="${APPO_DIR}/search_cache/search_cache.json"
TOOL_CONFIG_PATH="${SEARCH_CACHE_PATH}"

appo_prepare_dirs() {
    if [ ! -d "$SAVE_PATH" ]; then
        mkdir -p "$SAVE_PATH"
    fi
    if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
        mkdir -p "$ROLLOUT_SAVE_PATH"
    fi
}

appo_run_training() {
    python -m verl.trainer.main_ppo \
        --config-path="${CONFIG_PATH}" \
        --config-name="${CONFIG_NAME}" \
        algorithm.adv_estimator=appo \
        algorithm.reward_scale_discount=0.9 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        data.train_files="${TRAIN_FILES}" \
        data.val_files="${VALID_FILES}" \
        data.prompt_key="${PROMPT_KEY}" \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
        data.max_response_length="${MAX_RESPONSE_LENGTH}" \
        actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
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
        actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
        actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
        actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
        actor_rollout_ref.rollout.initial_rollouts="${INITIAL_ROLLOUTS}" \
        actor_rollout_ref.rollout.appo_dynamic_branching=True \
        actor_rollout_ref.rollout.reward_scale_discount=0.9 \
        actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file="${SEARCH_CACHE_PATH}" \
        actor_rollout_ref.rollout.tools.tool_instances.search.class_path="${SEARCH_CLASS_PATH}" \
        actor_rollout_ref.rollout.multi_turn.enable="${ENABLE_MULTI_TURN}" \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        reward_model.reward_manager="${REWARD_MANAGER}" \
        custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
        custom_reward_function.name="${CUSTOM_REWARD_FUNCTION_NAME}" \
        trainer.critic_warmup=0 \
        trainer.logger="[console, wandb]" \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.default_local_dir="${SAVE_PATH}" \
        trainer.val_before_train=False \
        trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}" \
        hydra.run.dir="${SAVE_PATH}/outputs" 2>&1 | tee "${SAVE_PATH}/run.log"
}
