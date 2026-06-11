<h1 align="center">APPO: Agentic Procedural Policy Optimization</h1>

<p align="center">
  <strong>APPO trains tool-using reasoning agents by branching at procedural decision points.</strong>
</p>

APPO extends the ARPO agentic RL stack with **procedure-aware branching**: it scores candidate branch tokens inside a rollout, expands the most informative decision points, and maps branch outcomes back to credit on the original trajectory. Cold-start SFT, datasets, tool cache, and evaluation all follow ARPO.

## Table of Contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)
  - [Cold-Start SFT Stage](#️-cold-start-sft-stage-optional)
  - [APPO RL Stage](#-appo-rl-stage)
  - [Evaluation](#-evaluation)
- [Citation](#-citation)

## 💡 Overview

ARPO branches when tool-call rounds become high-entropy. APPO locates **procedure points** via a **Branching Score**:

```text
BS_t = Z(Entropy_t) * Z(FutureValue_t)

FutureValue_t = exp(Σ_{k >= t} γ^(k-t) · (log π_current(a_k|s_k) - log π_rollout(a_k|s_k)))
```

Branch continuations reuse the same tool loop as initial rollouts, provide reward/advantage signals, and are excluded from actor loss. Only initial rollout tokens are optimized.

## 🏃 Quick Start

Reproducing APPO follows the same three-stage pipeline as ARPO: cold-start SFT (optional) → APPO RL training → evaluation.

## ❄️ Cold-Start SFT Stage (Optional)

This stage helps reproduce the full pipeline. You can skip it if you already have a warm-started checkpoint.

### 1. Environment Setup

```bash
cd LLaMA-Factory

conda create -n sft python=3.10
conda activate sft
pip install -r requirements.txt
```

### 2. Fine-Tuning Model

1. Prepare the SFT dataset following [ARPO](https://github.com/RUC-NLPIR/ARPO) and register it in `LLaMA-Factory/data/dataset_info.json`.
2. Update `LLaMA-Factory/arpo_train_sft/yaml/qwen.yaml` with your `model_name_or_path`, `dataset`, and `output_dir`.
3. Launch SFT:

```bash
bash APPO/scripts/sft_train.sh
```

## 🔥 APPO RL Stage

APPO RL builds on the same VERL + tool-use stack as ARPO. Datasets and search cache live under `ARPO/`.

### 1. Environment Setup

```bash
conda create -n appo python=3.10
conda activate appo

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

cd APPO
pip install -r verl_arpo_entropy/requirements.txt
pip install -e verl_arpo_entropy
```

### 2. Preparation

**Data** — use the same RL datasets as ARPO (see the ARPO README *Data Preparation* section). In this repo they are under `ARPO/rl_datasets/`.

**Tool API**

Replace Bright Data `api_key` and `zone` in:

- `APPO/scripts/config/ppo_trainer.yaml`
- `APPO/scripts/config/ppo_trainer_dr.yaml`
- `APPO/verl_arpo_entropy/verl/workers/rollout/tools/config_example.yaml`

### 3. APPO RL Training

Update `ACTOR_MODEL_PATH` and `SAVE_PATH` in the launch script, then run:

```bash
cd APPO/scripts

# 7B reasoning (Qwen)
bash APPO_7B_Reasoning_1node.sh

# 8B deep search
bash APPO_8B_Deepsearch_1node.sh

# 14B deep search
bash APPO_14B_Deepsearch_1node.sh
```

Legacy entrypoints with cluster-specific paths are also available:

```bash
bash APPO_1node.sh          # Qwen-style
bash APPO_1node_llama.sh    # Llama-style
```

Convert VERL checkpoints to Hugging Face format:

```bash
bash APPO/merge_ckpt/convert_checkpoint_from_verl_to_hf_qwen3.sh
```

**Key APPO settings**

```bash
algorithm.adv_estimator=appo
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=sync_with_tool
actor_rollout_ref.rollout.n=16
actor_rollout_ref.rollout.initial_rollouts=8
actor_rollout_ref.rollout.appo_dynamic_branching=True
actor_rollout_ref.rollout.reward_scale_discount=0.9
actor_rollout_ref.actor.policy_loss.loss_mode=future_kl
```

## ✅ Evaluation

Evaluation reuses the ARPO `evaluation/` pipeline unchanged.

### 1. Setup vLLM Inference Environment

```bash
cd evaluation/vllm_scripts
conda create -n vllm_env python=3.10
conda activate vllm_env
pip install -r requirements.txt
```

Edit model paths in `vllm_launch_reasoning_model_cuda4-7.sh` and a summarization launch script, then start services:

```bash
bash vllm_launch_reasoning_model_cuda4-7.sh
bash vllm_launch_summarize_model_cuda0-3_qwen3_8b.sh
```

### 2. Setup Evaluation Environment

```bash
conda create -n evaluation python=3.10
conda activate evaluation
cd evaluation
pip install -r requirements.txt
```

### 3. Run Evaluation

Edit `evaluation/infer_local_sds.sh` with your model path, output path, and Bing API credentials, then:

```bash
bash evaluation/infer_local_sds.sh
```

### 4. Calculate Metrics

```bash
bash evaluation/deploy_qwen2.5_72B_instruct.sh
bash evaluation/evaluate_passk.sh
```

## 📄 Citation

APPO citation: coming soon.

If you use the ARPO codebase or datasets, please cite:

```bibtex
@article{dong2025arpo,
  title   = {Agentic Reinforced Policy Optimization},
  author  = {Guanting Dong and Hangyu Mao and Kai Ma and others},
  journal = {CoRR},
  volume  = {abs/2507.19849},
  year    = {2025}
}
```
