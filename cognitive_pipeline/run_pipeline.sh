#!/bin/bash
# 认知思维模式数据管线 - 一键运行脚本
#
# 用法：
#   bash run_pipeline.sh [选项]
#
# 选项：
#   --skip-phase1              跳过 Phase 1（GPT-4o 打标），直接用已有种子数据
#   --skip-phase2              跳过 Phase 2（Qwen3 全量打标）
#   --api-key   <KEY>          OpenAI API Key（也可通过 OPENAI_API_KEY 环境变量传入）
#   --api-url   <URL>          OpenAI API Base URL，用于自定义网关
#                              默认：https://api.openai.com/v1
#
# 示例：
#   bash run_pipeline.sh --api-key sk-xxx --api-url https://your-gateway.com/v1
#   bash run_pipeline.sh --skip-phase1  # 跳过 Phase 1，直接跑 Phase 2

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PIPELINE_DIR}/logs"
mkdir -p "${LOG_DIR}"

SKIP_PHASE1=false
SKIP_PHASE2=false
CLI_API_KEY=""
CLI_API_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-phase1) SKIP_PHASE1=true; shift ;;
        --skip-phase2) SKIP_PHASE2=true; shift ;;
        --api-key)     CLI_API_KEY="$2"; shift 2 ;;
        --api-url)     CLI_API_URL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 命令行参数优先于环境变量
if [ -n "${CLI_API_KEY}" ]; then
    export OPENAI_API_KEY="${CLI_API_KEY}"
fi
if [ -n "${CLI_API_URL}" ]; then
    export OPENAI_BASE_URL="${CLI_API_URL}"
fi

echo "======================================"
echo "  Cognitive Pipeline Starting"
echo "======================================"
echo "Pipeline dir: ${PIPELINE_DIR}"
echo "Skip Phase 1: ${SKIP_PHASE1}"
echo "Skip Phase 2: ${SKIP_PHASE2}"
echo ""

# 检查 Python 环境
PYTHON=$(which python3)
echo "Python: ${PYTHON}"

# ============================================================
# Phase 1: GPT-4o 打标种子数据
# ============================================================
if [ "${SKIP_PHASE1}" = false ]; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: OPENAI_API_KEY is not set. Please export OPENAI_API_KEY=your_key"
        exit 1
    fi

    echo ""
    echo "--- Phase 1: GPT-4o Seed Annotation ---"
    echo "Start time: $(date)"

    ${PYTHON} "${PIPELINE_DIR}/phase1_gpt_annotate.py"

    echo "Phase 1 complete at: $(date)"
else
    echo "--- Phase 1: SKIPPED ---"
    if [ ! -f "${PIPELINE_DIR}/phase1_gpt/filtered_seed.jsonl" ]; then
        echo "WARNING: Phase 1 seed data not found at phase1_gpt/filtered_seed.jsonl"
        echo "Phase 2 will run without few-shot examples."
    fi
fi

# ============================================================
# Phase 2: Qwen3-30B 全量打标
# ============================================================
if [ "${SKIP_PHASE2}" = false ]; then
    echo ""
    echo "--- Phase 2: Qwen3-30B Full Annotation ---"
    echo "Checking vllm service at http://localhost:8001 ..."

    if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo ""
        echo "vllm service is not running. Please start it first:"
        echo ""
        echo "  conda activate sft  # 或你的 vllm 环境"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model /mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B-Instruct-2507 \\"
        echo "    --port 8001 \\"
        echo "    --max-model-len 16384 \\"
        echo "    --tensor-parallel-size 4 \\"
        echo "    --served-model-name qwen3-30b"
        echo ""
        echo "Then re-run: bash run_pipeline.sh --skip-phase1"
        exit 1
    fi

    echo "vllm service is running."
    echo "Start time: $(date)"

    ${PYTHON} "${PIPELINE_DIR}/phase2_qwen_annotate.py"

    echo "Phase 2 complete at: $(date)"
else
    echo "--- Phase 2: SKIPPED ---"
fi

# ============================================================
# 构建最终数据集
# ============================================================
echo ""
echo "--- Building Final Dataset ---"
echo "Start time: $(date)"

${PYTHON} "${PIPELINE_DIR}/phase3_build_dataset.py" 

echo ""
echo "======================================"
echo "  Pipeline Complete!"
echo "  Final dataset: ${PIPELINE_DIR}/output/cognitive_sft_data.jsonl"
echo "======================================"




# bash /mnt/workspace/wxc/AERPO/cognitive_pipeline/run_pipeline.sh \
#   --api-key fEUqZab1vvhezVgMElSNPzzi \
#   --api-url https://ai-llm-gateway.amap.com/v1

python -m vllm.entrypoints.openai.api_server \
  --model /mnt/workspace/wxc/Agent/models/Qwen2.5-7B-Instruct \
  --port 8001 --max-model-len 16384 --tensor-parallel-size 4 \
  --served-model-name qwen3-30b


#/mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B-Instruct-2507 


# CUDA_VISIBLE_DEVICES=0,1,2,4 /mnt/workspace/wxc/miniconda3/envs/sft/bin/python -m vllm.entrypoints.openai.api_server \
#   --model /mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B-Instruct-2507 \
#   --port 8001 \
#   --max-model-len 16384 \
#   --tensor-parallel-size 2 \
#   --served-model-name qwen3-30b > /mnt/workspace/wxc/AERPO/cognitive_pipeline/vllm.log 2>&1 &
# echo "vllm PID=$!"



# /mnt/workspace/wxc/miniconda3/envs/sft/bin/python \
#   /mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_qwen_annotate.py \
#   > /mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_annotate.log 2>&1 &
# echo "打标 PID=$!"