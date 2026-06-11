"""
认知思维模式数据管线 - 统一配置
"""

import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = "/mnt/workspace/wxc/AERPO/cognitive_pipeline"
INPUT_JSONL = "/mnt/workspace/wxc/AERPO/LLaMA-Factory/data/final_5w4_still_filtered.jsonl"

PHASE1_DIR = os.path.join(BASE_DIR, "phase1_gpt")
PHASE2_DIR = os.path.join(BASE_DIR, "phase2_qwen")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Phase 1 输出
PHASE1_ANNOTATED = os.path.join(PHASE1_DIR, "annotated_seed.jsonl")
PHASE1_FILTERED = os.path.join(PHASE1_DIR, "filtered_seed.jsonl")

# Phase 2 输出
PHASE2_ANNOTATED = os.path.join(PHASE2_DIR, "annotated_full.jsonl")

# 最终输出
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "cognitive_sft_data.jsonl")

# ============================================================
# 模型配置
# ============================================================
QWEN3_MODEL_PATH = "/mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B-Instruct-2507"
QWEN3_VLLM_BASE_URL = "http://localhost:8002/v1"  # vllm 服务地址

# GPT 配置（从环境变量读取 API Key）
GPT_MODEL = "claude_sonnet4_5"
GPT_API_KEY = "fEUqZab1vvhezVgMElSNPzzi"
GPT_BASE_URL = "https://ai-llm-gateway.amap.com/v1"
GPT_MAX_TOKENS = 8192

# ============================================================
# 管线参数
# ============================================================
PHASE1_SAMPLE_SIZE = 500       # Phase 1 从全量数据中采样的条数
PHASE1_NUM_VOTES = 3           # 多数投票次数
PHASE1_TEMPERATURE = 0.3       # GPT 采样温度（低温保证一致性）

PHASE2_BATCH_SIZE = 50         # Phase 2 批处理大小
PHASE2_MAX_WORKERS = 4         # 并发请求数
PHASE2_TEMPERATURE = 0.2

MAX_CHARS_PER_SAMPLE = 30000   # 超长样本过滤阈值

# ============================================================
# 7 种认知思维模式定义
# quick       经验快速判断
# plan        计划与任务分解
# deduce      演绎/因果推理
# induce      归纳综合
# hypo        假设提出
# act         工具/实践反馈整合
# meta        元认知控制
# ============================================================
COGNITIVE_MODES = {
    "quick": {
        "tag": "quick",
        "name": "经验快速判断",
        "description": (
            "Based on existing knowledge or intuition, directly give a conclusion without "
            "unfolding a complete derivation. The answer comes immediately from memory or "
            "experience, with no explicit step-by-step reasoning chain. "
            "Use for common sense, known facts, and low-uncertainty judgments. "
            "Do NOT use when explicit reasoning steps, systematic verification, or "
            "multi-step derivation are present."
        ),
        "example": "I know from memory that the speed of light is approximately 3×10⁸ m/s.",
    },
    "plan": {
        "tag": "plan",
        "name": "计划与任务分解",
        "description": (
            "Explicitly organize goals, constraints, sub-tasks, and execution order — "
            "the 'how to approach this' organizational layer. "
            "Use when the paragraph is about breaking down the problem, listing steps, "
            "or determining the solution route before actual computation begins. "
            "Do NOT use for specific derivation details or conclusion calculations."
        ),
        "example": "First I'll find the total distance, then divide by time to get the average speed.",
    },
    "deduce": {
        "tag": "deduce",
        "name": "演绎/因果推理",
        "description": (
            "Starting from premises, rules, or formulas, derive a conclusion along an explicit "
            "logical chain. Covers conditional derivation, causal chains, formula substitution, "
            "and logical deduction. The reasoning moves from known facts to a new conclusion "
            "through explicit steps. "
            "Do NOT use when only a conjecture is proposed without derivation."
        ),
        "example": "Since F=ma and F=10N, m=2kg, therefore a = F/m = 10/2 = 5 m/s².",
    },
    "induce": {
        "tag": "induce",
        "name": "归纳综合",
        "description": (
            "Summarize a general rule or conclusion from multiple examples, evidence pieces, "
            "or observations. Moving from specific instances to a general pattern. "
            "Use for multi-evidence aggregation, consistency induction, and pattern extraction. "
            "Do NOT use when reasoning from a single premise by applying a rule (that is deduce)."
        ),
        "example": "All three search results mention the same date, so the answer is likely correct.",
    },
    "hypo": {
        "tag": "hypo",
        "name": "假设提出",
        "description": (
            "Explicitly propose a candidate explanation, candidate answer, or working hypothesis. "
            "This is the act of committing to a tentative answer to proceed with — "
            "'assume A holds', 'the candidate is X', 'let's guess and go'. "
            "Do NOT include the subsequent verification process itself "
            "(verification steps belong to deduce or quick)."
        ),
        "example": "Let me assume the answer is 10 years and proceed from there.",
    },
    "act": {
        "tag": "act",
        "name": "工具/实践反馈整合",
        "description": (
            "Update the reasoning based on outputs from external tools or execution results. "
            "Use when the paragraph reads search/result/python outputs and then modifies or "
            "advances the reasoning accordingly. "
            "Do NOT use for pure internal thinking that does not depend on external feedback."
        ),
        "example": "After searching, I found that HD 80606 has a mass of 0.98 solar masses, so I update my calculation.",
    },
    "meta": {
        "tag": "meta",
        "name": "元认知控制",
        "description": (
            "Monitor, correct, clarify, or reconstruct the reasoning process itself. "
            "Use for: clarifying the question scope or key terms before solving; "
            "pointing out a previous error and adjusting strategy; "
            "rewriting or restarting a reasoning path. "
            "Do NOT use for concrete entity-level derivation steps."
        ),
        "example": (
            "The question asks for rigorous imprisonment specifically, not just any sentence. "
            "/ Wait, I made an error earlier — the formula should use AU not km. Let me redo this."
        ),
    },
}

# ============================================================
# System Prompt 扩展（追加到原有 system prompt 之后）
# ============================================================

COGNITIVE_SYSTEM_PROMPT_ADDON = (
    "\n\nWhen you feel that a paragraph of your reasoning forms a self-contained stage "
    "(for example: a quick judgment, a planning step, a deduction, an induction, "
    "a hypothesis, integrating tool feedback, or a metacognitive correction), "
    "wrap that paragraph with <|extra_0|>...<|extra_1|>. "
    "Do not annotate tool calls or their results — only wrap pure reasoning paragraphs. "
    "Not every paragraph needs a tag; only use <|extra_0|> when a stage boundary is clear.\n"
)


def get_mode_definitions_for_prompt() -> str:
    """生成用于打标 prompt 的模式定义文本"""
    lines = []
    for key, mode in COGNITIVE_MODES.items():
        lines.append(f"- <{mode['tag']}></{mode['tag']}> ({mode['name']}): {mode['description']}")
        lines.append(f"  Example: {mode['example']}")
    return "\n".join(lines)
