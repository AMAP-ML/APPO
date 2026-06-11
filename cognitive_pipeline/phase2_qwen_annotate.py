"""
Phase 2：使用本地 Qwen3-30B-A3B 对全量数据（54K）批量打标。
策略：用 Phase 1 的种子数据作为 few-shot 示例，通过 vllm 服务批量推理。
运行前需先启动 vllm 服务：
  python -m vllm.entrypoints.openai.api_server \
    --model /mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B-Instruct-2507 \
    --port 8001 --max-model-len 16384 --tensor-parallel-size 4 \
    --served-model-name qwen3-30b
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COGNITIVE_MODES,
    INPUT_JSONL,
    MAX_CHARS_PER_SAMPLE,
    PHASE1_FILTERED,
    PHASE2_ANNOTATED,
    PHASE2_DIR,
    PHASE2_MAX_WORKERS,
    PHASE2_TEMPERATURE,
    QWEN3_VLLM_BASE_URL,
    get_mode_definitions_for_prompt,
)

VALID_TAGS = set(COGNITIVE_MODES.keys())
TOOL_TAGS = {"search", "result", "python", "answer"}
ALL_ALLOWED_TAGS = VALID_TAGS | TOOL_TAGS

# Phase 2 few-shot 示例数量（从 Phase 1 种子数据中取）
FEW_SHOT_COUNT = 3

# 旧标签 → 新7类标签的映射（用于把 Phase 1/旧版种子数据里的旧标签转换成新标签作为 few-shot 示例）
# 覆盖：原8类旧标签 + 中间过渡的6类标签
OLD_TO_NEW_TAG_MAP = {
    # 原8类 → 新7类
    "fast":      "quick",
    "deduce":    "deduce",
    "inductive": "induce",
    "analogy":   "deduce",    # 类比本质是演绎迁移
    "verify":    "hypothesis",
    "practice":  "interaction",
    "reflect":   "meta",
    "clarify":   "meta",
    # 中间6类 → 新7类
    "reason":    "deduce",    # reason 拆分为 deduce/induce，默认映射到 deduce
    "hypotest":  "hypothesis",
    "transfer":  "deduce",    # 类比迁移归入演绎
}
OLD_TAGS = set(OLD_TO_NEW_TAG_MAP.keys())

def remap_old_tags_to_new(text: str) -> str:
    """把文本里的旧8类标签替换成新6类标签，用于处理 Phase 1 种子数据的 few-shot 示例"""
    for old_tag, new_tag in OLD_TO_NEW_TAG_MAP.items():
        text = re.sub(rf"<{old_tag}>", f"<{new_tag}>", text)
        text = re.sub(rf"</{old_tag}>", f"</{new_tag}>", text)
    return text

# think 标签常量（用 chr 拼接避免工具解析问题）
_LT = chr(60)
_GT = chr(62)
THINK_OPEN = _LT + "think" + _GT
THINK_CLOSE = _LT + "/think" + _GT
_THINK_PATTERN = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
    re.DOTALL,
)


def extract_think_blocks(gpt_response: str) -> list:
    """提取 gpt 回复中所有 think 块的内容（不含标签本身）"""
    return _THINK_PATTERN.findall(gpt_response)

def validate_annotation(original_think: str, annotated_think: str) -> bool:
    """验证打标结果的合法性"""
    all_tags_alternation = "|".join(re.escape(t) for t in VALID_TAGS)

    any_open_tag_pattern = re.compile(r"<([a-z_]+)>")
    for match in any_open_tag_pattern.finditer(annotated_think):
        tag_name = match.group(1)
        if tag_name not in ALL_ALLOWED_TAGS:
            return False

    cognitive_tag_pattern = re.compile(
        rf"<({all_tags_alternation})>(.*?)</\1>", re.DOTALL
    )
    inner_cognitive_tag_pattern = re.compile(rf"<(?:{all_tags_alternation})>")
    for match in cognitive_tag_pattern.finditer(annotated_think):
        inner_content = match.group(2)
        if inner_cognitive_tag_pattern.search(inner_content):
            return False

    stripped_annotated = re.sub(rf"</?(?:{all_tags_alternation})>", "", annotated_think)
    stripped_original = re.sub(rf"</?(?:{all_tags_alternation})>", "", original_think)

    # 忽略空白字符差异（多余空格、换行等），只比较实际文本内容
    def normalize_whitespace(text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    if normalize_whitespace(stripped_annotated) != normalize_whitespace(stripped_original):
        return False

    return True


def reassemble_response(original_response: str, annotated_think_blocks: list) -> Optional[str]:
    """将打标后的 think 块内容替换回原始回复中"""
    result = original_response
    think_matches = list(_THINK_PATTERN.finditer(result))

    if len(think_matches) != len(annotated_think_blocks):
        return None

    for match, annotated_content in zip(reversed(think_matches), reversed(annotated_think_blocks)):
        start, end = match.span(1)
        result = result[:start] + annotated_content + result[end:]

    return result


def load_few_shot_examples(seed_jsonl_path: str, count: int) -> list:
    """从 Phase 1 种子数据中加载 few-shot 示例。
    
    Phase 1 种子数据里可能是旧的8类标签，加载时自动映射到新的6类标签。
    """
    examples = []
    if not os.path.exists(seed_jsonl_path):
        print(f"WARNING: Seed data not found at {seed_jsonl_path}, running without few-shot")
        return examples

    # 同时匹配新6类标签和旧8类标签
    all_known_tags = VALID_TAGS | OLD_TAGS
    with open(seed_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gpt_response = obj["conversations"][1]["value"]
            think_blocks = extract_think_blocks(gpt_response)
            # 只选有打标内容的样本作为 few-shot（兼容新旧标签）
            has_annotation = any(
                re.search(rf"<(?:{'|'.join(all_known_tags)})>", block)
                for block in think_blocks
            )
            if has_annotation and think_blocks:
                # 把旧标签映射到新标签后再存入
                new_gpt_response = remap_old_tags_to_new(gpt_response)
                new_obj = dict(obj)
                new_obj["conversations"] = [
                    obj["conversations"][0],
                    {**obj["conversations"][1], "value": new_gpt_response},
                ]
                examples.append(new_obj)
            if len(examples) >= count:
                break

    print(f"Loaded {len(examples)} few-shot examples from seed data")
    return examples


def build_few_shot_annotation_prompt(
    think_content: str,
    few_shot_examples: list,
) -> str:
    """构建带 few-shot 示例的打标 prompt"""
    mode_defs = get_mode_definitions_for_prompt()

    few_shot_section = ""
    if few_shot_examples:
        few_shot_lines = []
        for i, example in enumerate(few_shot_examples):
            gpt_response = example["conversations"][1]["value"]
            think_blocks = extract_think_blocks(gpt_response)
            if not think_blocks:
                continue
            # 取第一个 think 块作为示例
            original_block = re.sub(
                rf"</?(?:{'|'.join(VALID_TAGS)})>", "", think_blocks[0]
            ).strip()
            annotated_block = think_blocks[0].strip()
            few_shot_lines.append(f"### Example {i + 1}")
            few_shot_lines.append(f"Input:\n{original_block[:800]}")
            few_shot_lines.append(f"Output:\n{annotated_block[:1000]}")
            few_shot_lines.append("")
        few_shot_section = "\n## Examples:\n" + "\n".join(few_shot_lines)

    return f"""You are an expert in cognitive science and reasoning analysis.

Your task is to annotate a reasoning paragraph extracted from an AI's thinking process.
For each paragraph that clearly belongs to one of the 6 cognitive thinking modes below, wrap that paragraph with the corresponding XML tag.

## 6 Cognitive Thinking Modes:
{mode_defs}
{few_shot_section}
## Rules:
1. Annotate at paragraph granularity (one complete reasoning step per tag).
2. A paragraph may have AT MOST ONE cognitive mode tag.
3. Not every paragraph needs a tag — only annotate when the mode is CLEARLY identifiable.
4. Tags must NOT overlap or nest with each other.
5. Do NOT add, remove, or modify any text — only wrap existing paragraphs with tags.
6. The output must contain EXACTLY the same text as the input, just with optional wrapping tags added.

## Input reasoning content:
---
{think_content}
---

## Output:
Return ONLY the annotated reasoning content. Do not include any explanation or preamble."""


def annotate_think_block_with_qwen(
    client: OpenAI,
    think_content: str,
    few_shot_examples: list,
    sample_idx: int,
    block_idx: int,
) -> str:
    """调用 Qwen3 对单个 think 块打标，失败时返回原始内容"""
    if len(think_content.strip()) < 50:
        return think_content

    prompt = build_few_shot_annotation_prompt(think_content, few_shot_examples)
    try:
        response = client.chat.completions.create(
            model="qwen3-30b",
            messages=[{"role": "user", "content": prompt}],
            temperature=PHASE2_TEMPERATURE,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_content = response.choices[0].message.content
        if not raw_content:
            print(f"  [S{sample_idx} B{block_idx}] Empty response, using original")
            return think_content
        annotated_content = raw_content.strip()

        if not validate_annotation(think_content, annotated_content):
            print(f"  [S{sample_idx} B{block_idx}] Validation failed, using original")
            return think_content

        return annotated_content

    except Exception as error:
        print(f"  [S{sample_idx} B{block_idx}] Qwen error: {error}")
        return think_content  # 失败时保留原文，不丢弃样本


def process_single_sample(
    client: OpenAI,
    sample: dict,
    sample_idx: int,
    few_shot_examples: list,
) -> dict:
    """
    对单条样本进行打标。Phase 2 不做多数投票，失败时保留原文。
    始终返回样本（不丢弃），但标记是否成功打标。
    """
    gpt_response = sample["conversations"][1]["value"]
    think_blocks = extract_think_blocks(gpt_response)

    if not think_blocks:
        result = dict(sample)
        result["_annotated"] = False
        return result

    annotated_blocks = []
    for block_idx, think_content in enumerate(think_blocks):
        annotated = annotate_think_block_with_qwen(
            client, think_content, few_shot_examples, sample_idx, block_idx
        )
        annotated_blocks.append(annotated)

    annotated_response = reassemble_response(gpt_response, annotated_blocks)
    if annotated_response is None:
        result = dict(sample)
        result["_annotated"] = False
        return result

    result = dict(sample)
    result["conversations"] = [
        dict(sample["conversations"][0]),
        {**dict(sample["conversations"][1]), "value": annotated_response},
    ]
    result["_annotated"] = True
    return result


def load_all_samples(jsonl_path: str) -> list:
    """加载全量数据，过滤超长样本"""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            convs = obj.get("conversations") or obj.get("messages", [])
            total_chars = sum(
                len(str(t.get("value") or t.get("content", ""))) for t in convs
            )
            if total_chars <= MAX_CHARS_PER_SAMPLE:
                samples.append(obj)
    print(f"Loaded {len(samples)} valid samples (filtered out overlength)")
    return samples


def run_phase2():
    client = OpenAI(api_key="EMPTY", base_url=QWEN3_VLLM_BASE_URL)

    # 验证 vllm 服务是否可用
    try:
        models = client.models.list()
        print(f"vllm service available, models: {[m.id for m in models.data]}")
    except Exception as error:
        print(f"ERROR: Cannot connect to vllm service at {QWEN3_VLLM_BASE_URL}")
        print("Please start vllm first:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model /mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B-Instruct-2507 \\")
        print("    --port 8001 --max-model-len 16384 --tensor-parallel-size 4 \\")
        print("    --served-model-name qwen3-30b")
        print(f"Error: {error}")
        sys.exit(1)

    print(f"=== Phase 2: Qwen3-30B Annotation ===")

    few_shot_examples = load_few_shot_examples(PHASE1_FILTERED, FEW_SHOT_COUNT)
    samples = load_all_samples(INPUT_JSONL)

    os.makedirs(PHASE2_DIR, exist_ok=True)

    total = len(samples)
    annotated_count = 0

    # 断点续跑：直接数已处理行数，用切片跳过已处理样本
    # 注意：并发写入时顺序不保证，所以只能保证"跳过前 N 条"而非精确对应
    already_processed = 0
    if os.path.exists(PHASE2_ANNOTATED):
        with open(PHASE2_ANNOTATED, "r", encoding="utf-8") as f:
            already_processed = sum(1 for line in f if line.strip())
        if already_processed > 0:
            print(f"Resuming from checkpoint: {already_processed}/{total} samples already processed")

    remaining_samples = samples[already_processed:]
    if not remaining_samples:
        print("All samples already processed.")
        return

    with open(PHASE2_ANNOTATED, "a", encoding="utf-8") as out_f:
        def process_and_return(args):
            offset_idx, sample = args
            real_idx = already_processed + offset_idx
            return process_single_sample(client, sample, real_idx, few_shot_examples)

        with ThreadPoolExecutor(max_workers=PHASE2_MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_and_return, (i, sample)): i
                for i, sample in enumerate(remaining_samples)
            }

            completed = already_processed
            for future in as_completed(futures):
                result = future.result()
                clean_item = {k: v for k, v in result.items() if not k.startswith("_")}
                out_f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                out_f.flush()
                if result.get("_annotated"):
                    annotated_count += 1
                completed += 1
                if completed % 500 == 0:
                    print(f"Progress: {completed}/{total} ({completed / total * 100:.1f}%)")

    print(f"\n=== Phase 2 Complete ===")
    print(f"Total samples: {total}")
    print(f"Successfully annotated: {annotated_count}")
    print(f"Output saved to: {PHASE2_ANNOTATED}")


if __name__ == "__main__":
    run_phase2()
