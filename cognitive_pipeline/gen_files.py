"""
用 Python 字符串拼接生成 phase1 和 phase2 文件，
避免 create_file 工具吞掉 XML 标签的问题。
"""
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 生成 phase1_gpt_annotate.py
# ============================================================

THINK_OPEN = ""
THINK_OPEN_ESC = r"\<think\>"   # 用于注释，不用于代码

phase1_code = '''"""
Phase 1: GPT-4o 对少量样本（200条）进行思维模式打标。
策略：每条样本调用 GPT-4o 3次（多数投票），只保留 3次结果完全一致的样本作为种子数据。
"""

import json
import os
import random
import re
import sys
import time
from typing import Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COGNITIVE_MODES,
    GPT_API_KEY,
    GPT_BASE_URL,
    GPT_MODEL,
    INPUT_JSONL,
    MAX_CHARS_PER_SAMPLE,
    PHASE1_ANNOTATED,
    PHASE1_FILTERED,
    PHASE1_NUM_VOTES,
    PHASE1_SAMPLE_SIZE,
    PHASE1_TEMPERATURE,
    get_mode_definitions_for_prompt,
)

VALID_TAGS = set(COGNITIVE_MODES.keys())
TOOL_TAGS = {"search", "result", "python", "answer"}
ALL_ALLOWED_TAGS = VALID_TAGS | TOOL_TAGS

# 正则：匹配  块，捕获内容
_THINK_PATTERN = re.compile(r"''' + repr(THINK_OPEN) + r'''(.*?)''' + repr(THINK_CLOSE) + r'''", re.DOTALL)


def extract_think_blocks(gpt_response: str) -> list:
    """提取 gpt 回复中所有  块的内容（不含标签本身）"""
    return _THINK_PATTERN.findall(gpt_response)


def reassemble_response(original_response: str, annotated_think_blocks: list) -> Optional[str]:
    """将打标后的 think 块内容替换回原始回复中"""
    result = original_response
    think_matches = list(_THINK_PATTERN.finditer(result))

    if len(think_matches) != len(annotated_think_blocks):
        return None

    # 从后往前替换，避免偏移量变化
    for match, annotated_content in zip(reversed(think_matches), reversed(annotated_think_blocks)):
        start, end = match.span(1)  # span(1) 是捕获组的范围，即 think 内容部分
        result = result[:start] + annotated_content + result[end:]

    return result


def validate_annotation(original_think: str, annotated_think: str) -> bool:
    """
    验证打标结果的合法性：
    1. 去掉所有认知标记后，文本内容与原始完全一致
    2. 没有非法标签
    3. 认知标签没有嵌套
    """
    all_tags_alternation = "|".join(re.escape(t) for t in VALID_TAGS)

    # 检查是否有非法标签
    any_open_tag_pattern = re.compile(r"<([a-z_]+)>")
    for match in any_open_tag_pattern.finditer(annotated_think):
        tag_name = match.group(1)
        if tag_name not in ALL_ALLOWED_TAGS:
            return False

    # 检查认知标签嵌套
    cognitive_tag_pattern = re.compile(
        rf"<({all_tags_alternation})>(.*?)</\\1>", re.DOTALL
    )
    inner_cognitive_tag_pattern = re.compile(rf"<(?:{all_tags_alternation})>")
    for match in cognitive_tag_pattern.finditer(annotated_think):
        inner_content = match.group(2)
        if inner_cognitive_tag_pattern.search(inner_content):
            return False

    # 去掉所有认知标签后，文本应与原始一致
    stripped_annotated = re.sub(rf"</?(?:{all_tags_alternation})>", "", annotated_think)
    stripped_original = re.sub(rf"</?(?:{all_tags_alternation})>", "", original_think)
    if stripped_annotated.strip() != stripped_original.strip():
        return False

    return True


def build_annotation_prompt(think_content: str) -> str:
    """构建打标 prompt，输入为单个 think 块的内容（不含标签）"""
    mode_defs = get_mode_definitions_for_prompt()
    return (
        "You are an expert in cognitive science and reasoning analysis.\\n\\n"
        "Your task is to annotate a reasoning paragraph extracted from an AI\'s thinking process.\\n"
        "The reasoning may contain multiple paragraphs. For each paragraph that clearly belongs to "
        "one of the 8 cognitive thinking modes below, wrap that paragraph with the corresponding XML tag.\\n\\n"
        "## 8 Cognitive Thinking Modes:\\n"
        + mode_defs
        + "\\n\\n"
        "## Rules:\\n"
        "1. Annotate at paragraph granularity (one complete reasoning step per tag).\\n"
        "2. A paragraph may have AT MOST ONE cognitive mode tag.\\n"
        "3. Not every paragraph needs a tag — only annotate when the mode is CLEARLY identifiable.\\n"
        "4. Tags must NOT overlap or nest with each other.\\n"
        "5. Do NOT add, remove, or modify any text — only wrap existing paragraphs with tags.\\n"
        "6. The output must contain EXACTLY the same text as the input, just with optional wrapping tags added.\\n"
        "7. Do NOT annotate tool-related content (search queries, search results, python code, final answers).\\n\\n"
        "## Input reasoning content:\\n"
        "---\\n"
        + think_content
        + "\\n---\\n\\n"
        "## Output:\\n"
        "Return ONLY the annotated reasoning content (same text with optional cognitive mode tags added). "
        "Do not include any explanation or preamble."
    )


def annotate_single_think_block(
    client: OpenAI,
    think_content: str,
    sample_idx: int,
    block_idx: int,
    vote_idx: int,
) -> Optional[str]:
    """调用 GPT-4o 对单个 think 块打标，返回打标后的内容"""
    if len(think_content.strip()) < 50:
        return think_content

    prompt = build_annotation_prompt(think_content)
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=PHASE1_TEMPERATURE,
            max_tokens=8192,
        )
        annotated_content = response.choices[0].message.content.strip()

        if not validate_annotation(think_content, annotated_content):
            print(f"  [S{sample_idx} V{vote_idx} B{block_idx}] Validation failed, using original")
            return think_content

        return annotated_content

    except Exception as error:
        print(f"  [S{sample_idx} V{vote_idx} B{block_idx}] GPT error: {error}")
        return None


def annotate_all_think_blocks(
    client: OpenAI,
    think_blocks: list,
    sample_idx: int,
    vote_idx: int,
) -> Optional[list]:
    """对一条样本的所有 think 块打标，任意一块失败则返回 None"""
    annotated_blocks = []
    for block_idx, think_content in enumerate(think_blocks):
        result = annotate_single_think_block(
            client, think_content, sample_idx, block_idx, vote_idx
        )
        if result is None:
            return None
        annotated_blocks.append(result)
        time.sleep(0.3)
    return annotated_blocks


def normalize_for_comparison(text: str) -> str:
    """标准化文本用于多数投票比较"""
    return re.sub(r"\\s+", " ", text).strip()


def process_single_sample(
    client: OpenAI,
    sample: dict,
    sample_idx: int,
) -> Optional[dict]:
    """
    对单条样本进行 PHASE1_NUM_VOTES 次打标，取多数投票一致的结果。
    返回打标后的样本，或 None（如果投票不一致或出错）。
    """
    gpt_response = sample["conversations"][1]["value"]
    think_blocks = extract_think_blocks(gpt_response)

    if not think_blocks:
        result = dict(sample)
        result["_annotated"] = False
        result["_vote_consistent"] = True
        return result

    vote_responses = []
    for vote_idx in range(PHASE1_NUM_VOTES):
        annotated_blocks = annotate_all_think_blocks(
            client, think_blocks, sample_idx, vote_idx
        )
        if annotated_blocks is None:
            print(f"  [Sample {sample_idx}] Vote {vote_idx} failed, skipping sample")
            return None

        annotated_response = reassemble_response(gpt_response, annotated_blocks)
        if annotated_response is None:
            print(f"  [Sample {sample_idx}] Reassemble failed at vote {vote_idx}")
            return None

        vote_responses.append(annotated_response)
        time.sleep(0.5)

    normalized_votes = [normalize_for_comparison(r) for r in vote_responses]
    if len(set(normalized_votes)) == 1:
        final_response = vote_responses[0]
        result = dict(sample)
        result["conversations"] = [
            dict(sample["conversations"][0]),
            {**dict(sample["conversations"][1]), "value": final_response},
        ]
        result["_annotated"] = True
        result["_vote_consistent"] = True
        return result
    else:
        print(f"  [Sample {sample_idx}] Votes inconsistent ({len(set(normalized_votes))} unique), discarding")
        return None


def load_samples(jsonl_path: str, sample_size: int, seed: int = 42) -> list:
    """从 JSONL 文件中随机采样，过滤超长样本"""
    all_samples = []
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
                all_samples.append(obj)

    random.seed(seed)
    sampled = random.sample(all_samples, min(sample_size, len(all_samples)))
    print(f"Loaded {len(all_samples)} valid samples, sampled {len(sampled)}")
    return sampled


def run_phase1():
    if not GPT_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL)

    print(f"=== Phase 1: GPT-4o Annotation ===")
    print(f"Sample size: {PHASE1_SAMPLE_SIZE}, Votes per sample: {PHASE1_NUM_VOTES}")

    samples = load_samples(INPUT_JSONL, PHASE1_SAMPLE_SIZE)

    all_results = []
    consistent_count = 0

    for idx, sample in enumerate(samples):
        print(f"Processing sample {idx + 1}/{len(samples)}...")
        result = process_single_sample(client, sample, idx)
        if result is not None:
            all_results.append(result)
            if result.get("_vote_consistent"):
                consistent_count += 1

    os.makedirs(os.path.dirname(PHASE1_ANNOTATED), exist_ok=True)
    with open(PHASE1_ANNOTATED, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")

    consistent_results = [r for r in all_results if r.get("_vote_consistent")]
    with open(PHASE1_FILTERED, "w", encoding="utf-8") as f:
        for item in consistent_results:
            clean_item = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\\n")

    print(f"\\n=== Phase 1 Complete ===")
    print(f"Total processed: {len(samples)}")
    print(f"Vote-consistent seed samples: {consistent_count}")
    print(f"Consistency rate: {consistent_count / len(samples) * 100:.1f}%")
    print(f"Seed data saved to: {PHASE1_FILTERED}")


if __name__ == "__main__":
    run_phase1()
'''

# ============================================================
# 生成 phase2_qwen_annotate.py
# ============================================================

phase2_code = '''"""
Phase 2: 使用本地 Qwen3-30B-A3B 对全量数据（54K）批量打标。
策略：用 Phase 1 的种子数据作为 few-shot 示例，通过 vllm 服务批量推理。
运行前需先启动 vllm 服务：
  python -m vllm.entrypoints.openai.api_server \\
    --model /mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B \\
    --port 8001 --max-model-len 16384 --tensor-parallel-size 4 \\
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
    PHASE2_MAX_WORKERS,
    PHASE2_TEMPERATURE,
    QWEN3_VLLM_BASE_URL,
    get_mode_definitions_for_prompt,
)

VALID_TAGS = set(COGNITIVE_MODES.keys())
TOOL_TAGS = {"search", "result", "python", "answer"}
ALL_ALLOWED_TAGS = VALID_TAGS | TOOL_TAGS
FEW_SHOT_COUNT = 3

_THINK_PATTERN = re.compile(r"''' + repr(THINK_OPEN) + r'''(.*?)''' + repr(THINK_CLOSE) + r'''", re.DOTALL)


def extract_think_blocks(gpt_response: str) -> list:
    """提取 gpt 回复中所有  块的内容（不含标签本身）"""
    return _THINK_PATTERN.findall(gpt_response)


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


def validate_annotation(original_think: str, annotated_think: str) -> bool:
    """验证打标结果的合法性"""
    all_tags_alternation = "|".join(re.escape(t) for t in VALID_TAGS)

    any_open_tag_pattern = re.compile(r"<([a-z_]+)>")
    for match in any_open_tag_pattern.finditer(annotated_think):
        tag_name = match.group(1)
        if tag_name not in ALL_ALLOWED_TAGS:
            return False

    cognitive_tag_pattern = re.compile(
        rf"<({all_tags_alternation})>(.*?)</\\1>", re.DOTALL
    )
    inner_cognitive_tag_pattern = re.compile(rf"<(?:{all_tags_alternation})>")
    for match in cognitive_tag_pattern.finditer(annotated_think):
        inner_content = match.group(2)
        if inner_cognitive_tag_pattern.search(inner_content):
            return False

    stripped_annotated = re.sub(rf"</?(?:{all_tags_alternation})>", "", annotated_think)
    stripped_original = re.sub(rf"</?(?:{all_tags_alternation})>", "", original_think)
    if stripped_annotated.strip() != stripped_original.strip():
        return False

    return True


def load_few_shot_examples(seed_jsonl_path: str, count: int) -> list:
    """从 Phase 1 种子数据中加载 few-shot 示例（只选有打标内容的样本）"""
    examples = []
    if not os.path.exists(seed_jsonl_path):
        print(f"WARNING: Seed data not found at {seed_jsonl_path}, running without few-shot")
        return examples

    all_tags_alternation = "|".join(re.escape(t) for t in VALID_TAGS)
    has_annotation_pattern = re.compile(rf"<(?:{all_tags_alternation})>")

    with open(seed_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gpt_response = obj["conversations"][1]["value"]
            think_blocks = extract_think_blocks(gpt_response)
            has_annotation = any(
                has_annotation_pattern.search(block) for block in think_blocks
            )
            if has_annotation and think_blocks:
                examples.append(obj)
            if len(examples) >= count:
                break

    print(f"Loaded {len(examples)} few-shot examples from seed data")
    return examples


def build_few_shot_annotation_prompt(think_content: str, few_shot_examples: list) -> str:
    """构建带 few-shot 示例的打标 prompt"""
    mode_defs = get_mode_definitions_for_prompt()
    all_tags_alternation = "|".join(re.escape(t) for t in VALID_TAGS)
    strip_tags_pattern = re.compile(rf"</?(?:{all_tags_alternation})>")

    few_shot_section = ""
    if few_shot_examples:
        few_shot_lines = ["\\n## Examples (annotated reasoning samples for reference):"]
        for i, example in enumerate(few_shot_examples):
            gpt_response = example["conversations"][1]["value"]
            think_blocks = extract_think_blocks(gpt_response)
            if not think_blocks:
                continue
            # 取第一个有打标的 think 块作为示例
            for block in think_blocks:
                if re.compile(rf"<(?:{all_tags_alternation})>").search(block):
                    original_block = strip_tags_pattern.sub("", block).strip()
                    annotated_block = block.strip()
                    few_shot_lines.append(f"\\n### Example {i + 1}")
                    few_shot_lines.append(f"Input:\\n{original_block[:600]}")
                    few_shot_lines.append(f"Output:\\n{annotated_block[:800]}")
                    break
        few_shot_section = "\\n".join(few_shot_lines)

    return (
        "You are an expert in cognitive science and reasoning analysis.\\n\\n"
        "Your task is to annotate a reasoning paragraph extracted from an AI\'s thinking process.\\n"
        "For each paragraph that clearly belongs to one of the 8 cognitive thinking modes below, "
        "wrap that paragraph with the corresponding XML tag.\\n\\n"
        "## 8 Cognitive Thinking Modes:\\n"
        + mode_defs
        + few_shot_section
        + "\\n\\n## Rules:\\n"
        "1. Annotate at paragraph granularity (one complete reasoning step per tag).\\n"
        "2. A paragraph may have AT MOST ONE cognitive mode tag.\\n"
        "3. Not every paragraph needs a tag — only annotate when the mode is CLEARLY identifiable.\\n"
        "4. Tags must NOT overlap or nest with each other.\\n"
        "5. Do NOT add, remove, or modify any text — only wrap existing paragraphs with tags.\\n"
        "6. The output must contain EXACTLY the same text as the input, just with optional wrapping tags added.\\n\\n"
        "## Input reasoning content:\\n"
        "---\\n"
        + think_content
        + "\\n---\\n\\n"
        "## Output:\\n"
        "Return ONLY the annotated reasoning content. Do not include any explanation or preamble."
    )


def annotate_think_block_with_qwen(
    client: OpenAI,
    think_content: str,
    few_shot_examples: list,
    sample_idx: int,
    block_idx: int,
) -> str:
    """调用 Qwen3 对单个 think 块打标，失败时返回原始内容（不丢弃样本）"""
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
        annotated_content = response.choices[0].message.content.strip()

        if not validate_annotation(think_content, annotated_content):
            print(f"  [S{sample_idx} B{block_idx}] Validation failed, using original")
            return think_content

        return annotated_content

    except Exception as error:
        print(f"  [S{sample_idx} B{block_idx}] Qwen error: {error}")
        return think_content


def process_single_sample(
    client: OpenAI,
    sample: dict,
    sample_idx: int,
    few_shot_examples: list,
) -> dict:
    """对单条样本进行打标。Phase 2 不做多数投票，失败时保留原文，始终返回样本。"""
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


def count_already_processed(output_path: str) -> int:
    """统计已处理的样本数，用于断点续跑"""
    if not os.path.exists(output_path):
        return 0
    count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def run_phase2():
    client = OpenAI(api_key="EMPTY", base_url=QWEN3_VLLM_BASE_URL)

    # 验证 vllm 服务是否可用
    try:
        models = client.models.list()
        print(f"vllm service available, models: {[m.id for m in models.data]}")
    except Exception as error:
        print(f"ERROR: Cannot connect to vllm service at {QWEN3_VLLM_BASE_URL}")
        print(f"Please start vllm first:")
        print(f"  python -m vllm.entrypoints.openai.api_server \\\\")
        print(f"    --model /mnt/workspace/wxc/Agent/models/Qwen3-30B-A3B \\\\")
        print(f"    --port 8001 --max-model-len 16384 --tensor-parallel-size 4 \\\\")
        print(f"    --served-model-name qwen3-30b")
        print(f"Error detail: {error}")
        sys.exit(1)

    print(f"=== Phase 2: Qwen3-30B Annotation ===")

    few_shot_examples = load_few_shot_examples(PHASE1_FILTERED, FEW_SHOT_COUNT)
    samples = load_all_samples(INPUT_JSONL)

    os.makedirs(os.path.dirname(PHASE2_ANNOTATED), exist_ok=True)

    # 断点续跑：统计已处理数量，跳过已处理的样本
    already_processed = count_already_processed(PHASE2_ANNOTATED)
    if already_processed > 0:
        print(f"Resuming from checkpoint: {already_processed} samples already processed")

    remaining_samples = samples[already_processed:]
    total = len(samples)
    annotated_count = 0

    with open(PHASE2_ANNOTATED, "a", encoding="utf-8") as out_f:
        def process_and_return(args):
            idx, sample = args
            return process_single_sample(client, sample, idx, few_shot_examples)

        with ThreadPoolExecutor(max_workers=PHASE2_MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_and_return, (already_processed + i, sample)): i
                for i, sample in enumerate(remaining_samples)
            }

            completed = already_processed
            for future in as_completed(futures):
                result = future.result()
                clean_item = {k: v for k, v in result.items() if not k.startswith("_")}
                out_f.write(json.dumps(clean_item, ensure_ascii=False) + "\\n")
                out_f.flush()
                if result.get("_annotated"):
                    annotated_count += 1
                completed += 1
                if completed % 100 == 0:
                    print(f"Progress: {completed}/{total} ({completed / total * 100:.1f}%)")

    print(f"\\n=== Phase 2 Complete ===")
    print(f"Total samples: {total}")
    print(f"Successfully annotated: {annotated_count}")
    print(f"Output saved to: {PHASE2_ANNOTATED}")


if __name__ == "__main__":
    run_phase2()
'''

# ============================================================
# 写入文件
# ============================================================

phase1_path = os.path.join(PIPELINE_DIR, "phase1_gpt_annotate.py")
phase2_path = os.path.join(PIPELINE_DIR, "phase2_qwen_annotate.py")

with open(phase1_path, "w", encoding="utf-8") as f:
    f.write(phase1_code)
print(f"Written: {phase1_path}")

with open(phase2_path, "w", encoding="utf-8") as f:
    f.write(phase2_code)
print(f"Written: {phase2_path}")

# 语法检查
import ast
for path in [phase1_path, phase2_path]:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"Syntax OK: {path}")
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {path}: {e}")
