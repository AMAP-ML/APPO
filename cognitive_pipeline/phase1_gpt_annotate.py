"""
Phase 1：使用 GPT-4o 对少量样本（200条）进行思维模式打标。
策略：每条样本调用 GPT-4o 3次（多数投票），只保留 3次结果完全一致的样本作为种子数据。
"""

import json
import os
import random
import re
import sys
import time
from typing import Optional

import requests
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COGNITIVE_MODES,
    GPT_API_KEY,
    GPT_BASE_URL,
    GPT_MAX_TOKENS,
    GPT_MODEL,
    INPUT_JSONL,
    MAX_CHARS_PER_SAMPLE,
    PHASE1_ANNOTATED,
    PHASE1_DIR,
    PHASE1_FILTERED,
    PHASE1_NUM_VOTES,
    PHASE1_SAMPLE_SIZE,
    PHASE1_TEMPERATURE,
    get_mode_definitions_for_prompt,
)

VALID_TAGS = set(COGNITIVE_MODES.keys())
# 工具标签不打认知标记
TOOL_TAGS = {"search", "result", "python", "answer", "think"}
ALL_ALLOWED_TAGS = VALID_TAGS | TOOL_TAGS

# think 标签常量（用 chr 拼接避免工具解析问题）
_LT = chr(60)
_GT = chr(62)
THINK_OPEN = _LT + "think" + _GT
THINK_CLOSE = _LT + "/think" + _GT
_THINK_PATTERN = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
    re.DOTALL,
)

def extract_think_blocks(gpt_response: str) -> List[str]:
    """提取 gpt 回复中所有 think 块的内容（不含标签本身）"""
    return _THINK_PATTERN.findall(gpt_response)

def build_annotation_prompt(think_content: str) -> str:
    """构建打标 prompt，输入为单个 <think> 块的内容（不含标签）"""
    mode_defs = get_mode_definitions_for_prompt()
    prompt = f"""You are an expert in cognitive science and reasoning analysis.

Your task is to annotate a reasoning paragraph extracted from an AI's thinking process.
The reasoning may contain multiple paragraphs. For each paragraph that clearly belongs to one of the 8 cognitive thinking modes below, wrap that paragraph with the corresponding XML tag.

## 8 Cognitive Thinking Modes:
{mode_defs}

## Rules:
1. Annotate at paragraph granularity (one complete reasoning step per tag).
2. A paragraph may have AT MOST ONE cognitive mode tag.
3. Not every paragraph needs a tag — only annotate when the mode is CLEARLY identifiable.
4. Tags must NOT overlap or nest with each other.
5. Do NOT add, remove, or modify any text — only wrap existing paragraphs with tags.
6. The output must contain EXACTLY the same text as the input, just with optional wrapping tags added.
7. Do NOT annotate tool-related content (search queries, search results, python code, final answers).

## Input reasoning content:
---
{think_content}
---

## Output:
Return ONLY the annotated reasoning content (same text with optional cognitive mode tags added). Do not include any explanation or preamble.
"""
    return prompt


def reassemble_response(original_response: str, annotated_think_blocks: List[str]) -> Optional[str]:
    """将打标后的 think 块内容替换回原始回复中"""
    result = original_response
    think_matches = list(_THINK_PATTERN.finditer(result))

    if len(think_matches) != len(annotated_think_blocks):
        return None

    # 从后往前替换，避免字符偏移量变化
    for match, annotated_content in zip(reversed(think_matches), reversed(annotated_think_blocks)):
        group_start, group_end = match.span(1)  # 捕获组 = think 内容部分
        result = result[:group_start] + annotated_content + result[group_end:]

    return result


def validate_annotation(original_think: str, annotated_think: str) -> bool:
    """
    验证打标结果的合法性：
    1. 去掉所有认知标记后，文本内容与原始完全一致
    2. 没有非法标签
    3. 认知标签没有嵌套
    """
    all_tags_pattern = "|".join(re.escape(t) for t in VALID_TAGS)

    # 检查是否有非法标签（不在允许列表里的小写字母标签）
    any_open_tag_pattern = re.compile(r"<([a-z_]+)>")
    for match in any_open_tag_pattern.finditer(annotated_think):
        tag_name = match.group(1)
        if tag_name not in ALL_ALLOWED_TAGS:
            return False

    # 检查认知标签嵌套：认知标签内部不能再有认知标签
    cognitive_tag_pattern = re.compile(
        rf"<({all_tags_pattern})>(.*?)</\1>", re.DOTALL
    )
    inner_cognitive_tag_pattern = re.compile(rf"<(?:{all_tags_pattern})>")
    for match in cognitive_tag_pattern.finditer(annotated_think):
        inner_content = match.group(2)
        if inner_cognitive_tag_pattern.search(inner_content):
            return False

    # 去掉所有认知标签后，文本应与原始一致（允许空白微小差异）
    stripped_annotated = re.sub(rf"</?(?:{all_tags_pattern})>", "", annotated_think)
    stripped_original = re.sub(rf"</?(?:{all_tags_pattern})>", "", original_think)

    # 忽略空白差异：将连续空白统一为单个空格后再比对
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    if normalize_whitespace(stripped_annotated) != normalize_whitespace(stripped_original):
        return False

    return True


def annotate_single_think_block(
    think_content: str,
    sample_idx: int,
    block_idx: int,
    vote_idx: int,
) -> Optional[str]:
    """调用 GPT-4o 对单个 think 块打标，返回打标后的内容"""
    if len(think_content.strip()) < 50:
        # 太短的 think 块不打标，直接保留原文
        return think_content

    prompt = build_annotation_prompt(think_content)
    try:
        url = GPT_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GPT_API_KEY}",
        }
        payload = {
            "model": GPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": PHASE1_TEMPERATURE,
            # Keep completion budget conservative for gateway compatibility.
            "max_tokens": GPT_MAX_TOKENS,
        }
        http_response = requests.post(url, headers=headers, json=payload, timeout=120)
        if http_response.status_code >= 400:
            error_text = ""
            try:
                error_obj = http_response.json()
                error_text = json.dumps(error_obj, ensure_ascii=False)
            except Exception:
                error_text = http_response.text
            print(
                f"  [S{sample_idx} V{vote_idx} B{block_idx}] "
                f"HTTP {http_response.status_code}: {error_text[:500]}"
            )
            return None
        data = http_response.json()
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not raw_content:
            print(f"  [S{sample_idx} V{vote_idx} B{block_idx}] Empty response, using original")
            return think_content
        annotated_content = raw_content.strip()
        # 修复 GPT 输出末尾截断的闭合标签，如 </verify 缺少 >
        annotated_content = re.sub(r"</([a-z_]+)\s*$", r"</\1>", annotated_content, flags=re.MULTILINE)

        if not validate_annotation(think_content, annotated_content):
            print(
                f"  [S{sample_idx} V{vote_idx} B{block_idx}] Validation failed, using original"
            )
            print(annotated_content)
            return None
        else:
            print("pass")

        return annotated_content

    except Exception as error:
        print(f"  [S{sample_idx} V{vote_idx} B{block_idx}] GPT error: {error}")
        return None


def annotate_all_think_blocks(
    think_blocks: List[str],
    sample_idx: int,
    vote_idx: int,
) -> Optional[List[str]]:
    """对一条样本的所有 think 块打标，任意一块失败则返回 None"""
    annotated_blocks = []
    for block_idx, think_content in enumerate(think_blocks):
        result = annotate_single_think_block(
            think_content, sample_idx, block_idx, vote_idx
        )
        if result is None:
            return None
        annotated_blocks.append(result)
        time.sleep(0.3)  # 避免 API 限速
    return annotated_blocks


def normalize_for_comparison(text: str) -> str:
    """标准化文本用于多数投票比较（去除空白差异）"""
    return re.sub(r"\s+", " ", text).strip()


def process_single_sample(
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
        # 没有 think 块，直接保留原样本（不打标）
        result = dict(sample)
        result["_annotated"] = False
        result["_vote_consistent"] = True
        return result

    vote_responses = []
    for vote_idx in range(PHASE1_NUM_VOTES):
        annotated_blocks = annotate_all_think_blocks(
            think_blocks, sample_idx, vote_idx
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

    # 多数投票：PHASE1_NUM_VOTES 次结果必须完全一致
    normalized_votes = [normalize_for_comparison(r) for r in vote_responses]
    if len(set(normalized_votes)) == 1:
        # 全部一致，使用第一次的结果
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


def load_samples(jsonl_path: str, sample_size: int, seed: int = 42) -> List[Dict]:
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
    print(f"=== Phase 1: GPT-4o Annotation ===")
    print(f"Sample size: {PHASE1_SAMPLE_SIZE}, Votes per sample: {PHASE1_NUM_VOTES}")

    samples = load_samples(INPUT_JSONL, PHASE1_SAMPLE_SIZE)

    os.makedirs(PHASE1_DIR, exist_ok=True)
    total_count = 0
    consistent_count = 0

    # 打开两个文件，边处理边追加写入，不等全部完成
    with open(PHASE1_ANNOTATED, "w", encoding="utf-8") as f_all, \
         open(PHASE1_FILTERED, "w", encoding="utf-8") as f_filtered:

        for idx, sample in enumerate(samples):
            print(f"Processing sample {idx + 1}/{len(samples)}...")
            result = process_single_sample(sample, idx)
            if result is None:
                continue

            total_count += 1
            # 立即写入全量结果文件（含内部标记字段）
            f_all.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_all.flush()

            if result.get("_vote_consistent"):
                consistent_count += 1
                # 立即写入种子数据文件（清理内部标记字段）
                clean_item = {k: v for k, v in result.items() if not k.startswith("_")}
                f_filtered.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                f_filtered.flush()
                print(f"  [Sample {idx}] ✓ Saved to seed data ({consistent_count} so far)")

    print(f"\n=== Phase 1 Complete ===")
    print(f"Total processed: {len(samples)}")
    print(f"Vote-consistent seed samples: {consistent_count}")
    print(f"Consistency rate: {consistent_count / len(samples) * 100:.1f}%")
    print(f"Seed data saved to: {PHASE1_FILTERED}")



if __name__ == "__main__":
    run_phase1()
