"""
Phase 1b：针对 Analogy（类比迁移）稀缺问题的定向补充打标脚本。

策略：
1. 从全量数据集中，用关键词筛选出 think 块里含有类比推理特征的样本
2. 用专门强化 Analogy 的 prompt 对这批样本打标（单次，不做多数投票）
3. 输出到 phase1_gpt/analogy_seed.jsonl，后续与 filtered_seed.jsonl 合并使用

目标：补充约 100 条高质量 Analogy 标注样本，缓解长尾分布问题。
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COGNITIVE_MODES,
    GPT_API_KEY,
    GPT_BASE_URL,
    GPT_MAX_TOKENS,
    GPT_MODEL,
    INPUT_JSONL,
    MAX_CHARS_PER_SAMPLE,
    PHASE1_DIR,
    PHASE1_TEMPERATURE,
    get_mode_definitions_for_prompt,
)

# ============================================================
# 常量
# ============================================================

ANALOGY_OUTPUT = os.path.join(PHASE1_DIR, "analogy_seed.jsonl")

# 目标采样数量
ANALOGY_TARGET_COUNT = 100

# think 标签常量（用 chr 拼接避免工具解析问题）
_LT = chr(60)
_GT = chr(62)
THINK_OPEN = _LT + "think" + _GT
THINK_CLOSE = _LT + "/think" + _GT
_THINK_PATTERN = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
    re.DOTALL,
)

VALID_TAGS = set(COGNITIVE_MODES.keys())
TOOL_TAGS = {"search", "result", "python", "answer", "think"}
ALL_ALLOWED_TAGS = VALID_TAGS | TOOL_TAGS

# 类比推理关键词（英文 + 中文，覆盖 think 块里的常见表达）
ANALOGY_KEYWORDS_EN = [
    r"\bsimilar to\b",
    r"\banalogous\b",
    r"\bjust like\b",
    r"\bsame as\b",
    r"\blike a\b",
    r"\bresembles\b",
    r"\bequivalent to\b",
    r"\bmirrors\b",
    r"\bparallel to\b",
    r"\bakin to\b",
    r"\bcorresponds to\b",
    r"\bby analogy\b",
    r"\bthink of it as\b",
    r"\bthis is like\b",
    r"\bthis works like\b",
    r"\bsame pattern\b",
    r"\bsame approach\b",
    r"\bsame idea\b",
    r"\bsame logic\b",
    r"\bsame principle\b",
    r"\bapply the same\b",
    r"\buse the same\b",
    r"\bfollows the same\b",
    r"\bsimilar pattern\b",
    r"\bsimilar approach\b",
    r"\bsimilar idea\b",
    r"\bsimilar logic\b",
    r"\bsimilar principle\b",
    r"\bclassic .{0,30} problem\b",
    r"\bstandard .{0,30} problem\b",
    r"\bthis is essentially\b",
    r"\bthis reduces to\b",
    r"\bthis is equivalent\b",
]

ANALOGY_KEYWORDS_ZH = [
    r"类似于",
    r"类比",
    r"就像",
    r"相当于",
    r"和.{0,10}一样",
    r"与.{0,10}类似",
    r"同样的思路",
    r"同样的方法",
    r"同样的逻辑",
    r"同样的原理",
    r"套用",
    r"借鉴",
    r"参照",
    r"对应于",
    r"映射到",
]

_ANALOGY_PATTERN = re.compile(
    "|".join(ANALOGY_KEYWORDS_EN + ANALOGY_KEYWORDS_ZH),
    re.IGNORECASE,
)


# ============================================================
# 工具函数
# ============================================================

def extract_think_blocks(gpt_response: str) -> List[str]:
    """提取回复中所有 think 块的内容（不含标签本身）"""
    return _THINK_PATTERN.findall(gpt_response)


def has_analogy_signal(think_blocks: List[str]) -> bool:
    """判断 think 块中是否含有类比推理关键词"""
    combined_text = "\n".join(think_blocks)
    return bool(_ANALOGY_PATTERN.search(combined_text))


def build_analogy_focused_prompt(think_content: str) -> str:
    """
    构建专门强化 Analogy 识别的打标 prompt。
    与通用 prompt 的区别：
    - 在 Rules 里额外强调 Analogy 的识别
    - 提供 2 个 Analogy few-shot 示例
    - 其他 7 种模式仍然正常打标（不遗漏）
    """
    mode_defs = get_mode_definitions_for_prompt()

    few_shot_examples = (
        "## Few-shot Examples for Analogy:\n\n"
        "Example 1 — Input paragraph:\n"
        "  This problem is essentially the same as the classic two-body orbital mechanics problem. "
        "I can apply Kepler's third law directly, just substituting the masses.\n"
        "Expected output:\n"
        f"  {_LT}analogy{_GT}This problem is essentially the same as the classic two-body orbital mechanics problem. "
        f"I can apply Kepler's third law directly, just substituting the masses.{_LT}/analogy{_GT}\n\n"
        "Example 2 — Input paragraph:\n"
        "  Think of the electric field lines like water flowing in a pipe — "
        "the flux through a closed surface follows the same conservation principle.\n"
        "Expected output:\n"
        f"  {_LT}analogy{_GT}Think of the electric field lines like water flowing in a pipe — "
        f"the flux through a closed surface follows the same conservation principle.{_LT}/analogy{_GT}\n\n"
        "Example 3 — Input paragraph (Chinese):\n"
        "  这道题和经典的背包问题思路完全一样，只需要把物品重量换成时间代价即可套用 DP 转移方程。\n"
        "Expected output:\n"
        f"  {_LT}analogy{_GT}这道题和经典的背包问题思路完全一样，只需要把物品重量换成时间代价即可套用 DP 转移方程。{_LT}/analogy{_GT}\n"
    )

    prompt = f"""You are an expert in cognitive science and reasoning analysis, with special focus on analogical reasoning.

Your task is to annotate a reasoning paragraph extracted from an AI's thinking process.
For each paragraph that clearly belongs to one of the 8 cognitive thinking modes below, wrap that paragraph with the corresponding XML tag.

## 8 Cognitive Thinking Modes:
{mode_defs}

## Special Attention — Analogy (analogical transfer):
Pay extra attention to the <analogy> tag. A paragraph should be tagged as <analogy> when it:
- Explicitly maps the current problem to a known problem/domain (e.g., "this is like X", "similar to Y", "same as Z")
- Draws a structural parallel between two situations
- Applies a solution from one domain to another
- Uses phrases like: "similar to", "just like", "same as", "analogous to", "equivalent to", "this reduces to", "classic X problem", "same pattern/approach/logic"

{few_shot_examples}

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


def validate_annotation(original_think: str, annotated_think: str) -> bool:
    """验证打标结果合法性"""
    all_tags_pattern = "|".join(re.escape(t) for t in VALID_TAGS)

    # 检查非法标签
    any_open_tag_pattern = re.compile(r"<([a-z_]+)>")
    for match in any_open_tag_pattern.finditer(annotated_think):
        tag_name = match.group(1)
        if tag_name not in ALL_ALLOWED_TAGS:
            return False

    # 检查认知标签嵌套
    cognitive_tag_pattern = re.compile(
        rf"<({all_tags_pattern})>(.*?)</\1>", re.DOTALL
    )
    inner_cognitive_tag_pattern = re.compile(rf"<(?:{all_tags_pattern})>")
    for match in cognitive_tag_pattern.finditer(annotated_think):
        inner_content = match.group(2)
        if inner_cognitive_tag_pattern.search(inner_content):
            return False

    # 去掉认知标签后文本应与原始一致
    stripped_annotated = re.sub(rf"</?(?:{all_tags_pattern})>", "", annotated_think)
    stripped_original = re.sub(rf"</?(?:{all_tags_pattern})>", "", original_think)

    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    if normalize_whitespace(stripped_annotated) != normalize_whitespace(stripped_original):
        return False

    return True


def has_analogy_tag(annotated_response: str) -> bool:
    """检查打标结果中是否真的出现了 <analogy> 标签"""
    return bool(re.search(r"<analogy>", annotated_response))


def annotate_think_block_with_analogy_focus(
    think_content: str,
    sample_idx: int,
    block_idx: int,
) -> Optional[str]:
    """调用 GPT-4o 对单个 think 块打标（Analogy 强化版 prompt）"""
    if len(think_content.strip()) < 50:
        return think_content

    prompt = build_analogy_focused_prompt(think_content)
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
                f"  [S{sample_idx} B{block_idx}] HTTP {http_response.status_code}: {error_text[:300]}"
            )
            return None

        data = http_response.json()
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not raw_content:
            print(f"  [S{sample_idx} B{block_idx}] Empty response, using original")
            return think_content

        annotated_content = raw_content.strip()
        # 修复末尾截断的闭合标签
        annotated_content = re.sub(
            r"</([a-z_]+)\s*$", r"</\1>", annotated_content, flags=re.MULTILINE
        )

        if not validate_annotation(think_content, annotated_content):
            print(f"  [S{sample_idx} B{block_idx}] Validation failed, skipping block")
            return None

        return annotated_content

    except Exception as error:
        print(f"  [S{sample_idx} B{block_idx}] GPT error: {error}")
        return None


def reassemble_response(original_response: str, annotated_think_blocks: List[str]) -> Optional[str]:
    """将打标后的 think 块内容替换回原始回复中"""
    result = original_response
    think_matches = list(_THINK_PATTERN.finditer(result))

    if len(think_matches) != len(annotated_think_blocks):
        return None

    for match, annotated_content in zip(reversed(think_matches), reversed(annotated_think_blocks)):
        group_start, group_end = match.span(1)
        result = result[:group_start] + annotated_content + result[group_end:]

    return result


# ============================================================
# 主流程
# ============================================================

def load_and_filter_analogy_samples(
    jsonl_path: str,
    target_count: int,
) -> List[Dict]:
    """
    从全量数据集中筛选含类比推理关键词的样本。
    过滤条件：
    1. 样本总长度不超过 MAX_CHARS_PER_SAMPLE
    2. think 块中含有类比推理关键词
    """
    print(f"Scanning {jsonl_path} for analogy signals...")
    candidate_samples = []
    total_scanned = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_scanned += 1

            obj = json.loads(line)
            convs = obj.get("conversations") or obj.get("messages", [])
            total_chars = sum(
                len(str(t.get("value") or t.get("content", ""))) for t in convs
            )
            if total_chars > MAX_CHARS_PER_SAMPLE:
                continue

            # 找 assistant 回复（第二条 conversation）
            if len(convs) < 2:
                continue
            gpt_response = convs[1].get("value") or convs[1].get("content", "")
            think_blocks = extract_think_blocks(gpt_response)

            if not think_blocks:
                continue

            if has_analogy_signal(think_blocks):
                candidate_samples.append(obj)

    print(f"Scanned {total_scanned} samples, found {len(candidate_samples)} with analogy signals")

    # 如果候选数量超过目标，随机截取（seed 改为 2025，避免与上次采样重复）
    if len(candidate_samples) > target_count:
        import random
        random.seed(2025)
        candidate_samples = random.sample(candidate_samples, target_count)
        print(f"Randomly sampled {target_count} from candidates")

    return candidate_samples


def run_analogy_targeted():
    """主函数：定向筛选 + Analogy 强化打标"""
    print("=== Phase 1b: Analogy-Targeted Annotation ===")
    print(f"Target: {ANALOGY_TARGET_COUNT} analogy-rich samples")
    print(f"Output: {ANALOGY_OUTPUT}")
    print()

    # 第一步：筛选含类比信号的样本
    samples = load_and_filter_analogy_samples(INPUT_JSONL, ANALOGY_TARGET_COUNT)
    if not samples:
        print("No analogy samples found. Exiting.")
        return

    os.makedirs(PHASE1_DIR, exist_ok=True)

    success_count = 0
    analogy_confirmed_count = 0  # 打标后真的出现了 <analogy> 标签的数量

    with open(ANALOGY_OUTPUT, "a", encoding="utf-8") as f_out:
        for idx, sample in enumerate(samples):
            print(f"Processing sample {idx + 1}/{len(samples)}...")

            convs = sample.get("conversations") or sample.get("messages", [])
            gpt_response = convs[1].get("value") or convs[1].get("content", "")
            think_blocks = extract_think_blocks(gpt_response)

            if not think_blocks:
                # 没有 think 块，直接保留（不打标）
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()
                success_count += 1
                continue

            # 对每个 think 块用 Analogy 强化 prompt 打标
            annotated_blocks = []
            all_blocks_ok = True
            for block_idx, think_content in enumerate(think_blocks):
                result = annotate_think_block_with_analogy_focus(
                    think_content, idx, block_idx
                )
                if result is None:
                    print(f"  [Sample {idx}] Block {block_idx} failed, skipping sample")
                    all_blocks_ok = False
                    break
                annotated_blocks.append(result)
                time.sleep(0.3)

            if not all_blocks_ok:
                continue

            # 重新组装回原始回复
            annotated_response = reassemble_response(gpt_response, annotated_blocks)
            if annotated_response is None:
                print(f"  [Sample {idx}] Reassemble failed, skipping")
                continue

            # 构建输出样本
            output_sample = dict(sample)
            output_sample["conversations"] = [
                dict(convs[0]),
                {**dict(convs[1]), "value": annotated_response},
            ]
            output_sample["_source"] = "analogy_targeted"

            # 统计是否真的打出了 <analogy> 标签
            if has_analogy_tag(annotated_response):
                analogy_confirmed_count += 1
                print(f"  [Sample {idx}] ✓ Analogy tag confirmed ({analogy_confirmed_count} so far)")
            else:
                print(f"  [Sample {idx}] ⚠ No analogy tag in output (keyword matched but GPT didn't tag)")

            f_out.write(json.dumps(output_sample, ensure_ascii=False) + "\n")
            f_out.flush()
            success_count += 1

            time.sleep(0.5)

    print(f"\n=== Phase 1b Complete ===")
    print(f"Total processed: {len(samples)}")
    print(f"Successfully annotated: {success_count}")
    print(f"Samples with confirmed <analogy> tag: {analogy_confirmed_count}")
    print(f"Analogy confirmation rate: {analogy_confirmed_count / max(success_count, 1) * 100:.1f}%")
    print(f"Output saved to: {ANALOGY_OUTPUT}")


if __name__ == "__main__":
    run_analogy_targeted()
