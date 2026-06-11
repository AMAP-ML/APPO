"""
Phase 3: 构建最终训练数据集。

功能：
  1. 读取 Phase 2 打标结果
  2. 改写 system prompt（追加认知模式说明）
  3. 统计打标质量（各标签使用频次）
  4. 输出 LLaMA-Factory 格式的最终 JSONL

运行方式：
  python phase3_build_dataset.py
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    COGNITIVE_MODES,
    COGNITIVE_SYSTEM_PROMPT_ADDON,
    FINAL_OUTPUT,
    OUTPUT_DIR,
    PHASE2_ANNOTATED,
)

# XML 标签辅助（用 chr 拼接避免工具解析问题）
_LT = chr(60)
_GT = chr(62)

VALID_TAGS = list(COGNITIVE_MODES.keys())


# ── 工具函数 ──────────────────────────────────────────────

def count_tag_usage(text: str) -> Counter:
    """统计文本中各认知标签的使用次数。"""
    tag_counter = Counter()
    for tag_name in VALID_TAGS:
        open_tag = _LT + tag_name + _GT
        tag_counter[tag_name] = text.count(open_tag)
    return tag_counter


def has_any_annotation(text: str) -> bool:
    """判断文本中是否有任何认知标签。"""
    for tag_name in VALID_TAGS:
        open_tag = _LT + tag_name + _GT
        if open_tag in text:
            return True
    return False


def rewrite_system_prompt(original_system: str) -> str:
    """在原有 system prompt 末尾追加认知模式说明。"""
    if not original_system:
        return COGNITIVE_SYSTEM_PROMPT_ADDON.strip()
    return original_system.rstrip() + "\n" + COGNITIVE_SYSTEM_PROMPT_ADDON


def process_sample(sample: dict) -> dict:
    """处理单条样本：改写 system prompt，保持其余字段不变。"""
    result = dict(sample)
    original_system = sample.get("system", "")
    result["system"] = rewrite_system_prompt(original_system)
    return result


# ── 主流程 ────────────────────────────────────────────────

def run_phase3():
    if not os.path.exists(PHASE2_ANNOTATED):
        print(f"ERROR: Phase 2 output not found: {PHASE2_ANNOTATED}")
        print("Please run phase2_qwen_annotate.py first.")
        sys.exit(1)

    print("=== Phase 3: Build Final Dataset ===")
    print(f"Input:  {PHASE2_ANNOTATED}")
    print(f"Output: {FINAL_OUTPUT}")

    total_count = 0
    annotated_count = 0
    global_tag_counter: Counter = Counter()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PHASE2_ANNOTATED, "r", encoding="utf-8") as in_f, \
         open(FINAL_OUTPUT, "w", encoding="utf-8") as out_f:

        for line in in_f:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            processed = process_sample(sample)
            out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")

            # 统计打标情况（基于 gpt 回复）
            conversations = sample.get("conversations", [])
            gpt_value = (
                conversations[1].get("value", "")
                if len(conversations) > 1
                else ""
            )
            if has_any_annotation(gpt_value):
                annotated_count += 1
                global_tag_counter.update(count_tag_usage(gpt_value))

            total_count += 1
            if total_count % 5000 == 0:
                print(f"Processed {total_count} samples...")

    annotation_rate = annotated_count / total_count * 100 if total_count > 0 else 0

    print(f"\n=== Phase 3 Complete ===")
    print(f"Total samples:      {total_count}")
    print(f"Annotated samples:  {annotated_count} ({annotation_rate:.1f}%)")
    print(f"Unannotated:        {total_count - annotated_count}")
    print(f"\nTag usage statistics:")
    for tag_name in VALID_TAGS:
        count = global_tag_counter.get(tag_name, 0)
        bar = "#" * min(count // 100, 50)
        print(f"  {_LT}{tag_name}{_GT}: {count:>6}  {bar}")
    print(f"\nFinal dataset saved to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    run_phase3()
