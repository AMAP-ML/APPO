"""
最终数据集构建：
1. 合并 Phase 1 种子数据 + Phase 2 全量打标数据
2. 对 system prompt 追加 8 种思维模式定义
3. 输出 LLaMA-Factory 格式的 JSONL 文件
4. 同时生成 LLaMA-Factory 的 dataset_info.json 注册条目
"""

import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    COGNITIVE_MODES,
    COGNITIVE_SYSTEM_PROMPT_ADDON,
    FINAL_OUTPUT,
    OUTPUT_DIR,
    PHASE1_FILTERED,
    PHASE2_ANNOTATED,
)

VALID_TAGS = set(COGNITIVE_MODES.keys())


def count_annotations(gpt_response: str) -> Counter:
    """统计一条样本中各思维模式标签的出现次数"""
    counts = Counter()
    for tag in VALID_TAGS:
        pattern = re.compile(rf"<{tag}>", re.DOTALL)
        counts[tag] = len(pattern.findall(gpt_response))
    return counts


def augment_system_prompt(original_system: str) -> str:
    """在原有 system prompt 后追加思维模式定义"""
    if not original_system:
        return COGNITIVE_SYSTEM_PROMPT_ADDON.strip()
    return original_system + COGNITIVE_SYSTEM_PROMPT_ADDON


def process_sample(sample: dict) -> dict:
    """处理单条样本：更新 system prompt"""
    result = dict(sample)
    original_system = sample.get("system", "")
    result["system"] = augment_system_prompt(original_system)
    return result


def load_jsonl(path: str) -> list:
    """加载 JSONL 文件"""
    samples = []
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def deduplicate_by_question(samples: list) -> list:
    """
    按问题内容去重：Phase 1 的种子数据和 Phase 2 的全量数据可能有重叠。
    优先保留 Phase 1 的高质量版本（Phase 1 先加载）。
    """
    seen_questions = set()
    deduped = []
    for sample in samples:
        convs = sample.get("conversations", [])
        if not convs:
            continue
        question = convs[0].get("value", "").strip()
        if question not in seen_questions:
            seen_questions.add(question)
            deduped.append(sample)
    return deduped


def run_build():
    print("=== Building Final Dataset ===")

    # 加载 Phase 1 种子数据（高质量，优先）
    phase1_samples = load_jsonl(PHASE1_FILTERED)
    print(f"Phase 1 seed samples: {len(phase1_samples)}")

    # 加载 Phase 2 全量打标数据
    phase2_samples = load_jsonl(PHASE2_ANNOTATED)
    print(f"Phase 2 annotated samples: {len(phase2_samples)}")

    # 合并：Phase 1 优先，Phase 2 补充（去重）
    all_samples = phase1_samples + phase2_samples
    deduped_samples = deduplicate_by_question(all_samples)
    print(f"After deduplication: {len(deduped_samples)} samples")

    # 统计打标覆盖率
    tag_counter = Counter()
    annotated_sample_count = 0
    for sample in deduped_samples:
        gpt_response = sample["conversations"][1]["value"]
        counts = count_annotations(gpt_response)
        total_tags = sum(counts.values())
        if total_tags > 0:
            annotated_sample_count += 1
            tag_counter.update(counts)

    print(f"\nAnnotation statistics:")
    print(f"  Samples with at least one annotation: {annotated_sample_count}/{len(deduped_samples)}")
    print(f"  Tag distribution:")
    for tag, count in tag_counter.most_common():
        mode_name = COGNITIVE_MODES[tag]["name"]
        print(f"    <{tag}> ({mode_name}): {count}")

    # 更新 system prompt 并输出
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_samples = [process_sample(s) for s in deduped_samples]

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nFinal dataset saved to: {FINAL_OUTPUT}")
    print(f"Total samples: {len(final_samples)}")

    # 生成 LLaMA-Factory dataset_info.json 注册条目
    dataset_name = "cognitive_sft_data"
    dataset_info_entry = {
        dataset_name: {
            "file_name": os.path.basename(FINAL_OUTPUT),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }

    dataset_info_path = os.path.join(OUTPUT_DIR, "dataset_info_entry.json")
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info_entry, f, indent=2, ensure_ascii=False)

    print(f"\nDataset info entry saved to: {dataset_info_path}")
    print(f"To use in LLaMA-Factory, add the entry in {dataset_info_path} to:")
    print(f"  /mnt/workspace/wxc/AERPO/LLaMA-Factory/data/dataset_info.json")
    print(f"And copy {FINAL_OUTPUT} to:")
    print(f"  /mnt/workspace/wxc/AERPO/LLaMA-Factory/data/{os.path.basename(FINAL_OUTPUT)}")


if __name__ == "__main__":
    run_build()
