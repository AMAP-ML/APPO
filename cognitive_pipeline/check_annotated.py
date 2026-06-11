"""
检验 annotated_full.jsonl 里的合法标记轨迹数量
"""
import json
import re
from collections import defaultdict

VALID_TAGS = ["quick", "reason", "hypotest", "transfer", "interaction", "meta"]
DATA_PATH = "/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_qwen/annotated_full.jsonl"

total = 0
has_any_tag = 0
tag_counts = defaultdict(int)
samples_per_tag = defaultdict(int)

with open(DATA_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        sample = json.loads(line)
        text = ""
        for conv in sample.get("conversations", []):
            if conv.get("from") == "gpt":
                text += conv.get("value", "")

        found_tags = set()
        for tag in VALID_TAGS:
            count = len(re.findall(rf"<{tag}>", text))
            if count > 0:
                tag_counts[tag] += count
                found_tags.add(tag)

        if found_tags:
            has_any_tag += 1
            for tag in found_tags:
                samples_per_tag[tag] += 1

print(f"总条数:           {total}")
print(f"含合法标签的轨迹:  {has_any_tag}  ({has_any_tag/total*100:.1f}%)")
print(f"无标签轨迹:        {total - has_any_tag}  ({(total-has_any_tag)/total*100:.1f}%)")
print()
print("各标签出现次数（tag 实例总数）：")
for tag in VALID_TAGS:
    print(f"  <{tag}>: {tag_counts[tag]} 次，出现在 {samples_per_tag[tag]} 条样本中")
