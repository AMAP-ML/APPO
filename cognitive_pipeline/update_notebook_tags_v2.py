"""
更新 analyze_phase1.ipynb：
1. 把 COGNITIVE_TAGS 更新为7类（quick/plan/deduce/induce/hypothesis/interaction/meta）
2. 把数据路径改成 annotated_full.jsonl
3. 更新 TAG_COLORS 里的标签名
"""
import json
import re

nb_path = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/analyze_phase1.ipynb'
with open(nb_path) as f:
    nb = json.load(f)

NEW_TAGS = "['quick', 'plan', 'deduce', 'induce', 'hypothesis', 'interaction', 'meta']"
NEW_DATA_PATH = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_qwen/annotated_full.jsonl'

# TAG_COLORS 新7类配色
NEW_TAG_COLORS = (
    "TAG_COLORS = {\n"
    "    'quick':      '#e74c3c',\n"
    "    'plan':       '#9b59b6',\n"
    "    'deduce':     '#3498db',\n"
    "    'induce':     '#2ecc71',\n"
    "    'hypothesis': '#f39c12',\n"
    "    'interaction':'#1abc9c',\n"
    "    'meta':       '#e67e22',\n"
    "}\n"
)

# 旧 TAG_COLORS 的正则（匹配整个 TAG_COLORS = { ... } 块）
TAG_COLORS_PATTERN = re.compile(
    r"TAG_COLORS\s*=\s*\{[^}]*\}", re.DOTALL
)

# COGNITIVE_TAGS 各种旧版本
OLD_TAGS_PATTERNS = [
    "['fast', 'deduce', 'inductive', 'analogy', 'verify', 'practice', 'reflect', 'clarify']",
    "['quick', 'reason', 'hypotest', 'transfer', 'interaction', 'meta']",
    "['quick', 'plan', 'deduce', 'induce', 'hypothesis', 'interaction', 'meta']",  # 已是新版，保持
]

OLD_DATA_PATHS = [
    '/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase1_gpt/filtered_seed.jsonl',
]

changed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source_text = ''.join(cell['source'])

    # 更新 COGNITIVE_TAGS
    for old_tags in OLD_TAGS_PATTERNS:
        if old_tags in source_text and old_tags != NEW_TAGS:
            source_text = source_text.replace(old_tags, NEW_TAGS)
            changed += 1

    # 更新数据路径
    for old_path in OLD_DATA_PATHS:
        if old_path in source_text:
            source_text = source_text.replace(old_path, NEW_DATA_PATH)
            changed += 1

    # 更新 TAG_COLORS 块
    if 'TAG_COLORS' in source_text and TAG_COLORS_PATTERN.search(source_text):
        new_text = TAG_COLORS_PATTERN.sub(NEW_TAG_COLORS.rstrip('\n'), source_text)
        if new_text != source_text:
            source_text = new_text
            changed += 1

    # 把修改后的文本重新拆成 source 列表
    lines = source_text.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(nb_path, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f'修改了 {changed} 处，notebook 已保存')
