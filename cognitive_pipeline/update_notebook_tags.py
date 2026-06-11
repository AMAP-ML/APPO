"""
更新 analyze_phase1.ipynb：
1. 把 COGNITIVE_TAGS 更新为7类
2. 把数据路径改成 annotated_full.jsonl
3. 更新 TAG_COLORS 里的标签名
"""
import json

nb_path = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/analyze_phase1.ipynb'
with open(nb_path) as f:
    nb = json.load(f)

# 所有需要替换的字符串对 (old, new)，new 为空字符串表示删除该行
replacements = [
    # COGNITIVE_TAGS 从各种旧版本更新到7类
    (
        "['fast', 'deduce', 'inductive', 'analogy', 'verify', 'practice', 'reflect', 'clarify']",
        "['quick', 'plan', 'deduce', 'induce', 'hypothesis', 'interaction', 'meta']"
    ),
    (
        "['quick', 'reason', 'hypotest', 'transfer', 'interaction', 'meta']",
        "['quick', 'plan', 'deduce', 'induce', 'hypothesis', 'interaction', 'meta']"
    ),
    # 数据路径更新
    (
        '/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase1_gpt/filtered_seed.jsonl',
        '/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_qwen/annotated_full.jsonl'
    ),
    (
        "DATA_PATH = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase1_gpt/filtered_seed.jsonl'",
        "DATA_PATH = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_qwen/annotated_full.jsonl'"
    ),
    # TAG_COLORS：旧8类 → 新7类
    (
        "'fast':      '#e74c3c'",
        "'quick':     '#e74c3c'"
    ),
    (
        "'deduce':    '#3498db'",
        "'deduce':    '#3498db'"  # 保持不变
    ),
    (
        "'inductive': '#2ecc71'",
        "'induce':    '#2ecc71'"
    ),
    (
        "'analogy':   '#9b59b6'",
        "'plan':      '#9b59b6'"
    ),
    (
        "'verify':    '#f39c12'",
        "'hypothesis':'#f39c12'"
    ),
    (
        "'practice':  '#1abc9c'",
        "'interaction':'#1abc9c'"
    ),
    (
        "'reflect':   '#e67e22'",
        "'meta':      '#e67e22'"
    ),
    (
        "'clarify':   '#34495e'",
        ""  # 删除（clarify 已合并到 meta）
    ),
    # TAG_COLORS：旧6类 → 新7类
    (
        "'quick':     '#e74c3c'",
        "'quick':     '#e74c3c'"  # 保持不变
    ),
    (
        "'reason':    '#3498db'",
        "'reason':    '#3498db'"  # 会在下面被进一步处理
    ),
    (
        "'hypotest':  '#2ecc71'",
        "'hypothesis':'#2ecc71'"
    ),
    (
        "'transfer':  '#9b59b6'",
        "'plan':      '#9b59b6'"
    ),
    (
        "'interaction': '#f39c12'",
        "'interaction':'#f39c12'"  # 保持不变
    ),
    (
        "'meta':      '#1abc9c'",
        "'meta':      '#1abc9c'"  # 保持不变
    ),
    # reason 标签名更新
    (
        "'reason':    '#3498db'",
        "'deduce':    '#3498db'"
    ),
]

changed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_source = []
    for line in cell['source']:
        new_line = line
        for old, new in replacements:
            if old and old in new_line:
                if new == "":
                    new_line = None
                    changed += 1
                    break
                else:
                    new_line = new_line.replace(old, new)
                    changed += 1
        if new_line is not None:
            new_source.append(new_line)
    cell['source'] = new_source

with open(nb_path, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f'修改了 {changed} 处，notebook 已保存')
