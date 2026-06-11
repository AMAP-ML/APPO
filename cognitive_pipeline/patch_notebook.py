"""
修改 analyze_phase1.ipynb：
1. 把 rollout_data 改成 rollout_list（支持3条 rollout）
2. 把图5的 cell 改成循环展示3条 rollout 的熵曲线
"""
import json

nb_path = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/analyze_phase1.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Step 1：修改数据加载 cell，把 rollout_data -> rollout_list ──────────────
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = cell['source']
    new_source = []
    changed = False
    for line in source:
        if "rollout     = entropy_results['rollout_data']" in line:
            new_source.append("rollout_list = entropy_results.get('rollout_list', [])\n")
            new_source.append("# 兼容旧格式（rollout_data 单条）\n")
            new_source.append("if not rollout_list and 'rollout_data' in entropy_results:\n")
            new_source.append("    rollout_list = [entropy_results['rollout_data']]\n")
            changed = True
            print("Step1: 替换 rollout_data -> rollout_list")
        else:
            new_source.append(line)
    if changed:
        cell['source'] = new_source

# ── Step 2：找到图5的 cell，替换成支持3条 rollout 的新版本 ────────────────
NEW_CELL_SOURCE = [
    "# ============================================================\n",
    "# 图5：3条 Rollout 的逐 Token 熵曲线（每条独立一张图）\n",
    "# 每个思维模式标签只标记第一个 token 位置（避免多点问题）\n",
    "# 只展示 think 块内的 token 范围，让图更聚焦\n",
    "# ============================================================\n",
    "\n",
    "TAG_COLORS = {\n",
    "    'fast':      '#e74c3c',\n",
    "    'deduce':    '#3498db',\n",
    "    'inductive': '#2ecc71',\n",
    "    'analogy':   '#9b59b6',\n",
    "    'verify':    '#f39c12',\n",
    "    'practice':  '#1abc9c',\n",
    "    'reflect':   '#e67e22',\n",
    "    'clarify':   '#34495e',\n",
    "}\n",
    "\n",
    "def plot_rollout_entropy(rollout, rollout_idx, global_mean, COGNITIVE_TAGS, TAG_COLORS):\n",
    "    \"\"\"绘制单条 rollout 的逐 token 熵曲线\"\"\"\n",
    "    entropies = np.array(rollout['entropies'])\n",
    "    tag_mask  = rollout['tag_mask']\n",
    "    seq_len   = rollout['seq_len']\n",
    "    token_strings = rollout['token_strings']\n",
    "\n",
    "    # 找每个标签出现的第一个 token 位置（连续相同标签只取第一个）\n",
    "    tag_first_positions = {tag: [] for tag in COGNITIVE_TAGS}\n",
    "    prev_tag = None\n",
    "    for pos, tag_name in enumerate(tag_mask):\n",
    "        if tag_name != 'none' and tag_name in tag_first_positions:\n",
    "            if tag_name != prev_tag:\n",
    "                tag_first_positions[tag_name].append(pos)\n",
    "        prev_tag = tag_name if tag_name != 'none' else prev_tag\n",
    "\n",
    "    # 找 think 块的范围\n",
    "    think_start, think_end = 0, seq_len\n",
    "    for i, tok in enumerate(token_strings):\n",
    "        if '' in token_strings[i] or '/think>' in token_strings[i]:\n",
    "            think_end = i\n",
    "            break\n",
    "\n",
    "    # 如果找不到 think 块，以所有标签位置为中心取前后 300 个 token\n",
    "    all_tag_positions = [p for positions in tag_first_positions.values() for p in positions]\n",
    "    if not all_tag_positions:\n",
    "        think_start, think_end = 0, min(500, seq_len)\n",
    "    elif think_start == 0 and think_end == seq_len:\n",
    "        center = (min(all_tag_positions) + max(all_tag_positions)) // 2\n",
    "        think_start = max(0, center - 300)\n",
    "        think_end = min(seq_len, center + 300)\n",
    "\n",
    "    # 截取 think 块范围内的数据\n",
    "    plot_entropies = entropies[think_start:think_end]\n",
    "    plot_len = len(plot_entropies)\n",
    "    x_offset = think_start\n",
    "\n",
    "    # 平滑曲线（滑动平均，窗口=15）\n",
    "    window = 15\n",
    "    smoothed = np.convolve(plot_entropies, np.ones(window)/window, mode='same')\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(20, 4))\n",
    "\n",
    "    ax.plot(range(plot_len), plot_entropies, color='lightsteelblue', alpha=0.4, linewidth=0.6, label='Token entropy')\n",
    "    ax.plot(range(plot_len), smoothed, color='steelblue', alpha=0.9, linewidth=1.5, label=f'Smoothed (window={window})')\n",
    "    ax.axhline(global_mean, color='gray', linestyle='--', linewidth=1.2, label=f'Global mean ({global_mean:.2f})')\n",
    "\n",
    "    # 高亮思维模式标签位置（每个标签只画一个点 + 一条竖线）\n",
    "    y_max = max(plot_entropies.max() * 1.05, global_mean * 1.5)\n",
    "    for tag, positions in tag_first_positions.items():\n",
    "        for pos in positions:\n",
    "            plot_pos = pos - x_offset\n",
    "            if 0 <= plot_pos < plot_len:\n",
    "                ax.axvline(plot_pos, color=TAG_COLORS[tag], alpha=0.7, linewidth=2.0, linestyle='-')\n",
    "                ax.scatter([plot_pos], [entropies[pos]], color=TAG_COLORS[tag], s=80, zorder=6,\n",
    "                          edgecolors='white', linewidths=0.8)\n",
    "                ax.text(plot_pos, y_max * 0.95, f'<{tag}>',\n",
    "                       color=TAG_COLORS[tag], fontsize=7, ha='center', va='top',\n",
    "                       rotation=90, alpha=0.9)\n",
    "\n",
    "    # 图例\n",
    "    tag_patches = [\n",
    "        mpatches.Patch(color=TAG_COLORS[tag], label=f'<{tag}>')\n",
    "        for tag in COGNITIVE_TAGS\n",
    "        if tag_first_positions[tag]\n",
    "    ]\n",
    "    legend1 = ax.legend(handles=tag_patches, loc='upper right', fontsize=8, title='Cognitive Tags')\n",
    "    ax.add_artist(legend1)\n",
    "    ax.legend(loc='upper left', fontsize=8)\n",
    "\n",
    "    sample_idx = rollout.get('sample_idx', '?')\n",
    "    ax.set_title(\n",
    "        f'Rollout #{rollout_idx+1} (sample={sample_idx})  '\n",
    "        f'Token entropy curve  [tokens {think_start}~{think_end} / total {seq_len}]',\n",
    "        fontsize=11\n",
    "    )\n",
    "    ax.set_xlabel('Token position (within think block)')\n",
    "    ax.set_ylabel('Entropy (nats)')\n",
    "    ax.set_xlim(0, plot_len)\n",
    "    ax.set_ylim(0, y_max)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "    # 打印标签位置的熵值\n",
    "    print(f\"Rollout #{rollout_idx+1} 标签 token 熵值（每个标签只取第一个 token）：\")\n",
    "    for tag, positions in tag_first_positions.items():\n",
    "        if positions:\n",
    "            tag_entropies_at_pos = [entropies[p] for p in positions if p < len(entropies)]\n",
    "            ratio = np.mean(tag_entropies_at_pos) / global_mean if global_mean > 0 else 0\n",
    "            print(f\"  <{tag:10}> 位置={positions}  \"\n",
    "                  f\"熵={[f'{v:.3f}' for v in tag_entropies_at_pos]}  \"\n",
    "                  f\"均值={np.mean(tag_entropies_at_pos):.3f}  \"\n",
    "                  f\"全局均值={global_mean:.3f}  倍率={ratio:.2f}x\")\n",
    "    print()\n",
    "\n",
    "\n",
    "# 依次绘制3条 rollout\n",
    "for rollout_idx, rollout in enumerate(rollout_list):\n",
    "    plot_rollout_entropy(rollout, rollout_idx, global_mean, COGNITIVE_TAGS, TAG_COLORS)\n",
]

# 找到图5的 cell（包含 "图5" 或 "Rollout 的逐 Token 熵曲线" 的 cell）
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source_text = ''.join(cell['source'])
    if '图5' in source_text or 'Rollout 的逐 Token 熵曲线' in source_text:
        cell['source'] = NEW_CELL_SOURCE
        print(f"Step2: 已替换图5 cell（第 {i} 个 cell）")
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("notebook 修改完成！")
