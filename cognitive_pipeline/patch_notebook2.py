import json

nb_path = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/analyze_phase1.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

NEW_SOURCE = r"""TAG_COLORS = {
    'fast':      '#e74c3c',
    'deduce':    '#3498db',
    'inductive': '#2ecc71',
    'analogy':   '#9b59b6',
    'verify':    '#f39c12',
    'practice':  '#1abc9c',
    'reflect':   '#e67e22',
    'clarify':   '#34495e',
}

def plot_rollout_entropy(rollout, rollout_idx, global_mean, COGNITIVE_TAGS, TAG_COLORS):
    entropies = np.array(rollout['entropies'])
    tag_mask  = rollout['tag_mask']
    seq_len   = rollout['seq_len']
    token_strings = rollout['token_strings']

    # 找每个标签出现的第一个 token 位置（连续相同标签只取第一个）
    tag_first_positions = {tag: [] for tag in COGNITIVE_TAGS}
    prev_tag = None
    for pos, tag_name in enumerate(tag_mask):
        if tag_name != 'none' and tag_name in tag_first_positions:
            if tag_name != prev_tag:
                tag_first_positions[tag_name].append(pos)
        prev_tag = tag_name if tag_name != 'none' else prev_tag

    # 以所有标签位置为中心，取前后 400 个 token 的范围展示
    all_tag_positions = [p for positions in tag_first_positions.values() for p in positions]
    if all_tag_positions:
        center = (min(all_tag_positions) + max(all_tag_positions)) // 2
        think_start = max(0, center - 400)
        think_end   = min(seq_len, center + 400)
    else:
        think_start, think_end = 0, min(600, seq_len)

    plot_entropies = entropies[think_start:think_end]
    plot_len = len(plot_entropies)
    x_offset = think_start

    if plot_len == 0:
        print(f"Rollout #{rollout_idx+1}: 无有效 token 范围，跳过")
        return

    # 平滑曲线（滑动平均，窗口=15，但不超过序列长度）
    window = min(15, plot_len)
    smoothed = np.convolve(plot_entropies, np.ones(window)/window, mode='same')

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(range(plot_len), plot_entropies, color='lightsteelblue', alpha=0.4, linewidth=0.6, label='Token entropy')
    ax.plot(range(plot_len), smoothed, color='steelblue', alpha=0.9, linewidth=1.5, label=f'Smoothed (window={window})')
    ax.axhline(global_mean, color='gray', linestyle='--', linewidth=1.2, label=f'Global mean ({global_mean:.2f})')

    y_max = max(float(plot_entropies.max()) * 1.1, global_mean * 2.0, 1.0)

    for tag, positions in tag_first_positions.items():
        for pos in positions:
            plot_pos = pos - x_offset
            if 0 <= plot_pos < plot_len:
                ax.axvline(plot_pos, color=TAG_COLORS[tag], alpha=0.7, linewidth=2.0, linestyle='-')
                ax.scatter([plot_pos], [float(entropies[pos])], color=TAG_COLORS[tag], s=80, zorder=6,
                          edgecolors='white', linewidths=0.8)
                ax.text(plot_pos, y_max * 0.95, f'<{tag}>',
                       color=TAG_COLORS[tag], fontsize=7, ha='center', va='top',
                       rotation=90, alpha=0.9)

    tag_patches = [
        mpatches.Patch(color=TAG_COLORS[tag], label=f'<{tag}>')
        for tag in COGNITIVE_TAGS if tag_first_positions[tag]
    ]
    legend1 = ax.legend(handles=tag_patches, loc='upper right', fontsize=8, title='Cognitive Tags')
    ax.add_artist(legend1)
    ax.legend(loc='upper left', fontsize=8)

    sample_idx = rollout.get('sample_idx', '?')
    ax.set_title(
        f'Rollout #{rollout_idx+1} (sample={sample_idx})  Token entropy curve  '
        f'[tokens {think_start}~{think_end} / total {seq_len}]',
        fontsize=11
    )
    ax.set_xlabel('Token position (within displayed range)')
    ax.set_ylabel('Entropy (nats)')
    ax.set_xlim(0, plot_len)
    ax.set_ylim(0, y_max)
    plt.tight_layout()
    plt.show()

    print(f"Rollout #{rollout_idx+1} 标签 token 熵值（每个标签只取第一个 token）：")
    for tag, positions in tag_first_positions.items():
        if positions:
            tag_entropies_at_pos = [float(entropies[p]) for p in positions if p < len(entropies)]
            ratio = np.mean(tag_entropies_at_pos) / global_mean if global_mean > 0 else 0
            print(f"  <{tag:10}> 位置={positions}  "
                  f"熵={[f'{v:.3f}' for v in tag_entropies_at_pos]}  "
                  f"均值={np.mean(tag_entropies_at_pos):.3f}  "
                  f"全局均值={global_mean:.3f}  倍率={ratio:.2f}x")
    print()


# 依次绘制所有 rollout
for rollout_idx, rollout in enumerate(rollout_list):
    plot_rollout_entropy(rollout, rollout_idx, global_mean, COGNITIVE_TAGS, TAG_COLORS)
"""

# 把多行字符串转成 notebook source 格式（每行末尾加 \n，最后一行不加）
lines = NEW_SOURCE.split('\n')
source_lines = []
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        source_lines.append(line + '\n')
    else:
        if line:  # 最后一行非空才加
            source_lines.append(line)

# 找到图5的 cell 并替换
replaced = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source_text = ''.join(cell['source'])
    if 'plot_rollout_entropy' in source_text or ('rollout_list' in source_text and 'TAG_COLORS' in source_text):
        cell['source'] = source_lines
        print(f"已替换图5 cell（第 {i} 个 cell）")
        replaced = True
        break

if not replaced:
    print("未找到图5 cell，请检查")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("notebook 修改完成！")
