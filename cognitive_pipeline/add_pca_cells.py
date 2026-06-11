"""
在 analyze_phase1.ipynb 末尾追加两个 cell：
1. PCA 分析：用每条样本的标签频率向量做 PCA，验证8个维度的独立性
2. 层次聚类：用标签相关矩阵做层次聚类，验证标签间的语义分组
"""
import json

nb_path = '/mnt/workspace/wxc/AERPO/cognitive_pipeline/analyze_phase1.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Markdown 说明 cell ────────────────────────────────────────────────────────
MARKDOWN_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 思维模式标签的主成分分析（PCA）与层次聚类\n",
        "\n",
        "用每条样本中各思维模式标签的出现频率构建特征向量，通过 PCA 和层次聚类验证：\n",
        "1. **PCA**：8个维度是否真实独立，不是冗余的\n",
        "2. **层次聚类**：标签是否自然形成有语义意义的分组（如推理类、元认知类）"
    ]
}

# ── PCA + 层次聚类 code cell ──────────────────────────────────────────────────
PCA_CELL_SOURCE = """\
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import re
from collections import defaultdict

# ── 重新定义，保证本节 cell 可独立运行 ──────────────────────────────────────
COGNITIVE_TAGS = ['fast', 'deduce', 'inductive', 'analogy', 'verify', 'practice', 'reflect', 'clarify']
DATA_PATH = '/mnt/workspace/wxc/AERPO/LLaMA-Factory/data/final_5w4_still_filtered.jsonl'

TAG_COLORS = {
    'fast':      '#e74c3c',
    'deduce':    '#3498db',
    'inductive': '#2ecc71',
    'analogy':   '#9b59b6',
    'verify':    '#f39c12',
    'practice':  '#1abc9c',
    'reflect':   '#e67e22',
    'clarify':   '#34495e',
}

# ── 构建每条样本的标签频率特征向量 ──────────────────────────────────────────
def count_tags_in_sample(sample):
    text = ''
    for conv in sample.get('conversations', []):
        if conv.get('from') == 'gpt':
            text += conv.get('value', '')
    counts = {}
    for tag in COGNITIVE_TAGS:
        counts[tag] = len(re.findall(f'<{tag}>', text))
    return counts

sample_vectors = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        counts = count_tags_in_sample(sample)
        total = sum(counts.values())
        if total > 0:  # 只取有标签的样本
            vec = [counts[tag] / total for tag in COGNITIVE_TAGS]  # 归一化为频率
            sample_vectors.append(vec)

X = np.array(sample_vectors)
print(f"有效样本数（含至少一个标签）: {len(X)}")
print(f"特征维度: {X.shape[1]}（对应8个思维模式标签）")

# ── 图7：PCA 分析 ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=min(8, len(X)))
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# 子图1：解释方差比（碎石图）
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
axes[0].bar(range(1, len(explained_var)+1), explained_var * 100,
            color='steelblue', alpha=0.7, label='Individual')
axes[0].plot(range(1, len(cumulative_var)+1), cumulative_var * 100,
             'o-', color='tomato', linewidth=2, label='Cumulative')
axes[0].axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='80% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('PCA Scree Plot\\n(How many independent dimensions exist?)')
axes[0].legend()
axes[0].set_xticks(range(1, len(explained_var)+1))
for i, v in enumerate(explained_var):
    axes[0].text(i+1, v*100 + 0.5, f'{v*100:.1f}%', ha='center', fontsize=8)

# 子图2：PC1 vs PC2 散点图（按主要标签着色）
dominant_tag_idx = np.argmax(X, axis=1)
colors_per_sample = [list(TAG_COLORS.values())[i] for i in dominant_tag_idx]
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                          c=colors_per_sample, alpha=0.6, s=30, edgecolors='none')
axes[1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
axes[1].set_title('PCA: PC1 vs PC2\\n(colored by dominant tag in sample)')
tag_patches = [mpatches.Patch(color=TAG_COLORS[tag], label=tag) for tag in COGNITIVE_TAGS]
axes[1].legend(handles=tag_patches, fontsize=7, loc='best')

# 子图3：PCA 载荷图（每个标签在 PC1/PC2 上的载荷）
loadings = pca.components_[:2].T  # [8, 2]
for i, tag in enumerate(COGNITIVE_TAGS):
    axes[2].arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.02, head_length=0.02,
                  fc=TAG_COLORS[tag], ec=TAG_COLORS[tag], linewidth=2)
    axes[2].text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, f'<{tag}>',
                fontsize=9, color=TAG_COLORS[tag], ha='center', va='center', fontweight='bold')
axes[2].axhline(0, color='gray', linewidth=0.5)
axes[2].axvline(0, color='gray', linewidth=0.5)
axes[2].set_xlim(-1.2, 1.2)
axes[2].set_ylim(-1.2, 1.2)
axes[2].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
axes[2].set_title('PCA Loading Plot\\n(tag contributions to PC1 & PC2)')
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1)
axes[2].add_patch(circle)
axes[2].set_aspect('equal')

plt.suptitle('Figure 7: PCA Analysis of Cognitive Tag Frequency Vectors', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\\nPCA 解释方差摘要：")
for i, v in enumerate(explained_var):
    print(f"  PC{i+1}: {v*100:.2f}%  (累计: {cumulative_var[i]*100:.2f}%)")
n_components_80 = np.argmax(cumulative_var >= 0.8) + 1
print(f"\\n达到 80% 解释方差需要 {n_components_80} 个主成分（共8个标签维度）")
print(f"→ 说明8个标签维度中存在 {n_components_80} 个主要独立方向，标签间具有真实的多样性")

# ── 图8：标签相关矩阵 + 层次聚类树状图 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 子图1：标签频率的相关矩阵热图
corr_matrix = np.corrcoef(X.T)  # [8, 8]
im = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(COGNITIVE_TAGS)))
axes[0].set_yticks(range(len(COGNITIVE_TAGS)))
axes[0].set_xticklabels([f'<{t}>' for t in COGNITIVE_TAGS], rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels([f'<{t}>' for t in COGNITIVE_TAGS], fontsize=9)
for i in range(len(COGNITIVE_TAGS)):
    for j in range(len(COGNITIVE_TAGS)):
        axes[0].text(j, i, f'{corr_matrix[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=axes[0], shrink=0.8)
axes[0].set_title('Tag Frequency Correlation Matrix\\n(negative = complementary, positive = co-occurring)')

# 子图2：层次聚类树状图
distance_matrix = 1 - np.abs(corr_matrix)  # 相关性越高，距离越小
condensed_dist = squareform(distance_matrix, checks=False)
linkage_matrix = linkage(condensed_dist, method='ward')
tag_labels = [f'<{t}>' for t in COGNITIVE_TAGS]
dendrogram(linkage_matrix, labels=tag_labels, ax=axes[1],
           leaf_font_size=10, color_threshold=0.6,
           link_color_func=lambda k: 'steelblue')
axes[1].set_title('Hierarchical Clustering of Cognitive Tags\\n(based on co-occurrence patterns)')
axes[1].set_ylabel('Distance (1 - |correlation|)')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Figure 8: Tag Correlation & Hierarchical Clustering', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\n层次聚类结果解读：")
print("  - 距离近的标签在样本中倾向于共同出现（功能相似或互补）")
print("  - 距离远的标签在样本中独立出现（覆盖不同认知维度）")
print("  - 如果8个标签形成多个独立簇，说明分类具有多维度覆盖性，设计合理")
"""

# 把多行字符串转成 notebook source 格式
lines = PCA_CELL_SOURCE.split('\n')
source_lines = []
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        source_lines.append(line + '\n')
    else:
        if line:
            source_lines.append(line)

PCA_CODE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_lines
}

# 追加到 notebook 末尾
nb['cells'].append(MARKDOWN_CELL)
nb['cells'].append(PCA_CODE_CELL)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"已追加 PCA 分析 cell，当前总 cell 数: {len(nb['cells'])}")
print("notebook 修改完成！")
