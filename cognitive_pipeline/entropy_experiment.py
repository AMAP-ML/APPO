"""
熵实验：验证思维模式标签 token 的预测熵显著高于全局平均熵。

实验设计：
- 用 Qwen2.5-7B-Instruct 对 filtered_seed.jsonl 中的样本做 teacher forcing forward pass
- 计算每个 token 位置的预测熵 H = -sum(p * log(p))
- 标记思维模式开标签（<verify> 等）对应的 token 位置
- 输出 entropy_results.json 供 notebook 可视化
"""
import json
import os

# 指定使用 4、5、6 号 GPU，避开被占用的卡
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

# 必须在 import torch/transformers 之前清理分布式训练环境变量。
# 训练任务结束后 WORLD_SIZE 可能残留，导致 from_pretrained 自动触发 tensor parallelism。
os.environ.pop("WORLD_SIZE", None)
os.environ.pop("RANK", None)
os.environ.pop("LOCAL_RANK", None)
os.environ.pop("MASTER_ADDR", None)
os.environ.pop("MASTER_PORT", None)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
# ============================================================
# 配置
# ============================================================

MODEL_PATH = "/mnt/workspace/wxc/Agent/models/Qwen2.5-7B-Instruct"
DATA_PATH = "/mnt/workspace/wxc/AERPO/cognitive_pipeline/phase2_qwen/annotated_full.jsonl"
OUTPUT_PATH = "/mnt/workspace/wxc/AERPO/cognitive_pipeline/entropy_results.json"

COGNITIVE_TAGS = ["quick", "plan", "deduce", "induce", "hypothesis", "interaction", "meta"]

# 处理样本数量（None 表示跑全量数据，建议设 5000 以内保证运行时间可控）
MAX_SAMPLES = 2000

# ============================================================
# 工具函数
# ============================================================

def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 位置的预测熵。
    logits: [seq_len, vocab_size]
    返回: [seq_len] 的熵值（nats）
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def find_tag_token_positions(token_ids: list, offset_mapping: list, full_text: str, tag_name: str) -> list:
    """
    找到开标签 <tag_name> 在 token 序列中占用的所有 token 位置。

    使用调用方传入的 offset_mapping（来自原始 tokenize 结果），
    通过字符区间与标签字符区间的重叠精确定位，避免 decode→encode 往返带来的对齐误差。

    参数：
        token_ids: 原始 token id 列表（被预测的 token，即 input_ids[1:]）
        offset_mapping: 与 token_ids 对应的字符区间列表，每项为 (char_start, char_end)
        full_text: tokenize 时使用的原始文本（用于字符串搜索）
        tag_name: 认知标签名，如 "verify"
    """
    open_tag = f"<{tag_name}>"
    tag_token_positions = []
    search_start = 0

    while True:
        char_start = full_text.find(open_tag, search_start)
        if char_start == -1:
            break
        char_end = char_start + len(open_tag)

        # 找到所有与 [char_start, char_end) 有字符重叠的 token
        for token_pos, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
            if tok_char_end <= char_start:
                continue
            if tok_char_start >= char_end:
                break
            if token_pos < len(token_ids):
                tag_token_positions.append(token_pos)

        search_start = char_end

    return tag_token_positions


def build_chat_input(sample: dict, tokenizer) -> str:
    """把样本的 conversations 格式化为模型输入文本"""
    convs = sample.get("conversations", [])
    system_prompt = sample.get("system", "")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for conv in convs:
        role = "user" if conv.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": conv.get("value", "")})
    
    # 用 apply_chat_template 格式化（不加 generation prompt，因为是 teacher forcing）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def has_cognitive_tags(sample: dict) -> bool:
    """检查样本是否包含认知标签"""
    gpt_value = sample.get("conversations", [{}])[1].get("value", "")
    for tag in COGNITIVE_TAGS:
        if f"<{tag}>" in gpt_value:
            return True
    return False


# ============================================================
# 主实验流程
# ============================================================

def run_entropy_experiment():
    print("=== 熵实验：思维模式标签 token 的预测熵分析 ===")
    print(f"模型: {MODEL_PATH}")
    print(f"数据: {DATA_PATH}")
    print()

    # 加载模型和 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    print("加载模型（float16，自动分配 GPU）...")
    # 使用明确的 device_map 而非 "auto"，避免新版 transformers 在 WORLD_SIZE 环境变量存在时
    # 自动触发 tensor parallelism 初始化（modeling_utils.py 第 4175 行的逻辑）
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，设备: {next(model.parameters()).device}")
    print()

    # 加载数据，筛选有认知标签的样本
    all_samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if has_cognitive_tags(sample):
                    all_samples.append(sample)
                if MAX_SAMPLES is not None and len(all_samples) >= MAX_SAMPLES:
                    break

    print(f"筛选出 {len(all_samples)} 条含认知标签的样本")

    # 结果容器
    # global_entropies: 所有 token 的熵（用于计算全局均值）
    # tag_entropies: {tag_name: [entropy_values]}（各标签 token 的熵）
    # rollout_candidates: 候选 rollout 列表，最终取标签种类最多的前3条
    global_entropies = []
    tag_entropies = {tag: [] for tag in COGNITIVE_TAGS}
    rollout_candidates = []  # list of dict，每项含 _tag_type_count 用于排序

    for sample_idx, sample in enumerate(all_samples):
        print(f"Processing sample {sample_idx + 1}/{len(all_samples)}...")

        # 构建输入文本
        input_text = build_chat_input(sample, tokenizer)

        # Tokenize，同时获取 offset_mapping（字符区间），用于精确定位标签 token
        encoding = tokenizer(
            input_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = encoding["input_ids"].to(model.device)
        # offset_mapping: [seq_len, 2]，每项为 (char_start, char_end)
        offset_mapping_full = encoding["offset_mapping"][0].tolist()
        token_ids_list = input_ids[0].tolist()
        seq_len = len(token_ids_list)

        if seq_len < 10:
            print(f"  样本太短（{seq_len} tokens），跳过")
            continue

        # Teacher forcing forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # logits[i] 预测第 i+1 个 token，所以：
        # - prediction_logits[i] 对应被预测的 token_ids_list[i+1]
        # - offset_mapping 对应被预测 token 的字符区间为 offset_mapping_full[1:]
        prediction_logits = logits[:-1]  # [seq_len-1, vocab_size]
        token_entropies = compute_token_entropy(prediction_logits).cpu().numpy()  # [seq_len-1]

        predicted_token_ids = token_ids_list[1:]
        predicted_offset_mapping = offset_mapping_full[1:]  # 与 predicted_token_ids 对齐

        # 收集全局熵
        global_entropies.extend(token_entropies.tolist())

        # 一次性计算所有标签的 token 位置，避免重复调用
        all_tag_positions = {
            tag: find_tag_token_positions(
                predicted_token_ids, predicted_offset_mapping, input_text, tag
            )
            for tag in COGNITIVE_TAGS
        }

        # 收集各标签 token 对应的熵
        for tag, tag_positions in all_tag_positions.items():
            for pos in tag_positions:
                if pos < len(token_entropies):
                    tag_entropies[tag].append(float(token_entropies[pos]))

        # 统计本样本含有的标签种类数，用于选 rollout
        sample_tag_types = sum(1 for positions in all_tag_positions.values() if positions)

        # 收集 rollout 候选（取标签种类最多的前3条）
        if sample_tag_types > 0:
            token_strings = [
                tokenizer.decode([tid], skip_special_tokens=False)
                for tid in predicted_token_ids
            ]
            tag_mask = ["none"] * len(predicted_token_ids)
            for tag, positions in all_tag_positions.items():
                for pos in positions:
                    if pos < len(tag_mask):
                        tag_mask[pos] = tag

            rollout_candidates.append({
                "sample_idx": sample_idx,
                "token_strings": token_strings,
                "entropies": token_entropies.tolist(),
                "tag_mask": tag_mask,
                "seq_len": len(predicted_token_ids),
                "_tag_type_count": sample_tag_types,
            })
            # 按标签种类数降序，只保留前3条
            rollout_candidates.sort(key=lambda x: x["_tag_type_count"], reverse=True)
            rollout_candidates = rollout_candidates[:3]
            print(f"  → 加入 rollout 候选（{len(predicted_token_ids)} tokens，{sample_tag_types} 种标签，当前候选数={len(rollout_candidates)}）")

        # 打印当前样本的标签熵统计
        tag_counts = {tag: len(tag_entropies[tag]) for tag in COGNITIVE_TAGS if tag_entropies[tag]}
        if tag_counts:
            print(f"  → 累计标签 token 数: {tag_counts}")

        # 释放显存
        del outputs, logits, prediction_logits, token_entropies
        torch.cuda.empty_cache()

    # 汇总统计
    global_mean_entropy = float(np.mean(global_entropies)) if global_entropies else 0.0
    global_std_entropy = float(np.std(global_entropies)) if global_entropies else 0.0

    tag_stats = {}
    for tag in COGNITIVE_TAGS:
        values = tag_entropies[tag]
        if values:
            tag_stats[tag] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
                "values": values,
            }
        else:
            tag_stats[tag] = {"mean": 0.0, "std": 0.0, "count": 0, "values": []}

    # 清理 rollout 候选中的内部字段，不写入 JSON
    rollout_list = []
    for candidate in rollout_candidates:
        clean = {k: v for k, v in candidate.items() if not k.startswith("_")}
        rollout_list.append(clean)

    # 输出结果
    results = {
        "global_mean_entropy": global_mean_entropy,
        "global_std_entropy": global_std_entropy,
        "total_tokens": len(global_entropies),
        "num_samples": len(all_samples),
        "tag_stats": tag_stats,
        "rollout_list": rollout_list,  # 标签种类最多的前3条 rollout
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("=== 实验结果摘要 ===")
    print(f"全局平均熵: {global_mean_entropy:.4f} ± {global_std_entropy:.4f}")
    print(f"总 token 数: {len(global_entropies)}")
    print()
    print("各思维模式标签 token 的平均熵：")
    for tag in COGNITIVE_TAGS:
        stats = tag_stats[tag]
        if stats["count"] > 0:
            ratio = stats["mean"] / global_mean_entropy if global_mean_entropy > 0 else 0
            print(f"  <{tag}>: {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['count']}, 是全局均值的 {ratio:.2f}x)")
        else:
            print(f"  <{tag}>: 无数据")

    print()
    print(f"结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_entropy_experiment()
