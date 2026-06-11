"""
将 verl 保存的 FSDP 分片 checkpoint 合并为 HuggingFace 格式。

用法：
    python3 convert_fsdp_to_hf.py \
        --input_dir /data/Agent/logs/ckpt_savedir/.../global_step_80/actor \
        --output_dir /data/Agent/logs/ckpt_savedir/.../global_step_80/actor_hf \
        --world_size 8
"""

import argparse
import os
from pathlib import Path
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


def load_fsdp_shards(input_dir: str, world_size: int) -> OrderedDict:
    """
    加载所有 FSDP DTensor 分片，拼接为完整的 state_dict。
    verl 用 DTensor 格式保存，每个 rank 持有参数的 1/N 分片（to_local() 是局部 tensor）。
    需要把所有 rank 的分片按顺序 cat 拼接才能还原完整参数。
    拼接维度：根据 DTensor 的 placements 决定（Shard(dim) 表示沿 dim 维度分片）。
    """
    import torch.distributed.tensor as dt  # 确保 DTensor 可以被反序列化

    # 先加载所有 rank 的分片
    all_shards = []
    for rank in range(world_size):
        shard_path = os.path.join(input_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"找不到分片文件: {shard_path}")
        print(f"  加载 rank {rank}/{world_size-1}: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        all_shards.append(shard)

    # 按参数名合并
    merged_state_dict = OrderedDict()
    param_keys = list(all_shards[0].keys())
    print(f"  共 {len(param_keys)} 个参数，开始合并...")

    for key in param_keys:
        rank0_value = all_shards[0][key]

        if not isinstance(rank0_value, dt.DTensor):
            # 非 DTensor（如标量），直接取 rank_0 的值
            merged_state_dict[key] = rank0_value
            continue

        # 获取分片维度：DTensor.placements 描述了每个维度的分片方式
        placements = rank0_value.placements
        shard_dim = None
        for placement in placements:
            if isinstance(placement, dt.placement_types.Shard):
                shard_dim = placement.dim
                break

        # 提取每个 rank 的局部 tensor
        local_tensors = [all_shards[rank][key].to_local().cpu() for rank in range(world_size)]

        if shard_dim is not None:
            # 沿分片维度拼接
            merged = torch.cat(local_tensors, dim=shard_dim)
        else:
            # Replicate：所有 rank 相同，clone() 断开 shared storage 引用
            merged = local_tensors[0].clone()

        merged_state_dict[key] = merged

    print(f"  合并完成，共 {len(merged_state_dict)} 个参数")
    return merged_state_dict


def fix_state_dict_keys(state_dict: OrderedDict) -> OrderedDict:
    """
    修复 verl FSDP 保存的 key 名称，去掉可能的前缀（如 '_fsdp_wrapped_module.'）。
    """
    fixed = OrderedDict()
    prefixes_to_remove = ["_fsdp_wrapped_module.", "module.", "_orig_mod."]

    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        fixed[new_key] = value

    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="FSDP checkpoint 目录（包含 model_world_size_*_rank_*.pt）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 HuggingFace 格式目录")
    parser.add_argument("--world_size", type=int, required=True,
                        help="训练时的 world size（分片数量）")
    parser.add_argument("--model_path", type=str, default=None,
                        help="原始模型目录（含 config.json/tokenizer），不指定则从 input_dir 加载")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="保存的权重精度，默认 bfloat16")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    world_size = args.world_size

    print(f"[convert] 输入目录: {input_dir}")
    print(f"[convert] 输出目录: {output_dir}")
    print(f"[convert] world_size: {world_size}")

    # 1. 加载并合并 FSDP 分片
    print("\n[1/4] 加载 FSDP 分片...")
    merged_state_dict = load_fsdp_shards(input_dir, world_size)
    print(f"  合并后共 {len(merged_state_dict)} 个参数")

    # 2. 修复 key 名称
    print("\n[2/4] 修复参数名称...")
    merged_state_dict = fix_state_dict_keys(merged_state_dict)

    # 3. 加载 config 和 tokenizer
    # 优先从 --model_path 加载（原始模型目录），否则尝试从 input_dir 加载
    config_source = args.model_path if args.model_path else input_dir
    print(f"\n[3/4] 加载 config 和 tokenizer（来源: {config_source}）...")
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config_source, trust_remote_code=True)

    # 4. 用 config 初始化空模型，加载 state_dict，保存为 HF 格式
    print("\n[4/4] 保存为 HuggingFace 格式...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    target_dtype = dtype_map[args.dtype]

    # 转换权重精度（在加载前转换，避免加载后再转换多占显存）
    for key in merged_state_dict:
        tensor = merged_state_dict[key]
        if isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
            merged_state_dict[key] = tensor.to(target_dtype)

    # 直接用 from_pretrained 加载 config + state_dict，不用 init_empty_weights
    # （init_empty_weights 产生 meta tensor，无法直接 load_state_dict）
    print("  初始化模型（CPU）...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(target_dtype)

    # 加载权重（strict=False 容忍少量 key 不匹配）
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)
    if missing_keys:
        print(f"  警告：缺少 {len(missing_keys)} 个 key: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  警告：多余 {len(unexpected_keys)} 个 key: {unexpected_keys[:5]}...")

    # 保存为 HuggingFace 格式
    # FSDP 合并后部分 tensor 共享同一 storage，safetensors 不支持 shared tensors，
    # 用 safe_serialization=False 保存为普通 pytorch bin 格式即可绕过此限制。
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

    print(f"\n[done] 转换完成！HuggingFace 格式已保存到: {output_dir}")
    print(f"  可以用以下路径加载: MODEL_PATH=\"{output_dir}\"")


if __name__ == "__main__":
    main()
