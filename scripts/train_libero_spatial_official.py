"""
使用 LIBERO 官方架构和组件训练 BC 模型
直接使用 BCTransformerPolicy、SequenceVLDataset、ResNet 编码器等
"""

import os
import sys
import argparse
from pathlib import Path

# 添加 LIBERO 路径
BASE_DIR = Path(__file__).parent.parent
LIBERO_PATH = BASE_DIR / "LIBERO"
if str(LIBERO_PATH) not in sys.path:
    sys.path.insert(0, str(LIBERO_PATH))

# 导入 LIBERO 组件
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.models import get_policy_class
from libero.lifelong.algos import get_algo_list, get_algo_class
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    create_experiment_dir,
    get_task_embs,
)
from libero.lifelong.metric import evaluate_success
import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
import numpy as np
import time
from easydict import EasyDict
import yaml
from omegaconf import OmegaConf


def train_with_libero_architecture(
    benchmark_name: str = "LIBERO_SPATIAL",
    policy: str = "bc_transformer_policy",
    seed: int = 10000,
    device: str = "cpu",
    folder: str = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.0001,
    use_augmentation: bool = True,
    eval_every: int = 5,
    n_eval: int = 20,
    max_steps: int = 600,
):
    """
    使用 LIBERO 官方架构训练 BC 模型
    
    Args:
        benchmark_name: 基准名称
        policy: 策略类型 (bc_transformer_policy, bc_rnn_policy, bc_vilt_policy)
        seed: 随机种子
        device: 设备
        folder: 数据文件夹路径
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        use_augmentation: 是否使用数据增强
        eval_every: 评估频率（每 N 个 epoch）
        n_eval: 评估时的回合数
        max_steps: 每个回合的最大步数
    """
    
    # 设置数据路径
    if folder is None:
        folder = str(BASE_DIR / "data")
    
    # 设置默认路径
    bddl_folder = get_libero_path("bddl_files")
    init_states_folder = get_libero_path("init_states")
    
    # 手动构建配置（不使用 Hydra，直接加载 YAML 文件）
    # 加载基础配置
    config_path = LIBERO_PATH / "libero" / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    
    # 加载策略配置
    policy_config_path = LIBERO_PATH / "libero" / "configs" / "policy" / f"{policy}.yaml"
    with open(policy_config_path, 'r') as f:
        policy_cfg = yaml.safe_load(f)
    
    # 加载策略的子配置（从 defaults 中）
    # defaults 格式示例：
    # - data_augmentation@color_aug: batch_wise_img_color_jitter_group_aug.yaml
    # - language_encoder: mlp_encoder.yaml
    if "defaults" in policy_cfg:
        for default_item in policy_cfg["defaults"]:
            if isinstance(default_item, dict):
                for key, value in default_item.items():
                    # 处理 @ 语法，如 "data_augmentation@color_aug: batch_wise_img_color_jitter_group_aug.yaml"
                    if "@" in key:
                        parts = key.split("@")
                        parent_key = parts[0]  # data_augmentation
                        child_key = parts[1]   # color_aug
                    else:
                        parent_key = key
                        child_key = None
                    
                    # 查找配置文件（尝试多个路径）
                    possible_paths = [
                        LIBERO_PATH / "libero" / "configs" / "policy" / value,
                        LIBERO_PATH / "libero" / "configs" / "policy" / parent_key / value,
                        LIBERO_PATH / "libero" / "configs" / "policy" / f"{parent_key}s" / value,
                    ]
                    
                    sub_config_path = None
                    for path in possible_paths:
                        if path.exists():
                            sub_config_path = path
                            break
                    
                    if sub_config_path:
                        with open(sub_config_path, 'r') as f:
                            sub_cfg = yaml.safe_load(f)
                            
                            if child_key:
                                # 有 @ 的情况，创建嵌套结构
                                # 但根据 base_policy.py，它期望 color_aug 直接在 policy_cfg 下
                                # 所以对于 data_augmentation@color_aug，我们直接设置 color_aug
                                if parent_key == "data_augmentation" and child_key == "color_aug":
                                    policy_cfg["color_aug"] = sub_cfg
                                elif parent_key == "data_augmentation" and child_key == "translation_aug":
                                    policy_cfg["translation_aug"] = sub_cfg
                                elif parent_key == "position_encoding" and child_key == "temporal_position_encoding":
                                    # position_encoding@temporal_position_encoding -> temporal_position_encoding
                                    policy_cfg["temporal_position_encoding"] = sub_cfg
                                else:
                                    # 其他情况创建嵌套结构
                                    if parent_key not in policy_cfg:
                                        policy_cfg[parent_key] = {}
                                    if not isinstance(policy_cfg[parent_key], dict):
                                        policy_cfg[parent_key] = {}
                                    policy_cfg[parent_key][child_key] = sub_cfg
                            else:
                                # 没有 @ 的情况，直接设置
                                if parent_key not in policy_cfg:
                                    policy_cfg[parent_key] = sub_cfg
                                else:
                                    if isinstance(policy_cfg[parent_key], dict):
                                        policy_cfg[parent_key].update(sub_cfg)
                                    else:
                                        policy_cfg[parent_key] = sub_cfg
    
    # 确保 policy_cfg 是 EasyDict，并递归转换嵌套字典
    def to_easydict(obj):
        if isinstance(obj, dict):
            return EasyDict({k: to_easydict(v) for k, v in obj.items()})
        return obj
    
    policy_cfg = to_easydict(policy_cfg)
    
    # 加载数据配置
    data_config_path = LIBERO_PATH / "libero" / "configs" / "data" / "default.yaml"
    with open(data_config_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # 加载训练配置
    train_config_path = LIBERO_PATH / "libero" / "configs" / "train" / "default.yaml"
    with open(train_config_path, 'r') as f:
        train_cfg = yaml.safe_load(f)
    
    # 加载训练的子配置（optimizer 和 scheduler）
    if "defaults" in train_cfg:
        for default_item in train_cfg["defaults"]:
            if isinstance(default_item, dict):
                for key, value in default_item.items():
                    if "@" in key:
                        sub_key = key.split("@")[0]
                    else:
                        sub_key = key
                    
                    # 查找配置文件
                    sub_config_path = LIBERO_PATH / "libero" / "configs" / "train" / value
                    if not sub_config_path.exists():
                        sub_config_path = LIBERO_PATH / "libero" / "configs" / "train" / sub_key / value
                    
                    if sub_config_path.exists():
                        with open(sub_config_path, 'r') as f:
                            sub_cfg = yaml.safe_load(f)
                            if sub_key not in train_cfg:
                                train_cfg[sub_key] = sub_cfg
                            else:
                                if isinstance(train_cfg[sub_key], dict):
                                    train_cfg[sub_key].update(sub_cfg)
                                else:
                                    train_cfg[sub_key] = sub_cfg
    
    # 加载评估配置
    eval_config_path = LIBERO_PATH / "libero" / "configs" / "eval" / "default.yaml"
    with open(eval_config_path, 'r') as f:
        eval_cfg = yaml.safe_load(f)
    
    # 加载 lifelong 配置
    lifelong_config_path = LIBERO_PATH / "libero" / "configs" / "lifelong" / "multitask.yaml"
    with open(lifelong_config_path, 'r') as f:
        lifelong_cfg = yaml.safe_load(f)
    
    # 合并配置
    cfg = EasyDict({
        **base_cfg,
        "policy": policy_cfg,
        "data": data_cfg,
        "train": train_cfg,
        "eval": eval_cfg,
        "lifelong": lifelong_cfg,
    })
    
    # 覆盖用户指定的参数
    cfg.seed = seed
    cfg.benchmark_name = benchmark_name
    cfg.device = device
    cfg.folder = folder
    cfg.bddl_folder = bddl_folder
    cfg.init_states_folder = init_states_folder
    
    # 确保 train 配置存在
    if not hasattr(cfg, 'train'):
        cfg.train = EasyDict()
    if not hasattr(cfg.train, 'optimizer'):
        cfg.train.optimizer = EasyDict()
        cfg.train.optimizer.name = "torch.optim.AdamW"
        cfg.train.optimizer.kwargs = EasyDict()
    if not hasattr(cfg.train.optimizer, 'kwargs'):
        cfg.train.optimizer.kwargs = EasyDict()
    
    cfg.train.n_epochs = n_epochs
    cfg.train.batch_size = batch_size
    cfg.train.optimizer.kwargs.lr = lr
    if not hasattr(cfg.train.optimizer.kwargs, 'betas'):
        cfg.train.optimizer.kwargs.betas = [0.9, 0.999]
    if not hasattr(cfg.train.optimizer.kwargs, 'weight_decay'):
        cfg.train.optimizer.kwargs.weight_decay = 0.0001
    cfg.train.use_augmentation = use_augmentation
    # CPU 模式下不使用多进程，但需要确保 persistent_workers 设置正确
    cfg.train.num_workers = 0  # CPU 模式下不使用多进程
    
    # 确保 eval 配置存在
    if not hasattr(cfg, 'eval'):
        cfg.eval = EasyDict()
    cfg.eval.n_eval = n_eval
    cfg.eval.eval_every = eval_every
    cfg.eval.max_steps = max_steps
    cfg.eval.use_mp = False
    cfg.eval.num_procs = 1
    cfg.eval.eval = True
    cfg.eval.save_sim_states = False
    
    cfg.use_wandb = False
    cfg.pretrain = False
    cfg.pretrain_model_path = ""
    cfg.load_previous_model = False
    
    # 处理策略配置中的 defaults（加载子配置）
    if "defaults" in policy_cfg:
        # 需要加载子配置，但为了简化，我们假设配置已经完整
        # 实际使用时，LIBERO 的 Hydra 会自动处理这些
        pass
    
    # 控制随机种子
    control_seed(cfg.seed)
    
    print("="*60)
    print("使用 LIBERO 官方架构训练 BC 模型")
    print("="*60)
    print(f"基准: {cfg.benchmark_name}")
    print(f"策略: {cfg.policy.policy_type}")
    print(f"设备: {cfg.device}")
    print(f"数据路径: {cfg.folder}")
    print(f"训练轮数: {cfg.train.n_epochs}")
    print(f"批次大小: {cfg.train.batch_size}")
    print(f"学习率: {cfg.train.optimizer.kwargs.lr}")
    print(f"数据增强: {cfg.train.use_augmentation}")
    print(f"序列长度: {cfg.data.seq_len}")
    print("="*60)
    
    # 获取基准
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks
    
    print(f"\n基准包含 {n_manip_tasks} 个任务")
    
    # 准备数据集
    manip_datasets = []
    descriptions = []
    shape_meta = None
    
    print("\n加载数据集...")
    for i in range(n_manip_tasks):
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
            task_description = benchmark.get_task(i).language
            descriptions.append(task_description)
            manip_datasets.append(task_i_dataset)
            print(f"  ✓ 任务 {i+1}/{n_manip_tasks}: {benchmark.get_task_names()[i]}")
        except Exception as e:
            print(f"  ✗ 任务 {i+1} 加载失败: {e}")
            raise
    
    # 获取任务嵌入（使用 BERT）
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    
    # 创建序列数据集
    datasets = [
        SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
    ]
    
    print(f"\n数据集准备完成:")
    print(f"  总任务数: {len(datasets)}")
    print(f"  总演示数: {sum([d.n_demos for d in datasets])}")
    print(f"  总序列数: {sum([d.total_num_sequences for d in datasets])}")
    
    # 创建实验目录
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta
    
    # 创建算法（Multitask）
    # 注意：n_tasks 应该是 lifelong learning tasks 的数量，对于 multitask 就是 1
    # 但实际传入的是 n_manip_tasks，因为 task_group_size=1
    n_tasks = n_manip_tasks // cfg.data.task_group_size
    algo = safe_device(get_algo_class("Multitask")(n_tasks, cfg), cfg.device)
    
    print(f"\n策略架构: {cfg.policy.policy_type}")
    from libero.lifelong.utils import compute_flops
    GFLOPs, MParams = compute_flops(algo, datasets[0], cfg)
    print(f"模型复杂度: {GFLOPs:.1f} GFLOPs, {MParams:.1f} MParams")
    
    # 保存配置
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        import json
        from libero.lifelong.utils import NpEncoder
        json.dump(cfg, f, cls=NpEncoder, indent=4)
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练（Multitask 模式：同时训练所有任务）")
    print("="*60)
    
    print("[DEBUG] 设置模型为训练模式...", flush=True)
    algo.train()
    print("[DEBUG] 创建结果摘要字典...", flush=True)
    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "L_fwd": np.zeros((n_manip_tasks,)),
        "S_fwd": np.zeros((n_manip_tasks,)),
    }
    
    print("[DEBUG] 准备调用 learn_all_tasks...", flush=True)
    print(f"[DEBUG] 数据集数量: {len(datasets)}", flush=True)
    print(f"[DEBUG] 总序列数: {sum([d.total_num_sequences for d in datasets])}", flush=True)
    s_fwd, l_fwd = algo.learn_all_tasks(datasets, benchmark, result_summary)
    result_summary["L_fwd"][-1] = l_fwd
    result_summary["S_fwd"][-1] = s_fwd
    
    # 评估所有任务
    if cfg.eval.eval:
        print("\n" + "="*60)
        print("评估所有任务")
        print("="*60)
        
        from libero.lifelong.metric import evaluate_loss
        L = evaluate_loss(cfg, algo, benchmark, datasets)
        S = evaluate_success(
            cfg=cfg,
            algo=algo,
            benchmark=benchmark,
            task_ids=list(range(n_manip_tasks)),
            result_summary=None,
        )
        
        result_summary["L_conf_mat"][-1] = L
        result_summary["S_conf_mat"][-1] = S
        
        print("\n各任务损失:")
        print(("[All task loss ] " + " %4.2f |" * n_manip_tasks) % tuple(L))
        print("\n各任务成功率:")
        print(("[All task succ.] " + " %4.2f |" * n_manip_tasks) % tuple(S))
        print(f"\n平均成功率: {np.mean(S):.2%}")
        
        # 保存结果
        torch.save(result_summary, os.path.join(cfg.experiment_dir, "result.pt"))
        print(f"\n结果已保存到: {cfg.experiment_dir}")
    
    # 保存最终模型
    model_path = os.path.join(cfg.experiment_dir, "multitask_model.pth")
    from libero.lifelong.utils import torch_save_model
    torch_save_model(algo.policy, model_path, cfg=cfg)
    print(f"\n最终模型已保存到: {model_path}")
    
    return cfg.experiment_dir, result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LIBERO 官方架构训练 BC")
    parser.add_argument("--benchmark", type=str, default="LIBERO_SPATIAL",
                       help="基准名称")
    parser.add_argument("--policy", type=str, default="bc_transformer_policy",
                       choices=["bc_transformer_policy", "bc_rnn_policy", "bc_vilt_policy"],
                       help="策略类型")
    parser.add_argument("--seed", type=int, default=10000, help="随机种子")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--folder", type=str, default=None, help="数据文件夹")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    parser.add_argument("--no-augmentation", action="store_true",
                       help="禁用数据增强")
    parser.add_argument("--eval-every", type=int, default=5,
                       help="每 N 个 epoch 评估一次")
    parser.add_argument("--n-eval", type=int, default=20,
                       help="评估时的回合数")
    parser.add_argument("--max-steps", type=int, default=600,
                       help="每个回合的最大步数")
    
    args = parser.parse_args()
    
    train_with_libero_architecture(
        benchmark_name=args.benchmark,
        policy=args.policy,
        seed=args.seed,
        device=args.device,
        folder=args.folder,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_augmentation=not args.no_augmentation,
        eval_every=args.eval_every,
        n_eval=args.n_eval,
        max_steps=args.max_steps,
    )

