#!/bin/bash
# 使用 LIBERO 自带的 BC 训练框架（使用官方推荐配置）

cd LIBERO

# 激活环境
source ../bc-libero-env/bin/activate

# 设置 PYTHONPATH
export PYTHONPATH=.:$PYTHONPATH

# 运行 LIBERO 的 BC 训练（使用官方推荐参数）
# benchmark_name: LIBERO_SPATIAL
# policy: bc_transformer_policy (推荐), bc_rnn_policy, 或 bc_vilt_policy
# lifelong: multitask (多任务训练) 或 base (顺序训练)
# 
# 官方推荐参数：
# - n_epochs: 50 (默认)
# - batch_size: 32 (默认)
# - lr: 0.0001 (1e-4, 官方默认，使用 AdamW)
# - use_augmentation: true (默认)
# - optimizer: AdamW with weight_decay=0.0001
# - scheduler: CosineAnnealingLR
python libero/lifelong/main.py \
    seed=10000 \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_transformer_policy \
    lifelong=multitask \
    device=cpu \
    folder=../data \
    train.n_epochs=50 \
    train.batch_size=32 \
    train.optimizer.kwargs.lr=0.0001 \
    train.use_augmentation=true
