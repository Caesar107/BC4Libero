#!/bin/bash
# 使用 imitation 库训练 BC 模型（适合 CPU 训练）

cd "$(dirname "$0")"

# 激活环境
source bc-libero-env/bin/activate

# 运行训练脚本
python scripts/train_libero_spatial.py \
    --task libero_spatial \
    --data /home/ssd/zml/BC/data/libero_spatial \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cpu \
    --use-libero-config \
    "$@"
