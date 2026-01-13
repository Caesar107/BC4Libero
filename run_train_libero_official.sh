#!/bin/bash
# 使用 LIBERO 官方架构训练 BC 模型（完整功能）

cd "$(dirname "$0")"

# 激活环境
source bc-libero-env/bin/activate

# 设置 LIBERO 路径
export PYTHONPATH=./LIBERO:$PYTHONPATH

# 运行使用 LIBERO 官方架构的训练脚本
python scripts/train_libero_spatial_official.py \
    --benchmark LIBERO_SPATIAL \
    --policy bc_transformer_policy \
    --seed 10000 \
    --device cpu \
    --folder ./data \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001 \
    --eval-every 5 \
    --n-eval 20 \
    --max-steps 600 \
    "$@"

