#!/bin/bash
# 评估训练好的 BC 模型

cd "$(dirname "$0")"

# 激活环境
source bc-libero-env/bin/activate

# 设置 LIBERO 路径
export PYTHONPATH=./LIBERO:$PYTHONPATH

# 运行评估脚本
python scripts/evaluate_libero_spatial.py \
    --model models/bc_libero_spatial_final \
    --n-episodes 10 \
    --max-steps 500 \
    --device cpu \
    "$@"

