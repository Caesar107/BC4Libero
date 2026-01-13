#!/bin/bash
# 下载 LIBERO Spatial 数据集

# 激活环境
source bc-libero-env/bin/activate

# 设置 LIBERO 路径
export PYTHONPATH=./LIBERO:$PYTHONPATH

# 下载数据集到 data 目录
echo "开始下载 LIBERO Spatial 数据集到 data/ 目录..."
python LIBERO/benchmark_scripts/download_libero_datasets.py \
    --download-dir ./data \
    --datasets libero_spatial \
    --use-huggingface

echo "下载完成！数据集保存在: $(pwd)/data"

