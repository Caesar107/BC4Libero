# 开发流程总结

## 1. 项目初始化

1. 创建项目目录结构：`data/`, `models/`, `scripts/`, `configs/`, `logs/`
2. 编写 `scripts/train_libero_spatial.py` - 主训练脚本，用于 Libero Spatial BC 训练
3. 编写 `scripts/evaluate_model.py` - 模型评估脚本
4. 编写 `scripts/prepare_data.py` - 数据准备脚本
5. 编写 `scripts/test_environment.py` - 环境测试脚本
6. 编写 `configs/libero_spatial_config.yaml` - 训练配置文件
7. 编写 `requirements.txt` - 依赖列表

## 2. 环境配置

1. 创建 Python venv 环境 `bc-libero-env`
2. 安装基础依赖：numpy, torch, stable-baselines3, imitation, gymnasium 等
3. 发现 conda 网络问题，改用 venv + pip 安装
4. 使用清华镜像源解决网络连接问题

## 3. LIBERO 环境安装

1. 发现 LIBERO 无法通过 `pip install git+...` 直接安装
2. 从源码克隆并安装：`git clone LIBERO && cd LIBERO && pip install -e .`
3. 发现缺少 robosuite 依赖，安装 robosuite==1.4.0
4. 发现缺少 easydict 依赖，安装 easydict
5. 发现缺少 gym==0.25.2 依赖，安装旧版本 gym（LIBERO 需要）
6. 修复训练脚本，添加自动 PYTHONPATH 设置逻辑

## 4. 脚本优化

1. 编写 `run_train_libero.sh` - 自动设置 PYTHONPATH 的启动脚本
2. 编写 `scripts/check_libero.py` - LIBERO 安装检查脚本
3. 修复 `scripts/test_environment.py` 中 `get_libero_path()` 调用错误
4. 更新训练脚本，支持自动查找 LIBERO 路径

## 5. 最终验证

1. 验证 LIBERO 环境可正常导入
2. 验证 BC 训练算法可用
3. 验证 CUDA 支持正常
4. 确认所有依赖已安装完成

## 6. 修复 LIBERO Spatial 任务配置

1. 发现 LIBERO Spatial 任务不是通过任务名称，而是通过 BDDL 文件路径创建
2. 修复 `make_libero_env()` 函数，改为使用 `bddl_file_name` 参数
3. 添加自动查找 LIBERO bddl_files 目录的逻辑
4. 更新默认任务名称为实际的 BDDL 文件名

## 关键问题解决

- **问题1**: conda 网络连接失败 → 改用 venv + pip
- **问题2**: LIBERO 包无法直接导入 → 从源码安装并设置 PYTHONPATH
- **问题3**: robosuite 版本不匹配 → 安装指定版本 1.4.0
- **问题4**: 缺少多个依赖 → 依次安装 easydict, gym==0.25.2
- **问题5**: PYTHONPATH 需要手动设置 → 创建启动脚本自动设置

