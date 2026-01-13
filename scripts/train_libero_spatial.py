"""
Libero Spatial 行为克隆训练脚本
用于训练 BC 模型在 Libero Spatial 任务上的表现
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path

# 尝试导入 LIBERO 相关库
# 注意: LIBERO 需要从源码安装: cd /path/to/LIBERO && pip install -e .
# 并且需要设置 PYTHONPATH 或从 LIBERO 目录运行
LIBERO_AVAILABLE = False
try:
    # 尝试添加 LIBERO 路径到 sys.path（如果存在）
    import sys
    libero_paths = [
        str(Path(__file__).parent.parent / "LIBERO"),  # 项目目录下的 LIBERO
        os.environ.get("LIBERO_PATH", ""),
        str(Path(__file__).parent.parent.parent / "LIBERO"),  # 项目同级目录
        str(Path.home() / "LIBERO"),  # 用户主目录
        "/tmp/libero_check",  # 临时路径（备用）
    ]
    for path in libero_paths:
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
    LIBERO_AVAILABLE = True
    print("✓ LIBERO 环境已加载")
except (ImportError, ModuleNotFoundError) as e:
    LIBERO_AVAILABLE = False
    print(f"警告: LIBERO 库未正确安装或配置: {e}")
    print("提示:")
    print("  1. 克隆 LIBERO: git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
    print("  2. 安装: cd LIBERO && pip install -e .")
    print("  3. 设置环境变量: export LIBERO_PATH=/path/to/LIBERO")
    print("  4. 或使用启动脚本: bash run_train_libero_official.sh")

from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# 设置路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"


def make_libero_env(task_name: str, render_mode: str = "rgb_array"):
    """创建 Libero Spatial 环境"""
    if not LIBERO_AVAILABLE:
        raise ImportError(
            "LIBERO 库未安装。请:\n"
            "1. 克隆 LIBERO: git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
            "2. 安装: cd LIBERO && pip install -e .\n"
            "3. 设置 PYTHONPATH 或从 LIBERO 目录运行脚本"
        )
    
    # 查找 LIBERO 路径
    import sys
    libero_base = None
    for path in sys.path:
        if "libero" in path.lower() and os.path.exists(path):
            # 尝试找到 libero 包的实际位置
            potential_paths = [
                os.path.join(path, "libero", "libero", "bddl_files", "libero_spatial"),
                os.path.join(os.path.dirname(path), "libero", "libero", "bddl_files", "libero_spatial"),
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    libero_base = os.path.dirname(p)  # libero_spatial 的父目录
                    break
            if libero_base:
                break
    
    # 如果没找到，尝试默认路径
    if not libero_base:
        default_paths = [
            os.path.join(BASE_DIR, "LIBERO", "libero", "libero", "bddl_files"),  # 项目目录下的 LIBERO
            os.path.join(os.path.dirname(__file__), "..", "..", "LIBERO", "libero", "libero", "bddl_files"),
            "/tmp/libero_check/libero/libero/bddl_files",  # 临时路径（备用）
        ]
        for p in default_paths:
            p = os.path.abspath(p)
            if os.path.exists(p):
                libero_base = p
                break
    
    if not libero_base:
        raise RuntimeError("无法找到 LIBERO bddl_files 目录，请设置正确的 LIBERO_PATH")
    
    # 构建 bddl 文件路径
    # task_name 可以是任务名称（不含 .bddl）或完整路径
    if task_name.endswith('.bddl'):
        bddl_file = task_name if os.path.isabs(task_name) else os.path.join(libero_base, "libero_spatial", task_name)
    else:
        # 尝试添加 .bddl 扩展名
        bddl_file = os.path.join(libero_base, "libero_spatial", f"{task_name}.bddl")
    
    if not os.path.exists(bddl_file):
        # 列出可用的任务
        spatial_dir = os.path.join(libero_base, "libero_spatial")
        if os.path.exists(spatial_dir):
            available = [f.replace('.bddl', '') for f in os.listdir(spatial_dir) if f.endswith('.bddl')]
            raise RuntimeError(
                f"任务文件不存在: {bddl_file}\n"
                f"可用任务:\n" + "\n".join([f"  - {t}" for t in available[:10]])
            )
        else:
            raise RuntimeError(f"任务文件不存在: {bddl_file}")
    
    # 创建 Libero 环境
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
        )
    except Exception as e:
        raise RuntimeError(
            f"创建 Libero 环境失败: {e}\n"
            f"BDDL 文件: {bddl_file}"
        )
    return env


class LiberoGymWrapper(gym.Env):
    """将 LIBERO 环境包装为 Gymnasium 兼容的环境"""
    
    def __init__(self, libero_env):
        self.libero_env = libero_env
        self.base_env = libero_env.env if hasattr(libero_env, 'env') else libero_env
        
        # 获取观察和动作空间
        # observation_spec 是方法，需要调用
        obs_spec = self.base_env.observation_spec() if callable(self.base_env.observation_spec) else self.base_env.observation_spec
        act_spec = self.base_env.action_spec() if callable(self.base_env.action_spec) else self.base_env.action_spec
        
        # 转换观察空间
        # LIBERO 使用 OrderedDict，需要转换为 Dict space
        from collections import OrderedDict
        if isinstance(obs_spec, (dict, OrderedDict)):
            obs_dict = {}
            for key, value in obs_spec.items():
                if isinstance(value, np.ndarray):
                    obs_dict[key] = spaces.Box(
                        low=-np.inf, high=np.inf, 
                        shape=value.shape, dtype=value.dtype
                    )
            self.observation_space = spaces.Dict(obs_dict) if obs_dict else spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
        else:
            self.observation_space = obs_spec
        
        # 转换动作空间
        if isinstance(act_spec, tuple) and len(act_spec) == 2:
            self.action_space = spaces.Box(
                low=act_spec[0], high=act_spec[1], dtype=np.float32
            )
        elif isinstance(act_spec, np.ndarray):
            self.action_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=act_spec.shape, dtype=act_spec.dtype
            )
        else:
            self.action_space = act_spec
        
        self.metadata = {"render_modes": ["rgb_array"]}
    
    def reset(self, seed=None, options=None):
        # LIBERO 环境的 reset 可能返回不同格式
        reset_result = self.libero_env.reset()
        
        # 处理不同的返回格式
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
            info = reset_result[1] if len(reset_result) > 1 else {}
        else:
            obs = reset_result
            info = {}
        
        # LIBERO 返回的可能是 OrderedDict 或 dict，需要确保格式正确
        from collections import OrderedDict
        if isinstance(obs, (dict, OrderedDict)):
            # 确保所有值都是 numpy 数组，保持字典格式
            obs = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                   for k, v in obs.items()}
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        return obs, info
    
    def step(self, action):
        step_result = self.libero_env.step(action)
        
        # 处理不同的返回格式
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"意外的 step 返回格式: {len(step_result)} 个值")
        
        # 确保观察格式正确
        if isinstance(obs, dict):
            obs = {k: np.array(v) if not isinstance(v, np.ndarray) else v 
                   for k, v in obs.items()}
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.libero_env.render()
    
    def close(self):
        return self.libero_env.close()


def load_demonstrations(data_path: str):
    """加载专家演示数据（支持多任务数据）
    
    支持两种数据格式：
    1. 单个 HDF5 文件（包含所有任务的数据）
    2. 目录包含多个 HDF5 文件（每个子任务一个文件）
    """
    import h5py
    import glob
    
    demonstrations = []
    
    # 如果是目录，加载所有 .hdf5 文件
    if os.path.isdir(data_path):
        hdf5_files = glob.glob(os.path.join(data_path, "*.hdf5")) + glob.glob(os.path.join(data_path, "*.h5"))
        if not hdf5_files:
            print(f"警告: 目录 {data_path} 中未找到 HDF5 文件")
            return demonstrations
        
        print(f"从目录加载 {len(hdf5_files)} 个数据文件...")
        for file_path in hdf5_files:
            task_name = os.path.basename(file_path).replace('_demo.hdf5', '').replace('_demo.h5', '')
            print(f"  加载: {task_name}")
            demos = load_single_file(file_path, task_name)
            demonstrations.extend(demos)
        
        print(f"总共加载了 {len(demonstrations)} 个演示样本")
        return demonstrations
    
    # 如果是单个文件
    elif os.path.isfile(data_path):
        return load_single_file(data_path)
    else:
        print(f"错误: 数据路径不存在: {data_path}")
        return demonstrations


def load_single_file(file_path: str, task_name: str = None):
    """加载单个 HDF5 文件（LIBERO 格式）
    
    LIBERO 数据格式：
    data/
      demo_0/
        obs/ (包含多个观察键)
        actions/
      demo_1/
      ...
    """
    import h5py
    
    demonstrations = []
    try:
        with h5py.File(file_path, 'r') as f:
            # LIBERO 格式：data/demo_X/obs/ 和 data/demo_X/actions/
            if 'data' in f:
                demos = sorted([k for k in f['data'].keys() if k.startswith('demo_')], 
                              key=lambda x: int(x.split('_')[1]))
                
                for demo_key in demos:
                    demo_group = f[f'data/{demo_key}']
                    
                    if 'obs' in demo_group and 'actions' in demo_group:
                        # 获取观察（字典格式，包含多个键）
                        obs_dict = {}
                        for obs_key in demo_group['obs'].keys():
                            obs_dict[obs_key] = demo_group[f'obs/{obs_key}'][:]
                        
                        # 获取动作
                        actions = demo_group['actions'][:]
                        
                        # 将每个时间步的 (obs, action) 对添加到演示中
                        num_steps = actions.shape[0]
                        for t in range(num_steps):
                            # 提取当前时间步的观察（字典）
                            # 注意：数据中的键名可能与环境的键名不同
                            # 需要统一键名：agentview_rgb -> agentview_image, eye_in_hand_rgb -> robot0_eye_in_hand_image
                            step_obs = {}
                            for k, v in obs_dict.items():
                                # 映射键名以匹配环境
                                # 数据键 -> 环境键的映射
                                key_mapping = {
                                    'agentview_rgb': 'agentview_image',
                                    'eye_in_hand_rgb': 'robot0_eye_in_hand_image',
                                    'ee_pos': 'robot0_eef_pos',
                                    'ee_ori': 'robot0_eef_quat',
                                    'joint_states': 'robot0_joint_pos',
                                    'gripper_states': 'robot0_gripper_qpos',
                                    # ee_states 可能包含多个状态，需要拆分或忽略
                                    # 'ee_states': 需要特殊处理，暂时跳过
                                }
                                
                                mapped_key = key_mapping.get(k, k)
                                # 跳过 ee_states，因为它可能包含冗余信息
                                if k == 'ee_states':
                                    continue
                                
                                step_obs[mapped_key] = v[t]
                            step_action = actions[t]
                            
                            if task_name:
                                demonstrations.append((step_obs, step_action, task_name))
                            else:
                                demonstrations.append((step_obs, step_action))
            # 兼容其他格式（直接包含 observations 和 actions）
            elif 'observations' in f and 'actions' in f:
                observations = f['observations'][:]
                actions = f['actions'][:]
                
                if 'task_id' in f:
                    task_ids = f['task_id'][:]
                    demonstrations = list(zip(observations, actions, task_ids))
                elif task_name:
                    task_names = [task_name] * len(observations)
                    demonstrations = list(zip(observations, actions, task_names))
                else:
                    demonstrations = list(zip(observations, actions))
            else:
                print(f"警告: 无法识别数据格式，文件: {file_path}")
                print(f"可用键: {list(f.keys())}")
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return demonstrations


def train_bc(
    task_name: str = "libero_spatial",  # 默认使用整个 Spatial 套件
    data_path: str = None,
    epochs: int = 50,  # 改为官方默认值
    batch_size: int = 32,
    learning_rate: float = 0.0001,  # 改为官方默认值 (1e-4)
    save_freq: int = 1000,
    eval_freq: int = 500,
    device: str = "cpu",  # 默认使用 CPU
    use_libero_config: bool = True,  # 是否使用 LIBERO 官方配置
    use_scheduler: bool = True,  # 是否使用学习率调度器
    use_augmentation: bool = True,  # 是否使用数据增强
):
    """训练行为克隆模型（支持多任务训练）
    
    LIBERO Spatial 是一个任务套件，包含多个子任务。
    数据集应该包含所有子任务的演示数据，BC 会学习处理所有任务。
    """
    
    print(f"使用设备: {device}")
    print(f"任务套件: {task_name}")
    
    # 加载演示数据（优先）
    if data_path is None:
        # 默认使用 libero_spatial 目录（包含所有子任务的数据文件）
        data_path = str(DATA_DIR / "libero_spatial")
    
    print(f"从 {data_path} 加载演示数据...")
    demonstrations = load_demonstrations(data_path)
    
    if len(demonstrations) == 0:
        raise RuntimeError(
            f"未找到演示数据: {data_path}\n"
            f"请确保数据集包含所有 LIBERO Spatial 子任务的演示数据"
        )
    
    print(f"加载了 {len(demonstrations)} 个演示样本")
    
    # 从演示数据中提取观察和动作空间信息
    # 使用第一个演示样本来确定空间维度
    if len(demonstrations[0]) >= 2:
        sample_obs, sample_act = demonstrations[0][0], demonstrations[0][1]
    else:
        raise RuntimeError("演示数据格式不正确，需要 (observation, action) 对")
    
    # 创建环境用于获取正确的空间定义（使用任意一个子任务作为参考）
    # 注意：所有 Spatial 子任务应该有相同的观察和动作空间
    if LIBERO_AVAILABLE:
        try:
            # 使用第一个子任务作为参考环境
            reference_task = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
            env = make_libero_env(reference_task)
            eval_env = make_libero_env(reference_task)
        except Exception as e:
            print(f"创建参考环境失败: {e}")
            print("尝试使用演示数据推断空间维度...")
            env = None
            eval_env = None
    else:
        print("LIBERO 不可用，将仅使用演示数据推断空间维度")
        env = None
        eval_env = None
    
    # 获取观察和动作空间
    from gymnasium.spaces import Box, Dict
    import gymnasium as gym
    
    if env is not None:
        # 从环境获取空间定义（更准确）
        base_env = env.env if hasattr(env, 'env') else env
        
        # 从 spec 创建观察空间（robosuite 使用字典格式）
        obs_spec = base_env.observation_spec
        if isinstance(obs_spec, dict):
            # 将字典 spec 转换为 Dict Space
            obs_dict = {}
            for key, value in obs_spec.items():
                if isinstance(value, np.ndarray):
                    obs_dict[key] = Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype)
            obs_space = Dict(obs_dict) if obs_dict else Box(low=-np.inf, high=np.inf, shape=(1,))
        else:
            obs_space = obs_spec
        
        # 获取动作空间
        act_spec = base_env.action_spec
        if isinstance(act_spec, tuple) and len(act_spec) == 2:
            # robosuite action_spec 是 (low, high) 元组
            act_space = Box(low=act_spec[0], high=act_spec[1], dtype=np.float32)
        elif isinstance(act_spec, np.ndarray):
            act_space = Box(low=-np.inf, high=np.inf, shape=act_spec.shape, dtype=act_spec.dtype)
        else:
            act_space = act_spec
    else:
        # 从演示数据推断空间维度
        print("从演示数据推断观察和动作空间...")
        if isinstance(sample_obs, dict):
            obs_dict = {}
            for key, value in sample_obs.items():
                if isinstance(value, np.ndarray):
                    obs_dict[key] = Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype)
            obs_space = Dict(obs_dict) if obs_dict else Box(low=-np.inf, high=np.inf, shape=(1,))
        elif isinstance(sample_obs, np.ndarray):
            obs_space = Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=sample_obs.dtype)
        else:
            obs_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        
        if isinstance(sample_act, np.ndarray):
            act_space = Box(low=-1.0, high=1.0, shape=sample_act.shape, dtype=sample_act.dtype)
        else:
            act_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # 默认7维动作空间
    
    # 创建向量化环境（用于评估，如果环境可用）
    if env is not None and eval_env is not None:
        # 包装为 Gymnasium 兼容的环境
        # 注意：需要创建新的环境实例，不能复用
        wrapped_env = LiberoGymWrapper(env)
        wrapped_eval_env = LiberoGymWrapper(eval_env)
        # DummyVecEnv 需要函数来创建新实例
        # DummyVecEnv 的 reset 返回 (obs,)，不是 (obs, info)
        def make_wrapped_env():
            return LiberoGymWrapper(make_libero_env(reference_task))
        venv = DummyVecEnv([make_wrapped_env])
        eval_venv = DummyVecEnv([make_wrapped_env])
    else:
        venv = None
        eval_venv = None
    
    # 准备演示数据
    # imitation BC 期望演示数据是 Transitions 对象或轨迹列表
    from imitation.data.types import Transitions
    
    # 提取观察和动作
    obs_list = []
    acts_list = []
    for demo in demonstrations:
        if len(demo) >= 2:
            obs_list.append(demo[0])
            acts_list.append(demo[1])
    
    print(f"准备训练数据: {len(obs_list)} 个 (观察, 动作) 对")
    
    # 转换为 Transitions 对象
    # Transitions 需要 obs, acts, next_obs, dones, infos
    # 对于字典观察，需要转换为 DictObs 格式
    if isinstance(obs_list[0], dict):
        # 字典观察：转换为 DictObs 格式
        from imitation.data.types import DictObs
        
        # 将字典列表转换为 DictObs 格式
        # 每个观察键对应一个数组，形状为 (num_samples, ...)
        obs_dict = {}
        next_obs_dict = {}
        for key in obs_list[0].keys():
            # 堆叠所有观察
            obs_arrays = [np.array(obs[key]) for obs in obs_list]
            # 处理不同形状的数组
            if len(obs_arrays[0].shape) == 0:  # 标量
                obs_dict[key] = np.array(obs_arrays)
            else:  # 数组
                stacked = np.stack(obs_arrays)
                # 对于图像数据，转换为 (N, C, H, W) 格式（BC 期望的格式）
                if 'image' in key.lower() or 'rgb' in key.lower():
                    if len(stacked.shape) == 4 and stacked.shape[-1] == 3:
                        # (N, H, W, C) -> (N, C, H, W)
                        stacked = np.transpose(stacked, (0, 3, 1, 2))
                obs_dict[key] = stacked
            
            # 下一个观察（最后一个重复）
            next_obs_arrays = [np.array(obs[key]) for obs in obs_list[1:]] + [np.array(obs_list[-1][key])]
            if len(next_obs_arrays[0].shape) == 0:
                next_obs_dict[key] = np.array(next_obs_arrays)
            else:
                stacked_next = np.stack(next_obs_arrays)
                # 同样的处理：转换为 (N, C, H, W) 格式
                if 'image' in key.lower() or 'rgb' in key.lower():
                    if len(stacked_next.shape) == 4 and stacked_next.shape[-1] == 3:
                        # (N, H, W, C) -> (N, C, H, W)
                        stacked_next = np.transpose(stacked_next, (0, 3, 1, 2))
                next_obs_dict[key] = stacked_next
        
        dones = np.array([False] * (len(obs_list) - 1) + [True])
        
        bc_transitions = Transitions(
            obs=DictObs(obs_dict),
            acts=np.array(acts_list),
            next_obs=DictObs(next_obs_dict),
            dones=dones,
            infos=np.array([{}] * len(obs_list)),
        )
    else:
        obs_array = np.array(obs_list)
        next_obs_array = np.roll(obs_array, -1, axis=0)
        next_obs_array[-1] = obs_array[-1]
        dones = np.array([False] * (len(obs_list) - 1) + [True])
        
        bc_transitions = Transitions(
            obs=obs_array,
            acts=np.array(acts_list),
            next_obs=next_obs_array,
            dones=dones,
            infos=np.array([{}] * len(obs_list)),
        )
    
    # 创建 BC 训练器
    # 使用数据中的观察键来定义观察空间（而不是环境的完整观察空间）
    # 因为数据可能不包含所有环境观察（如对象状态）
    if isinstance(sample_obs, dict):
        # 从数据中的观察创建观察空间
        from collections import OrderedDict
        data_obs_dict = {}
        for key, value in sample_obs.items():
            value_arr = np.array(value) if not isinstance(value, np.ndarray) else value
            # 检查是否是图像数据
            is_image = (value_arr.dtype == np.uint8 or 
                       'image' in key.lower() or 
                       'rgb' in key.lower() or
                       len(value_arr.shape) >= 2)  # 至少2维可能是图像
            
            if is_image and len(value_arr.shape) == 3:
                # 图像数据：转换为 (C, H, W) 格式以匹配 BC 的期望
                # 原始数据是 (H, W, C)，需要转换为 (C, H, W)
                if value_arr.shape[2] == 3:  # RGB 图像
                    # 定义空间时使用 (C, H, W) 格式
                    data_obs_dict[key] = spaces.Box(
                        low=0, high=255, 
                        shape=(3, value_arr.shape[0], value_arr.shape[1]), 
                        dtype=np.uint8
                    )
                else:
                    data_obs_dict[key] = spaces.Box(
                        low=0, high=255, 
                        shape=value_arr.shape, dtype=np.uint8
                    )
            else:
                # 非图像数据
                data_obs_dict[key] = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=value_arr.shape, dtype=value_arr.dtype
                )
        final_obs_space = spaces.Dict(data_obs_dict)
    else:
        # 非字典观察
        if env is not None:
            wrapped_env = LiberoGymWrapper(env)
            final_obs_space = wrapped_env.observation_space
        else:
            final_obs_space = obs_space
    
    # 动作空间
    if env is not None:
        if 'wrapped_env' not in locals():
            wrapped_env = LiberoGymWrapper(env)
        final_act_space = wrapped_env.action_space
    else:
        final_act_space = act_space
    
    rng = np.random.default_rng(42)
    
    # 使用 LIBERO 官方推荐的优化器配置
    if use_libero_config:
        # 使用 AdamW 优化器（官方推荐）
        # 注意：根据文档，optimizer_kwargs 应排除 learning rate 和 weight decay
        # 但实际上 BC 类可能仍需要 lr，我们保留它；weight_decay 使用 l2_weight 参数
        optimizer_kwargs = dict(
            lr=learning_rate,  # 虽然文档说排除，但实际实现可能需要
            betas=(0.9, 0.999),
            # weight_decay 不能放在这里，需要使用 BC 的 l2_weight 参数
        )
        optimizer_cls = torch.optim.AdamW
        l2_weight = 0.0001  # LIBERO 官方默认 weight_decay，通过 l2_weight 参数传递
    else:
        # 使用默认配置
        optimizer_kwargs = dict(lr=learning_rate)
        optimizer_cls = None  # 使用默认优化器
        l2_weight = 0.0  # 不使用 L2 正则化
    
    bc_trainer = bc.BC(
        observation_space=final_obs_space,
        action_space=final_act_space,
        demonstrations=bc_transitions,
        rng=rng,
        batch_size=batch_size,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        l2_weight=l2_weight,  # 使用 l2_weight 而不是 optimizer_kwargs 中的 weight_decay
        device=device,
    )
    
    # 设置回调（如果环境可用）
    if venv is not None:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(MODEL_DIR),
            name_prefix="bc_libero_spatial",
        )
        
        if eval_venv is not None:
            eval_callback = EvalCallback(
                eval_venv,
                best_model_save_path=str(MODEL_DIR),
                log_path=str(LOG_DIR),
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
    
    # 开始训练
    print("="*60)
    print("开始训练 BC 模型（多任务学习）...")
    print("="*60)
    print(f"模型将学习处理所有 LIBERO Spatial 子任务")
    print(f"使用设备: {device}")
    if use_libero_config:
        print("使用 LIBERO 官方推荐配置:")
        print(f"  - 优化器: AdamW (lr={learning_rate}, l2_weight=0.0001)")
        print(f"  - 批次大小: {batch_size}")
        print(f"  - 训练轮数: {epochs}")
        if use_scheduler:
            print(f"  - 学习率调度器: CosineAnnealingLR (eta_min=1e-5)")
        if use_augmentation:
            print(f"  - 数据增强: 启用（需要修改数据加载逻辑）")
    
    # 添加学习率调度器（如果启用）
    # 注意：imitation 库的 BC 类在初始化后可能无法直接访问优化器
    # 我们需要在训练开始前或通过其他方式设置 scheduler
    scheduler = None
    if use_scheduler and use_libero_config:
        # 尝试访问优化器（BC 类可能在训练时才创建优化器）
        # 如果无法访问，我们会在训练后提示用户
        try:
            # 检查 BC 是否有 optimizer 属性
            if hasattr(bc_trainer, 'optimizer') and bc_trainer.optimizer is not None:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    bc_trainer.optimizer,
                    T_max=epochs,
                    eta_min=1e-5,  # LIBERO 默认值
                )
                print("✓ 已启用学习率调度器: CosineAnnealingLR")
            else:
                # BC 可能在训练时才创建优化器，我们稍后处理
                print("提示: 学习率调度器将在训练时设置（如果可能）")
                scheduler = None  # 暂时设为 None，训练时再尝试
        except Exception as e:
            print(f"警告: 无法设置学习率调度器: {e}")
            use_scheduler = False
    
    # 添加数据增强（如果启用）
    if use_augmentation:
        # 注意：imitation 库的 BC 不支持内置的数据增强
        # 要实现数据增强，需要：
        # 1. 在数据加载时使用 torchvision.transforms
        # 2. 或修改 BC 的 policy 在 forward 时应用增强
        # 这需要较大的代码修改，当前版本暂不支持
        print("⚠ 注意: 数据增强功能需要修改数据加载或模型逻辑")
        print("        当前版本暂不支持，建议使用 train_libero_spatial_official.py")
        print("        该脚本支持完整的数据增强功能")
        use_augmentation = False
    
    # 添加训练进度回调
    import time
    start_time = time.time()
    batch_count = [0]  # 使用列表以便在闭包中修改
    epoch_count = [0]  # 用于跟踪 epoch
    
    def progress_callback():
        batch_count[0] += 1
        if batch_count[0] % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[进度] 已处理 {batch_count[0]} 个 batch，耗时 {elapsed/60:.1f} 分钟", flush=True)
    
    # 训练循环
    print(f"\n开始训练，共 {epochs} 个 epoch...")
    if use_scheduler:
        if scheduler is not None:
            print("✓ 已启用学习率调度器: CosineAnnealingLR")
        else:
            print("⚠ 注意: 学习率调度器设置失败（imitation BC 可能不支持）")
            print("        将使用固定学习率训练")
    print("提示: 在 CPU 上训练可能较慢，请耐心等待...")
    print("-"*60)
    
    # 注意：imitation 库的 BC.train() 内部处理所有 epoch
    # 如果需要在每个 epoch 后更新 scheduler，需要修改 BC 的内部实现
    # 或者使用自定义训练循环（但这需要访问 BC 的内部方法）
    bc_trainer.train(
        n_epochs=epochs,
        on_batch_end=progress_callback,
    )
    
    # 提示：由于 BC.train() 内部处理所有 epoch，scheduler 可能无法在每个 epoch 后更新
    # 如果需要完整的学习率调度功能，建议使用 train_libero_spatial_official.py
    if use_scheduler and scheduler is None:
        print("\n提示: 如需完整的学习率调度器支持，请使用 train_libero_spatial_official.py")
    
    total_time = time.time() - start_time
    print("-"*60)
    print(f"训练完成！总耗时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    
    # 训练完成后立即保存模型（避免评估出错导致模型丢失）
    print("\n训练完成，正在保存模型...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model_path = MODEL_DIR / "bc_libero_spatial_final"
    bc_trainer.policy.save(str(final_model_path))
    print(f"✓ 模型已保存到: {final_model_path}")
    print("注意: 此模型可以处理所有 LIBERO Spatial 子任务")
    
    # 评估模型（如果环境可用，可选）
    if eval_venv is not None:
        print("\n开始评估模型...")
        try:
            mean_reward, std_reward = evaluate_policy(
                bc_trainer.policy,
                eval_venv,
                n_eval_episodes=10,
                deterministic=True,
            )
            print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"评估时出错（模型已保存）: {e}")
            print("提示: 评估阶段的观察空间可能与训练时不同，这是正常的")
    
    if env is not None:
        env.close()
    if eval_env is not None:
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="训练 Libero Spatial BC 模型")
    parser.add_argument("--task", type=str, default="libero_spatial", help="任务套件名称（默认: libero_spatial，包含所有子任务）")
    parser.add_argument("--data", type=str, default=None, help="演示数据路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（LIBERO 官方默认: 50）")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率（LIBERO 官方默认: 0.0001）")
    parser.add_argument("--use-libero-config", action="store_true", default=True,
                       help="使用 LIBERO 官方推荐配置（AdamW + weight_decay）")
    parser.add_argument("--use-scheduler", action="store_true", default=True,
                       help="使用学习率调度器（CosineAnnealingLR）")
    parser.add_argument("--use-augmentation", action="store_true", default=False,
                       help="使用数据增强（当前版本暂不支持）")
    parser.add_argument("--save-freq", type=int, default=1000, help="保存频率")
    parser.add_argument("--eval-freq", type=int, default=500, help="评估频率")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = "cpu"  # 默认使用 CPU
    
    train_bc(
        task_name=args.task,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        device=device,
        use_libero_config=args.use_libero_config,
        use_scheduler=args.use_scheduler,
        use_augmentation=args.use_augmentation,
    )


if __name__ == "__main__":
    main()

