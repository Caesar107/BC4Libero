"""
评估训练好的 BC 模型在 LIBERO Spatial 任务上的性能
参考 LIBERO 的评估方式
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict

# 修复 PyTorch 2.6+ 的 weights_only 问题
# BasePolicy 保存的模型包含 gymnasium.spaces，需要允许加载
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    """修补 torch.load 以支持加载包含 gymnasium.spaces 的模型"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# 添加 LIBERO 路径
BASE_DIR = Path(__file__).parent.parent
LIBERO_PATH = BASE_DIR / "LIBERO"
if str(LIBERO_PATH) not in sys.path:
    sys.path.insert(0, str(LIBERO_PATH))

# 尝试导入 LIBERO
LIBERO_AVAILABLE = False
try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import get_benchmark
    LIBERO_AVAILABLE = True
    print("✓ LIBERO 环境已加载")
except ImportError as e:
    print(f"警告: LIBERO 库未正确安装: {e}")
    print("请确保 LIBERO 已正确安装")

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.policies.base import FeedForward32Policy
from gymnasium.spaces import Dict, Box
from scripts.train_libero_spatial import LiberoGymWrapper, make_libero_env


def filter_obs_for_policy(obs, policy_obs_space):
    """
    从环境的完整观察中提取策略需要的观察键
    训练时使用的观察空间只包含数据中的键，不包含对象状态
    """
    if not hasattr(policy_obs_space, 'spaces'):
        # 如果不是字典空间，直接返回
        return obs
    
    filtered_obs = {}
    policy_keys = list(policy_obs_space.spaces.keys())
    
    for key in policy_keys:
        if key in obs:
            # 直接使用
            value = obs[key]
            value_arr = np.array(value) if not isinstance(value, np.ndarray) else value
            
            # 检查是否需要格式转换（图像从 (H, W, C) 转为 (C, H, W)）
            if 'image' in key.lower() and len(value_arr.shape) == 3 and value_arr.shape[2] == 3:
                # 训练时使用的是 (C, H, W) 格式
                value_arr = np.transpose(value_arr, (2, 0, 1))
            
            # 检查形状是否匹配策略的观察空间
            space = policy_obs_space.spaces[key]
            if hasattr(space, 'shape'):
                expected_shape = space.shape
                if value_arr.shape != expected_shape:
                    # 形状不匹配，尝试调整
                    if len(value_arr.shape) == 1 and len(expected_shape) == 1:
                        # 一维数组，尝试截断或填充
                        if value_arr.shape[0] > expected_shape[0]:
                            # 截断（例如 quat 4 -> 3）
                            value_arr = value_arr[:expected_shape[0]]
                        elif value_arr.shape[0] < expected_shape[0]:
                            # 填充
                            padding = np.zeros(expected_shape[0] - value_arr.shape[0], dtype=value_arr.dtype)
                            value_arr = np.concatenate([value_arr, padding])
            
            filtered_obs[key] = value_arr
        else:
            # 尝试键名映射
            mapped = False
            if key == 'agentview_image' and 'agentview_rgb' in obs:
                rgb = obs['agentview_rgb']
                if len(rgb.shape) == 3 and rgb.shape[2] == 3:
                    filtered_obs[key] = np.transpose(rgb, (2, 0, 1))
                else:
                    filtered_obs[key] = rgb
                mapped = True
            elif key == 'robot0_eye_in_hand_image' and 'eye_in_hand_rgb' in obs:
                rgb = obs['eye_in_hand_rgb']
                if len(rgb.shape) == 3 and rgb.shape[2] == 3:
                    filtered_obs[key] = np.transpose(rgb, (2, 0, 1))
                else:
                    filtered_obs[key] = rgb
                mapped = True
            
            if not mapped:
                # 如果找不到，使用零填充
                space = policy_obs_space.spaces[key]
                if hasattr(space, 'shape'):
                    filtered_obs[key] = np.zeros(space.shape, dtype=space.dtype)
                else:
                    filtered_obs[key] = np.zeros(1, dtype=np.float32)
    
    return filtered_obs


def evaluate_policy_on_task(
    policy,
    task_name: str,
    n_episodes: int = 10,
    max_steps: int = 500,
    device: str = "cpu",
    save_video: bool = True,
    video_dir: str = None,
):
    """
    在单个任务上评估策略
    
    Args:
        policy: 训练好的策略
        task_name: 任务名称
        n_episodes: 评估的回合数
        max_steps: 每个回合的最大步数
        device: 设备
        save_video: 是否保存视频
        video_dir: 视频保存目录
    """
    if not LIBERO_AVAILABLE:
        raise RuntimeError("LIBERO 不可用，无法进行评估")
    
    print(f"\n评估任务: {task_name}")
    print(f"评估回合数: {n_episodes}, 最大步数: {max_steps}")
    
    # 创建环境
    try:
        env = make_libero_env(task_name)
        wrapped_env = LiberoGymWrapper(env)
    except Exception as e:
        print(f"创建环境失败: {e}")
        return None
    
    successes = []
    episode_rewards = []
    episode_lengths = []
    
    # 设置视频保存目录
    if save_video:
        if video_dir is None:
            video_dir = BASE_DIR / "videos" / task_name.replace(" ", "_")
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"视频将保存到: {video_dir}")
    
    for episode in range(n_episodes):
        obs, info = wrapped_env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        # 过滤观察以匹配策略的观察空间
        # 确保 filtered_obs 是字典格式（策略期望 Dict 观察空间）
        if isinstance(obs, dict):
            filtered_obs = filter_obs_for_policy(obs, policy.observation_space)
        else:
            # 如果不是字典，可能是数组，需要转换为字典
            if isinstance(policy.observation_space, Dict):
                # 策略期望字典，但环境返回数组，尝试重建
                filtered_obs = filter_obs_for_policy({'obs': obs}, policy.observation_space)
            else:
                filtered_obs = obs
        
        # 调试：检查 filtered_obs 的格式
        if not isinstance(filtered_obs, dict) and isinstance(policy.observation_space, Dict):
            print(f"警告: filtered_obs 不是字典格式，但策略期望 Dict 观察空间")
            print(f"filtered_obs 类型: {type(filtered_obs)}")
            if hasattr(filtered_obs, 'shape'):
                print(f"filtered_obs 形状: {filtered_obs.shape}")
            # 尝试修复
            filtered_obs = filter_obs_for_policy(obs if isinstance(obs, dict) else {'obs': obs}, policy.observation_space)
        
        # 初始化视频帧列表
        video_frames = [] if save_video else None
        
        while not done and steps < max_steps:
            # 预测动作
            # 确保 filtered_obs 是字典格式
            if isinstance(policy.observation_space, Dict) and not isinstance(filtered_obs, dict):
                # 如果策略期望字典但得到的是数组，需要转换
                if isinstance(obs, dict):
                    filtered_obs = filter_obs_for_policy(obs, policy.observation_space)
                else:
                    filtered_obs = filter_obs_for_policy({'obs': obs}, policy.observation_space)
            
            try:
                action, _ = policy.predict(filtered_obs, deterministic=True)
            except ValueError as e:
                print(f"预测动作时出错: {e}")
                print(f"观察类型: {type(filtered_obs)}")
                if isinstance(filtered_obs, dict):
                    print(f"观察键: {list(filtered_obs.keys())}")
                    for k, v in filtered_obs.items():
                        print(f"  {k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
                else:
                    print(f"观察形状: {filtered_obs.shape if hasattr(filtered_obs, 'shape') else 'N/A'}")
                print(f"策略观察空间类型: {type(policy.observation_space)}")
                raise
            
            # 保存当前帧到视频
            if save_video and isinstance(obs, dict) and 'agentview_image' in obs:
                # 获取 agentview 图像（从 (C, H, W) 转回 (H, W, C) 用于显示）
                img = obs['agentview_image']
                if len(img.shape) == 3 and img.shape[0] == 3:
                    # (C, H, W) -> (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                elif len(img.shape) == 3 and img.shape[2] == 3:
                    # 已经是 (H, W, C)
                    img = img
                else:
                    img = None
                
                if img is not None:
                    video_frames.append(img.copy())
            
            # 执行动作
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # 过滤下一个观察
            if isinstance(obs, dict):
                filtered_obs = filter_obs_for_policy(obs, policy.observation_space)
            else:
                filtered_obs = obs
        
        # 检查是否成功（LIBERO 使用稀疏奖励，reward > 0 表示成功）
        success = episode_reward > 0
        successes.append(success)
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # 保存视频
        if save_video and video_frames:
            try:
                import imageio
                video_path = video_dir / f"episode_{episode+1:03d}_success_{success}_steps_{steps}.mp4"
                # 确保图像是 uint8 格式
                frames_uint8 = []
                for frame in video_frames:
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 255)).astype(np.uint8)
                    frames_uint8.append(frame)
                imageio.mimsave(str(video_path), frames_uint8, fps=10)
                print(f"  视频已保存: {video_path.name}")
            except Exception as e:
                print(f"  保存视频失败: {e}")
        
        print(f"  回合 {episode+1}/{n_episodes}: "
              f"成功={success}, 奖励={episode_reward:.2f}, 步数={steps}")
    
    env.close()
    
    # 计算统计信息
    success_rate = np.mean(successes)
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n任务 {task_name} 评估结果:")
    print(f"  成功率: {success_rate:.2%} ({sum(successes)}/{n_episodes})")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均步数: {avg_length:.1f}")
    
    return {
        'task_name': task_name,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'successes': successes,
        'rewards': episode_rewards,
        'lengths': episode_lengths,
    }


def evaluate_all_spatial_tasks(
    model_path: str,
    n_episodes: int = 10,
    max_steps: int = 500,
    device: str = "cpu",
    save_video: bool = True,
    video_dir: str = None,
):
    """
    评估所有 LIBERO Spatial 子任务
    
    Args:
        model_path: 模型路径
        n_episodes: 每个任务的评估回合数
        max_steps: 每个回合的最大步数
        device: 设备
        save_video: 是否保存视频
        video_dir: 视频保存目录
    """
    print(f"加载模型: {model_path}")
    try:
        # 使用 BasePolicy.save() 的保存格式加载
        from stable_baselines3.common.save_util import open_path
        
        # 从 zip 文件加载数据
        with open_path(model_path, mode='r') as file_:
            saved_variables = torch.load(file_, map_location=device, weights_only=False)
        
        # 从保存的数据中获取观察和动作空间
        if 'data' not in saved_variables:
            raise ValueError("保存的模型文件中没有 'data' 键")
        
        data = saved_variables['data']
        saved_obs_space = data.get('observation_space')
        act_space = data.get('action_space')
        
        if act_space is None:
            raise ValueError("保存的模型文件中没有动作空间信息")
        
        # 检查保存的权重期望的输入维度
        state_dict = saved_variables['state_dict']
        saved_input_dim = state_dict['mlp_extractor.policy_net.0.weight'].shape[1]
        
        # 使用保存的观察空间重建策略
        policy = FeedForward32Policy(
            observation_space=saved_obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
        )
        
        current_input_dim = policy.mlp_extractor.policy_net[0].in_features
        
        if saved_input_dim != current_input_dim:
            # 从训练数据重建观察空间
            from scripts.train_libero_spatial import load_demonstrations
            demos = load_demonstrations(str(BASE_DIR / "data" / "libero_spatial"))
            if len(demos) == 0:
                raise ValueError("无法加载训练数据来重建观察空间")
            
            sample_obs = demos[0][0]
            from gymnasium.spaces import Dict, Box
            
            train_obs_dict = {}
            for key, value in sample_obs.items():
                value_arr = np.array(value) if not isinstance(value, np.ndarray) else value
                if 'image' in key.lower() and len(value_arr.shape) == 3 and value_arr.shape[2] == 3:
                    train_obs_dict[key] = Box(
                        low=0, high=255, 
                        shape=(3, value_arr.shape[0], value_arr.shape[1]), 
                        dtype=np.uint8
                    )
                else:
                    train_obs_dict[key] = Box(
                        low=-np.inf, high=np.inf, 
                        shape=value_arr.shape, dtype=value_arr.dtype
                    )
            
            train_obs_space = Dict(train_obs_dict)
            # 使用 CombinedExtractor 处理 Dict 观察空间
            from stable_baselines3.common.torch_layers import CombinedExtractor
            policy = FeedForward32Policy(
                observation_space=train_obs_space,
                action_space=act_space,
                lr_schedule=lambda _: 3e-4,
                features_extractor_class=CombinedExtractor,
            )
            
            new_input_dim = policy.mlp_extractor.policy_net[0].in_features
            if new_input_dim != saved_input_dim:
                # 手动过滤不匹配的权重
                policy_state = policy.state_dict()
                matched_state_dict = {}
                for key, value in state_dict.items():
                    if key in policy_state and policy_state[key].shape == value.shape:
                        matched_state_dict[key] = value
                policy.load_state_dict(matched_state_dict, strict=False)
                print(f"✓ 加载了 {len(matched_state_dict)}/{len(state_dict)} 个匹配的权重参数")
            else:
                policy.load_state_dict(state_dict, strict=True)
        else:
            policy.load_state_dict(state_dict, strict=True)
        
        policy = policy.to(device)
        policy.eval()
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # LIBERO Spatial 的所有子任务
    spatial_tasks = [
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    ]
    
    results = []
    for task_name in spatial_tasks:
        try:
            result = evaluate_policy_on_task(
                policy, task_name, n_episodes, max_steps, device
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"评估任务 {task_name} 时出错: {e}")
            continue
    
    # 汇总结果
    if results:
        print("\n" + "="*60)
        print("总体评估结果")
        print("="*60)
        
        all_success_rates = [r['success_rate'] for r in results]
        all_avg_rewards = [r['avg_reward'] for r in results]
        
        print(f"平均成功率: {np.mean(all_success_rates):.2%}")
        print(f"平均奖励: {np.mean(all_avg_rewards):.2f}")
        print(f"\n各任务成功率:")
        for r in results:
            print(f"  {r['task_name']}: {r['success_rate']:.2%}")
        
        # 保存结果
        results_path = BASE_DIR / "evaluation_results.npz"
        np.savez(
            results_path,
            task_names=[r['task_name'] for r in results],
            success_rates=all_success_rates,
            avg_rewards=all_avg_rewards,
            avg_lengths=[r['avg_length'] for r in results],
        )
        print(f"\n结果已保存到: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="评估 LIBERO Spatial BC 模型")
    parser.add_argument(
        "--model",
        type=str,
        default=str(BASE_DIR / "models" / "bc_libero_spatial_final"),
        help="模型路径",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="单个任务名称（如果指定，只评估该任务）",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="每个任务的评估回合数",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="每个回合的最大步数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备 (cuda/cpu)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="是否保存评估视频",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="视频保存目录（默认: videos/<任务名>）",
    )
    
    args = parser.parse_args()
    
    print(f"加载模型: {args.model}")
    try:
        # BasePolicy.save() 保存的是 zip 文件，使用 BasePolicy.load() 加载
        # 但需要知道具体的策略类，BC 使用的是 FeedForward32Policy
        # 由于 BasePolicy 是抽象类，我们需要手动从 zip 加载
        
        from stable_baselines3.common.save_util import open_path
        
        # 从 zip 文件加载数据
        with open_path(args.model, mode='r') as file_:
            # BasePolicy.save() 保存的格式
            saved_variables = torch.load(file_, map_location=args.device, weights_only=False)
        
        # 从保存的数据中获取观察和动作空间
        if 'data' not in saved_variables:
            raise ValueError("保存的模型文件中没有 'data' 键")
        
        data = saved_variables['data']
        saved_obs_space = data.get('observation_space')
        act_space = data.get('action_space')
        
        if act_space is None:
            raise ValueError("保存的模型文件中没有动作空间信息")
        
        # 保存的模型使用的观察空间输入维度是 527，但保存的观察空间展平后是 98319
        # 这说明保存的观察空间和训练时使用的不同
        # 我们需要使用保存的观察空间，但只加载匹配的权重
        
        # 使用保存的观察空间重建策略
        # 如果观察空间是 Dict，需要使用 CombinedExtractor
        from stable_baselines3.common.torch_layers import CombinedExtractor
        from gymnasium.spaces import Dict as DictSpace
        policy_kwargs = {}
        if isinstance(saved_obs_space, DictSpace):
            policy_kwargs['features_extractor_class'] = CombinedExtractor
        
        policy = FeedForward32Policy(
            observation_space=saved_obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            **policy_kwargs,
        )
        
        # 检查保存的权重期望的输入维度
        state_dict = saved_variables['state_dict']
        saved_input_dim = state_dict['mlp_extractor.policy_net.0.weight'].shape[1]
        current_input_dim = policy.mlp_extractor.policy_net[0].in_features
        
        if saved_input_dim != current_input_dim:
            print(f"警告: 观察空间维度不匹配 (保存: {saved_input_dim}, 当前: {current_input_dim})")
            print("这可能是由于保存的观察空间和训练时使用的不同")
            print("尝试从训练数据重建观察空间...")
            
            # 从训练数据重建观察空间
            from scripts.train_libero_spatial import load_demonstrations
            demos = load_demonstrations(str(BASE_DIR / "data" / "libero_spatial"))
            if len(demos) == 0:
                raise ValueError("无法加载训练数据来重建观察空间")
            
            sample_obs = demos[0][0]
            from gymnasium.spaces import Dict, Box
            
            train_obs_dict = {}
            for key, value in sample_obs.items():
                value_arr = np.array(value) if not isinstance(value, np.ndarray) else value
                if 'image' in key.lower() and len(value_arr.shape) == 3 and value_arr.shape[2] == 3:
                    train_obs_dict[key] = Box(
                        low=0, high=255, 
                        shape=(3, value_arr.shape[0], value_arr.shape[1]), 
                        dtype=np.uint8
                    )
                else:
                    train_obs_dict[key] = Box(
                        low=-np.inf, high=np.inf, 
                        shape=value_arr.shape, dtype=value_arr.dtype
                    )
            
            train_obs_space = Dict(train_obs_dict)
            
            # 使用训练时的观察空间重建策略
            policy = FeedForward32Policy(
                observation_space=train_obs_space,
                action_space=act_space,
                lr_schedule=lambda _: 3e-4,
            )
            
            # 检查维度是否匹配
            new_input_dim = policy.mlp_extractor.policy_net[0].in_features
            if new_input_dim != saved_input_dim:
                print(f"警告: 即使使用训练数据，维度仍不匹配 (保存: {saved_input_dim}, 训练数据: {new_input_dim})")
                print("将手动过滤不匹配的权重")
                # 手动过滤不匹配的权重
                state_dict = saved_variables['state_dict']
                policy_state = policy.state_dict()
                matched_state_dict = {}
                for key, value in state_dict.items():
                    if key in policy_state and policy_state[key].shape == value.shape:
                        matched_state_dict[key] = value
                policy.load_state_dict(matched_state_dict, strict=False)
                print(f"✓ 加载了 {len(matched_state_dict)}/{len(state_dict)} 个匹配的权重参数")
            else:
                policy.load_state_dict(saved_variables['state_dict'], strict=True)
        else:
            # 维度匹配，正常加载
            policy.load_state_dict(saved_variables['state_dict'], strict=True)
        
        policy = policy.to(args.device)
        policy.eval()
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if args.task:
        # 评估单个任务
        evaluate_policy_on_task(
            policy, args.task, args.n_episodes, args.max_steps, args.device,
            save_video=args.save_video, video_dir=args.video_dir
        )
    else:
        # 评估所有任务
        evaluate_all_spatial_tasks(
            args.model, args.n_episodes, args.max_steps, args.device,
            save_video=args.save_video, video_dir=args.video_dir
        )


if __name__ == "__main__":
    main()

