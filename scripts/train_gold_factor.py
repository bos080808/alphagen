"""
# 黄金因子训练脚本 (Gold Factor Training Script)
#
# 本文件是使用PPO算法训练黄金价格预测因子的主执行脚本。主要功能包括：
#
# 1. 配置黄金数据和训练参数
# 2. 初始化强化学习环境和PPO策略
# 3. 训练模型并实现早停机制
# 4. 保存训练好的模型
#
# 与其他组件的关系：
# - 使用alphagen/rl/policy.py中的PPO策略
# - 使用alphagen/rl/env/factor_env.py作为训练环境
# - 使用alphagen_generic/data_collection/gold_data.py收集黄金数据
# - 训练结果保存到指定的输出目录
#
# 与rl.py的区别：
# - 使用GoldDataCollector而非GoldData加载并预处理数据
# - 实现了更完善的早停机制，可以检测训练不稳定性
# - 使用FactorGenerationEnv而非AlphaEnv作为环境
# - 专注于黄金市场因子生成，不包含LLM集成功能
# - 配置更加灵活，支持通过yaml文件定义
"""
import sys
from pathlib import Path
import logging
import argparse
import yaml
import torch
import time
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from alphagen_generic.data_collection.gold_data import GoldDataCollector
from alphagen.rl.env.factor_env import FactorGenerationEnv, EnvConfig
from alphagen.rl.policy import PPOPolicy, MultiMetricReward

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_step(step: str, start_time: float):
    """Log step completion with timing"""
    elapsed = time.time() - start_time
    logger.info(f"✓ {step} completed in {elapsed:.2f}s")

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def early_stopping_callback(locals, globals):
    """Callback for early stopping if training becomes unstable"""
    self = locals.get("self")
    if not hasattr(self, "_reward_history"):
        self._reward_history = []
        self._patience = 0
        self._best_reward = -np.inf
        self._max_patience = 5  # 如果5个周期没有改善就停止
        
    if len(self.ep_info_buffer) > 0:
        mean_reward = sum(ep_info["r"] for ep_info in self.ep_info_buffer) / len(self.ep_info_buffer)
        self._reward_history.append(mean_reward)
        
        # 检查是否有改善
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            self._patience = 0
        else:
            self._patience += 1
            
        # 检查是否应该停止
        if self._patience >= self._max_patience:
            logger.warning(f"Early stopping triggered after {self._patience} epochs without improvement")
            return False
            
        # 检查奖励是否异常
        if len(self._reward_history) > 10:
            recent_std = np.std(self._reward_history[-10:])
            if recent_std > 10 or np.isnan(mean_reward):  # 奖励波动太大或出现NaN
                logger.warning("Training unstable, stopping early")
                return False
            
        # 显示训练进度
        if self.num_timesteps % 100 == 0:
            progress = self.num_timesteps / self.n_steps * 100
            logger.info(f"   Progress: {progress:.1f}% - Mean Reward: {mean_reward:.3f} - Best: {self._best_reward:.3f}")
            
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Train gold factor generation model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-path', type=str, required=True, help='Path to gold data')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of rows to use (for testing)')
    return parser.parse_args()

def main():
    total_start = time.time()
    args = parse_args()
    
    # Load config
    start_time = time.time()
    logger.info("1. Loading configuration...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    log_step("Configuration loading", start_time)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data collector and prepare data
    start_time = time.time()
    logger.info("2. Loading and processing data...")
    data_collector = GoldDataCollector(
        data_path=args.data_path,
        features_config=config.get('features_config'),
        sample_size=args.sample_size
    )
    features, returns = data_collector.prepare_data()
    log_step("Data preparation", start_time)
    
    # Initialize environment
    start_time = time.time()
    logger.info("3. Setting up environment...")
    env_config = EnvConfig(**config['env_config'])
    reward_calculator = MultiMetricReward(
        weights=config.get('reward_weights'),
        min_periods=config.get('min_periods', 20)
    )
    
    env = FactorGenerationEnv(
        features=features,
        returns=returns,
        config=env_config,
        reward_calculator=reward_calculator
    )
    log_step("Environment setup", start_time)
    
    # Initialize policy
    start_time = time.time()
    logger.info("4. Initializing policy...")
    device = get_device()
    logger.info(f"   Using device: {device}")
    
    # 设置随机种子以提高稳定性
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    
    policy = PPOPolicy(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        **config['policy_config'],
        device=device
    )
    log_step("Policy initialization", start_time)
    
    # Train policy
    start_time = time.time()
    logger.info("5. Starting training...")
    total_timesteps = config['training']['total_timesteps']
    
    try:
        policy.train(
            env=env,
            total_timesteps=total_timesteps,
            callback=early_stopping_callback
        )
        log_step("Training", start_time)
        
        # Save trained policy
        start_time = time.time()
        logger.info("6. Saving model...")
        policy_path = output_dir / 'gold_factor_policy.zip'
        policy.save(str(policy_path))
        log_step("Model saving", start_time)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if hasattr(policy, "_reward_history"):
            logger.info(f"Best reward achieved: {policy._best_reward:.3f}")
    
    total_elapsed = time.time() - total_start
    logger.info(f"✓ Total execution time: {total_elapsed:.2f}s")

if __name__ == '__main__':
    main() 