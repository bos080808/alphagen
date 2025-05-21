"""
# 因子生成环境模块 (Factor Generation Environment Module)
#
# 本文件实现了强化学习环境，用于生成量化投资因子。主要内容包括：
#
# 1. EnvConfig：环境配置类，设置窗口大小、最大步数、奖励权重等
#
# 2. FactorGenerationEnv：自定义强化学习环境
#    - 状态：历史特征窗口
#    - 动作：生成的因子值
#    - 奖励：基于多种指标计算的综合得分
#
# 与其他组件的关系：
# - 与alphagen/rl/policy.py协同工作，环境提供状态和奖励，策略决定动作
# - 使用alphagen/data目录中的数据和表达式
# - 生成的因子最终存储在alphagen/models中的因子池中
"""
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from dataclasses import dataclass
import logging
from ..policy import MultiMetricReward

logger = logging.getLogger(__name__)

@dataclass
class EnvConfig:
    """Configuration for factor generation environment"""
    window_size: int = 60  # Historical window size for state
    max_steps: int = 252  # Maximum steps per episode (e.g., one trading year)
    reward_weights: Optional[Dict[str, float]] = None  # Weights for different metrics
    min_periods: int = 20  # Minimum periods for calculating metrics
    action_scale: float = 1.0  # Scale factor for actions
    state_scale: float = 1.0  # Scale factor for state normalization
    use_positions: bool = True  # Whether to use position information for reward calculation

class FactorGenerationEnv(gym.Env):
    """Custom Environment for Factor Generation"""
    
    def __init__(
        self,
        features: np.ndarray,  # Shape: (n_samples, n_features)
        returns: np.ndarray,   # Shape: (n_samples,)
        config: Optional[EnvConfig] = None,
        reward_calculator: Optional[MultiMetricReward] = None
    ):
        super().__init__()
        
        self.config = config or EnvConfig()
        self.features = features
        self.returns = returns
        self.reward_calculator = reward_calculator or MultiMetricReward(
            weights=self.config.reward_weights,
            min_periods=self.config.min_periods
        )
        
        # Validate data
        if len(features) != len(returns):
            raise ValueError("Features and returns must have the same length")
        if len(features) < self.config.window_size:
            raise ValueError("Not enough data points for the specified window size")
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-self.config.action_scale,
            high=self.config.action_scale,
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.window_size * features.shape[1],),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
    
    def _get_state(self) -> np.ndarray:
        """Get current state (historical window of features)"""
        start_idx = self.current_step - self.config.window_size
        end_idx = self.current_step
        state = self.features[start_idx:end_idx].copy()
        if self.config.state_scale != 1.0:
            state = state * self.config.state_scale
        return state.flatten()
    
    def _calculate_positions(self, factor_values: np.ndarray) -> np.ndarray:
        """Calculate positions based on factor values"""
        return np.sign(factor_values)
    
    def _get_reward(
        self,
        action: float,
        future_returns: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward using multi-metric system"""
        start_idx = max(0, self.current_step - self.config.min_periods)
        end_idx = self.current_step + 1
        
        factor_values = self.factor_history[start_idx:end_idx]
        returns = self.returns[start_idx:end_idx]
        
        positions = None
        if self.config.use_positions:
            positions = self._calculate_positions(factor_values)
        
        reward = self.reward_calculator.calculate_reward(
            factor_values=factor_values,
            returns=returns,
            positions=positions
        )
        
        metrics = self.reward_calculator.calculate_metrics(
            factor_values=factor_values,
            returns=returns,
            positions=positions
        )
        
        info = {
            'ic': metrics.ic,
            'ir': metrics.ir,
            'rank_ic': metrics.rank_ic,
            'sharpe': metrics.sharpe,
            'sortino': metrics.sortino
        }
        
        return reward, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step"""
        factor_value = float(action[0])
        self.factor_history[self.current_step] = factor_value
        
        reward, info = self._get_reward(factor_value, self.returns[self.current_step:])
        self.current_step += 1
        
        done = (self.current_step >= len(self.features) - 1) or \
               (self.current_step - self.start_step >= self.config.max_steps)
        
        state = self._get_state() if not done else np.zeros_like(self._get_state())
        
        return state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        min_start = self.config.window_size
        max_start = len(self.features) - self.config.max_steps
        self.start_step = np.random.randint(min_start, max_start)
        self.current_step = self.start_step
        
        self.factor_history = np.zeros(len(self.features))
        
        return self._get_state()
    
    def render(self, mode='human'):
        """Render environment (optional)"""
        pass 