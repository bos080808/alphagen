"""
# 策略模型模块 (Policy Model Module)
# 
# 本文件实现了基于PPO（Proximal Policy Optimization）算法的强化学习策略，
# 用于生成量化投资因子。主要内容包括：
# 
# 1. 神经网络特征提取器：用于处理观察数据
#    - TransformerSharedNet：基于Transformer架构的特征提取
#    - LSTMSharedNet：基于LSTM的特征提取
#    - Decoder：用于解码观察数据
# 
# 2. 奖励计算：MultiMetricReward类计算多种指标的综合奖励
#    - 信息系数(IC)、信息比率(IR)、排名IC、夏普比率、索提诺比率等
# 
# 3. PPO策略：PPOPolicy类实现了基于PPO算法的策略
#    - 训练、预测、保存和加载模型的方法
# 
# 与其他组件的关系：
# - 与alphagen/rl/env/factor_env.py协同工作，环境提供状态和奖励，策略决定动作
# - 使用alphagen/data/expression.py中的表达式表示生成的因子
# - 最终生成的因子存储在alphagen/models中的因子池中
"""
import gymnasium as gym
import math
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from stable_baselines3 import PPO

from alphagen.data.expression import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        "x: ([batch_size, ]seq_len, embedding_dim)"
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]  # type: ignore


class TransformerSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_encoder_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head,
                dim_feedforward=d_ffn, dropout=dropout,
                activation=lambda x: F.leaky_relu(x),               # type: ignore
                batch_first=True, device=device
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=1e-5, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        pad_mask = obs == 0
        src = self._pos_enc(self._token_emb(obs))
        res = self._transformer(src, src_key_padding_mask=pad_mask)
        return res.mean(dim=1)


class LSTMSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        real_len = (obs != 0).sum(1).max()
        src = self._pos_enc(self._token_emb(obs))
        res = self._lstm(src[:,:real_len])[0]
        return res.mean(dim=1)


class Decoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        # Actually an encoder for now
        self._decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, batch_first=True, device=device
            ),
            n_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        batch_size = obs.size(0)
        begins = torch.full(size=(batch_size, 1), fill_value=self._n_actions,
                            dtype=torch.long, device=obs.device)
        obs = torch.cat((begins, obs.type(torch.long)), dim=1)      # (bs, len)
        pad_mask = obs == 0
        res = self._token_emb(obs)                                  # (bs, len, d_model)
        res = self._pos_enc(res)                                    # (bs, len, d_model)
        res = self._decoder(res, src_key_padding_mask=pad_mask)     # (bs, len, d_model)
        return res.mean(dim=1)                                      # (bs, d_model)


@dataclass
class RewardMetrics:
    """Metrics used for reward calculation"""
    ic: float  # Information Coefficient
    ir: float  # Information Ratio
    rank_ic: float  # Rank Information Coefficient
    sharpe: float  # Sharpe Ratio
    sortino: float  # Sortino Ratio

class MultiMetricReward:
    """Calculate reward using multiple metrics"""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_periods: int = 20
    ):
        self.weights = weights or {
            'ic': 0.2,
            'ir': 0.2,
            'rank_ic': 0.2,
            'sharpe': 0.2,
            'sortino': 0.2
        }
        self.min_periods = min_periods
        
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_metrics(
        self,
        factor_values: np.ndarray,
        returns: np.ndarray,
        positions: Optional[np.ndarray] = None
    ) -> RewardMetrics:
        """Calculate all metrics for reward computation"""
        ic = np.corrcoef(factor_values, returns)[0, 1]
        rank_ic = self._rank_correlation(factor_values, returns)
        
        rolling_ic = self._rolling_correlation(factor_values, returns, self.min_periods)
        ir = rolling_ic.mean() / (rolling_ic.std() + 1e-8)
        
        if positions is not None:
            portfolio_returns = positions * returns
        else:
            portfolio_returns = np.sign(factor_values) * returns
            
        sharpe = self._calculate_sharpe(portfolio_returns)
        sortino = self._calculate_sortino(portfolio_returns)
        
        return RewardMetrics(
            ic=ic,
            ir=ir,
            rank_ic=rank_ic,
            sharpe=sharpe,
            sortino=sortino
        )
    
    def calculate_reward(
        self,
        factor_values: np.ndarray,
        returns: np.ndarray,
        positions: Optional[np.ndarray] = None
    ) -> float:
        """Calculate composite reward from multiple metrics"""
        metrics = self.calculate_metrics(factor_values, returns, positions)
        
        normalized_metrics = {
            'ic': abs(metrics.ic),
            'ir': self._normalize_unbounded(metrics.ir),
            'rank_ic': abs(metrics.rank_ic),
            'sharpe': self._normalize_unbounded(metrics.sharpe),
            'sortino': self._normalize_unbounded(metrics.sortino)
        }
        
        reward = sum(
            self.weights[metric] * value
            for metric, value in normalized_metrics.items()
        )
        
        return reward
    
    @staticmethod
    def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate rank correlation coefficient"""
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        return np.corrcoef(x_rank, y_rank)[0, 1]
    
    @staticmethod
    def _rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling correlation"""
        correlations = []
        for i in range(len(x) - window + 1):
            corr = np.corrcoef(x[i:i+window], y[i:i+window])[0, 1]
            correlations.append(corr)
        return np.array(correlations)
    
    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
        return (excess_returns.mean() / (excess_returns.std() + 1e-8)) * np.sqrt(252)
    
    @staticmethod
    def _calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        downside_std = downside_returns.std()
        return (excess_returns.mean() / (downside_std + 1e-8)) * np.sqrt(252)
    
    @staticmethod
    def _normalize_unbounded(value: float, clip_threshold: float = 3.0) -> float:
        """Normalize unbounded metrics using sigmoid-like function"""
        return 2 / (1 + np.exp(-value / clip_threshold)) - 1

class PPOPolicy:
    """PPO-based policy for factor generation"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        learning_rate: float = 1e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = "cuda"
    ):
        self.model = PPO(
            "MlpPolicy",
            None,  # Environment will be set later
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            device=device,
            policy_kwargs={
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
                "activation_fn": nn.Tanh
            }
        )
    
    def train(self, env, total_timesteps: int, callback=None):
        """Train the policy"""
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[dict]]:
        """Generate action from observation"""
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the policy"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load the policy"""
        self.model = PPO.load(path)
