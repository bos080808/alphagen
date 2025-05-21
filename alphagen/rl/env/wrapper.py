"""
# 强化学习环境包装器模块 (RL Environment Wrapper Module)
#
# 本文件实现了强化学习环境的包装器，为核心环境提供标准化接口。主要内容包括：
#
# 1. AlphaEnvWrapper：环境包装器类
#    - 将符号表达式令牌转换为离散动作
#    - 管理离散观察空间
#    - 提供动作掩码功能，限制非法操作
#    - 处理子表达式的集成
#
# 2. AlphaEnv：便捷函数，创建完整的环境
#    - 简化环境创建过程
#    - 组合核心环境和包装器
#
# 与其他组件的关系：
# - 包装alphagen/rl/env/core.py中的核心环境
# - 使用alphagen/config.py中的配置常量
# - 为alphagen/rl/policy.py提供标准化的训练接口
# - 支持stable-baselines3等标准RL库的使用
"""
from typing import Tuple, List, Optional
import gymnasium as gym
import numpy as np

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.core import AlphaEnvCore

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP


class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: AlphaEnvCore,
        subexprs: Optional[List[Expression]] = None
    ):
        super().__init__(env)
        self.subexprs = subexprs or []
        self.size_action = SIZE_ACTION + len(self.subexprs)
        self.action_space = gym.spaces.Discrete(self.size_action)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.size_action + SIZE_NULL - 1,
            shape=(MAX_EXPR_LENGTH, ),
            dtype=np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0              # Operators
        for i in range(offset, offset + SIZE_OP):
            if valid['op'][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        if action < len(self.subexprs):
            return ExpressionToken(self.subexprs[action])
        action -= len(self.subexprs)
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        assert False


def AlphaEnv(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, **kwargs):
    return AlphaEnvWrapper(AlphaEnvCore(pool=pool, **kwargs), subexprs=subexprs)
