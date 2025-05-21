"""
# 因子池基类模块 (Alpha Pool Base Module)
#
# 本文件定义了因子池的基类接口，用于管理和评估生成的量化投资因子。
# AlphaPoolBase类提供了因子池的核心功能：
#
# 1. 管理因子容量和状态
# 2. 尝试添加新因子并评估其性能
# 3. 测试因子组合的效果
#
# 与其他组件的关系：
# - 被alphagen/models/linear_alpha_pool.py继承并实现具体功能
# - 使用alphagen/data/calculator.py中的计算器评估因子
# - 存储由alphagen/rl/policy.py通过强化学习生成的因子表达式
"""
from typing import Tuple, Dict, Any, Callable
from abc import ABCMeta, abstractmethod

import torch
from ..data.alpha_calculator_base import AlphaCalculator
from ..data.expression import Expression


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        device: torch.device = torch.device("cpu")
    ):
        self.size = 0
        self.capacity = capacity
        self.calculator = calculator
        self.device = device
        self.eval_cnt = 0
        self.best_ic_ret: float = -1.

    @property
    def vacancy(self) -> int:
        return self.capacity - self.size
    
    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        "Get a dictionary representing the state of this pool."

    @abstractmethod
    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize this pool into a dictionary that can be dumped as json,
        i.e. no complex objects.
        """

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> float: ...

    @abstractmethod
    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]: ...
