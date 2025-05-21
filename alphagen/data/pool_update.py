"""
# 因子池更新模块 (Pool Update Module)
#
# 本文件定义了因子池更新的数据结构和类，用于跟踪和管理因子池的变化。主要内容包括：
#
# 1. PoolUpdate：因子池更新的基类
#    - 定义了跟踪因子池变化的接口
#    - 提供新旧池性能对比方法
#
# 2. 具体更新类型：
#    - SetPool：整体设置新因子池
#    - AddRemoveAlphas：添加或移除特定因子
#
# 与其他组件的关系：
# - 被alphagen/models/linear_alpha_pool.py使用，记录因子池的变更历史
# - 使用alphagen/data/expression.py中的表达式表示因子
# - 为强化学习训练过程提供因子池变化的跟踪和记录
"""
from abc import ABCMeta, abstractmethod
from typing import List, Optional, cast
from dataclasses import dataclass, MISSING

from .expression import Expression


@dataclass
class PoolUpdate(metaclass=ABCMeta):
    @property
    @abstractmethod
    def old_pool(self) -> List[Expression]: ...

    @property
    @abstractmethod
    def new_pool(self) -> List[Expression]: ...

    @property
    @abstractmethod
    def old_pool_ic(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def new_pool_ic(self) -> float: ...

    @property
    def ic_increment(self) -> float:
        return self.new_pool_ic - (self.old_pool_ic or 0.)
    
    @abstractmethod
    def describe(self) -> str: ...
    
    def describe_verbose(self) -> str: return self.describe()

    def _describe_ic_diff(self) -> str:
        return (
            f"{self.old_pool_ic:.4f} -> {self.new_pool_ic:.4f} "
            f"(increment of {self.ic_increment:.4f})"
        )

    def _describe_pool(self, title: str, pool: List[Expression]) -> str:
        list_exprs = "\n".join([f"  {expr}" for expr in pool])
        return f"{title}\n{list_exprs}"


class _PoolUpdateStub:
    old_pool: List[Expression] = cast(List[Expression], MISSING)
    new_pool: List[Expression] = cast(List[Expression], MISSING)
    old_pool_ic: Optional[float] = cast(Optional[float], MISSING)
    new_pool_ic: float = cast(float, MISSING)


@dataclass
class SetPool(_PoolUpdateStub, PoolUpdate):
    old_pool: List[Expression]
    new_pool: List[Expression]
    old_pool_ic: Optional[float]
    new_pool_ic: float
    
    def describe(self) -> str:
        pool = self._describe_pool("Alpha pool:", self.new_pool)
        return f"{pool}\nIC of the combination: {self.new_pool_ic:.4f}"
    
    def describe_verbose(self) -> str:
        if len(self.old_pool) == 0:
            return self.describe()
        old_pool = self._describe_pool("Old alpha pool:", self.old_pool)
        new_pool = self._describe_pool("New alpha pool:", self.new_pool)
        perf = f"IC of the pools: {self._describe_ic_diff()})"
        return f"{old_pool}\n{new_pool}\n{perf}"


@dataclass
class AddRemoveAlphas(_PoolUpdateStub, PoolUpdate):
    added_exprs: List[Expression]
    removed_idx: List[int]
    old_pool: List[Expression]
    old_pool_ic: float
    new_pool_ic: float

    @property
    def new_pool(self) -> List[Expression]:
        remain = [True] * len(self.old_pool)
        for i in self.removed_idx:
            remain[i] = False
        return [expr for i, expr in enumerate(self.old_pool) if remain[i]] + self.added_exprs

    def describe(self) -> str:
        def describe_exprs(title: str, exprs: List[Expression]) -> str:
            if len(exprs) == 0:
                return ""
            if len(exprs) == 1:
                return f"{title}: {exprs[0]}\n"
            exprs_str = "\n".join([f"  {expr}" for expr in exprs])
            return f"{title}s:\n{exprs_str}\n"

        added = describe_exprs("Added alpha", self.added_exprs)
        removed = describe_exprs("Removed alpha", [self.old_pool[i] for i in self.removed_idx])
        perf = f"IC of the combination: {self._describe_ic_diff()}"
        return f"{added}{removed}{perf}"
    
    def describe_verbose(self) -> str:
        old = self._describe_pool("Old alpha pool:", self.old_pool)
        return f"{old}\n{self.describe()}"
