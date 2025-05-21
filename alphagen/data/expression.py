"""
# 表达式模块 (Expression Module)
#
# 本文件定义了用于表示量化投资因子的表达式树结构。主要内容包括：
#
# 1. 表达式基类和各种具体表达式类型：
#    - 常量、变量、一元和二元操作符等
#    - 支持数学运算、逻辑操作和时序函数
#
# 2. 表达式的计算和优化功能：
#    - 表达式求值
#    - 表达式简化
#    - 表达式转换和序列化
#
# 与其他组件的关系：
# - 被alphagen/rl/policy.py使用，以表示生成的因子
# - 被alphagen/data/calculator.py使用，计算因子值
# - 被alphagen/models中的因子池使用，存储和管理因子
# - 与alphagen/data/parser.py配合，解析和生成表达式
"""
from abc import ABCMeta, abstractmethod
from typing import List, Type, Union, Tuple

import torch
from torch import Tensor
from alphagen.utils.maybe import Maybe, some, none
from alphagen_qlib.gold_data import GoldData, FeatureType


_ExprOrFloat = Union["Expression", float]
_DTimeOrInt = Union["DeltaTime", int]


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: _ExprOrFloat) -> "Add": return Add(self, other)
    def __radd__(self, other: float) -> "Add": return Add(other, self)
    def __sub__(self, other: _ExprOrFloat) -> "Sub": return Sub(self, other)
    def __rsub__(self, other: float) -> "Sub": return Sub(other, self)
    def __mul__(self, other: _ExprOrFloat) -> "Mul": return Mul(self, other)
    def __rmul__(self, other: float) -> "Mul": return Mul(other, self)
    def __truediv__(self, other: _ExprOrFloat) -> "Div": return Div(self, other)
    def __rtruediv__(self, other: float) -> "Div": return Div(other, self)
    def __pow__(self, other: _ExprOrFloat) -> "Pow": return Pow(self, other)
    def __rpow__(self, other: float) -> "Pow": return Pow(other, self)
    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(0., self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool: ...


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]

    def __str__(self) -> str: return '$' + self._feature.name.lower()

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(size=(days, data.n_stocks),
                          fill_value=self.value, dtype=dtype, device=device)

    def __str__(self) -> str: return str(self.value)

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return f"{self._delta_time}d"

    @property
    def is_featured(self): return False


def _into_expr(value: _ExprOrFloat) -> "Expression":
    return value if isinstance(value, Expression) else Constant(value)


def _into_delta_time(value: Union[int, DeltaTime]) -> DeltaTime:
    return value if isinstance(value, DeltaTime) else DeltaTime(value)


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]: ...

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *args) -> Maybe[str]: ...

    @classmethod
    def _check_arity(cls, *args) -> Maybe[str]:
        arity = cls.n_args()
        if len(args) == arity:
            return none(str)
        else:
            return some(f"{cls.__name__} expects {arity} operand(s), but received {len(args)}")

    @classmethod
    def _check_exprs_featured(cls, args: list) -> Maybe[str]:
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if not isinstance(arg, (Expression, float)):
                return some(f"{arg} is not a valid expression")
            if isinstance(arg, DeltaTime):
                return some(f"{cls.__name__} expects a normal expression for operand {i + 1}, "
                            f"but got {arg} (a DeltaTime)")
            any_is_featured = any_is_featured or (isinstance(arg, Expression) and arg.is_featured)
        if not any_is_featured:
            if len(args) == 1:
                return some(f"{cls.__name__} expects a featured expression for its operand, "
                            f"but {args[0]} is not featured")
            else:
                return some(f"{cls.__name__} expects at least one featured expression for its operands, "
                            f"but none of {args} is featured")
        return none(str)

    @classmethod
    def _check_delta_time(cls, arg) -> Maybe[str]:
        if not isinstance(arg, (DeltaTime, int)):
            return some(f"{cls.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return none(str)

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"


class UnaryOperator(Operator):
    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls): return UnaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0]]))

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self) -> Tuple[Expression, ...]: return (self._operand,)

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return BinaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]]))

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    @property
    def operands(self) -> Tuple[Expression, ...]: return (self._lhs, self._rhs)

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return RollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0]])
        ).or_else(
            lambda: cls._check_delta_time(args[1])
        )

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        dt = self._delta_time._delta_time
        start = period.start - dt
        stop = period.stop  # no -1 here: need the last day for the window
        op = self._operand.evaluate(data, slice(start, stop))
        return self._apply(op)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self) -> Tuple[Expression, ...]: return (self._operand, self._delta_time)

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)
        self._delta_time = _into_delta_time(delta_time)

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls): return PairRollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]])
        ).or_else(
            lambda: cls._check_delta_time(args[2])
        )

    def _unfold_one(self, expr: Expression,
                    data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        dt = self._delta_time._delta_time
        start = period.start - dt
        stop = period.stop  # no -1 here: need the last day for the window
        return expr.evaluate(data, slice(start, stop))

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        lhs_tensor = self._unfold_one(self._lhs, data, period)
        rhs_tensor = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs_tensor, rhs_tensor)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    @property
    def operands(self) -> Tuple[Expression, ...]: return (self._lhs, self._rhs, self._delta_time)

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# Concrete classes

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class CSRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        res = operand.clone()
        for i in range(res.size(0)):
            res[i, torch.isnan(res[i])] = float('-inf')
            res[i] = torch.argsort(torch.argsort(res[i])).float() / (res.size(1) - 1)
            res[i, torch.isnan(operand[i])] = float('nan')
        return res


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return (lhs > rhs).float()


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return (lhs < rhs).float()


class Ref(RollingOperator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        super().__init__(operand, delta_time)

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        dt = self._delta_time._delta_time
        start = period.start - dt
        stop = period.stop - dt
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        assert False, "Should not call _apply on Ref"


class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=0, keepdim=True)


class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=0, keepdim=True)


class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=0, keepdim=True)


class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=0, keepdim=True)


class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        m1 = operand.mean(dim=0, keepdim=True)
        m2 = ((operand - m1) ** 2).mean(dim=0, keepdim=True)
        m3 = ((operand - m1) ** 3).mean(dim=0, keepdim=True)
        return m3 / (m2 ** 1.5)


class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        m1 = operand.mean(dim=0, keepdim=True)
        m2 = ((operand - m1) ** 2).mean(dim=0, keepdim=True)
        m4 = ((operand - m1) ** 4).mean(dim=0, keepdim=True)
        return m4 / (m2 ** 2) - 3


class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=0, keepdim=True)[0]


class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=0, keepdim=True)[0]


class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=0, keepdim=True)[0]


class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        med = operand.median(dim=0, keepdim=True)[0]
        return (operand - med).abs().median(dim=0, keepdim=True)[0]


class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        def rank1d(x):
            ranks = x.argsort().argsort().float()
            ranks[x.isnan()] = float('nan')
            return ranks
        results = []
        for i in range(operand.shape[0]):
            results.append(rank1d(operand[i, :]).unsqueeze(0))
        return torch.cat(results, dim=0)


class Delta(RollingOperator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        super().__init__(operand, delta_time)

    def evaluate(self, data: GoldData, period: slice = slice(0, 1)) -> Tensor:
        dt = self._delta_time._delta_time
        curr = self._operand.evaluate(data, period)
        prev = self._operand.evaluate(data, slice(period.start - dt, period.stop - dt))
        return curr - prev

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        assert False, "Should not call _apply on Delta"


class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        length = operand.size(0)
        weights = torch.arange(1, length + 1, device=operand.device, dtype=operand.dtype)
        return (operand * weights.view(-1, 1)).sum(dim=0, keepdim=True) / weights.sum()


class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        length = operand.size(0)
        alpha = 2 / (length + 1)
        weights = torch.empty(length, device=operand.device, dtype=operand.dtype)
        curr_weight = 1.0
        for i in range(length - 1, -1, -1):
            weights[i] = curr_weight
            curr_weight *= (1 - alpha)
        weights /= weights.sum()
        return (operand * weights.view(-1, 1)).sum(dim=0, keepdim=True)


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_mean = lhs.mean(dim=0, keepdim=True)
        rhs_mean = rhs.mean(dim=0, keepdim=True)
        return ((lhs - lhs_mean) * (rhs - rhs_mean)).mean(dim=0, keepdim=True)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        lhs_mean = lhs.mean(dim=0, keepdim=True)
        rhs_mean = rhs.mean(dim=0, keepdim=True)
        lhs_std = lhs.std(dim=0, keepdim=True)
        rhs_std = rhs.std(dim=0, keepdim=True)
        return ((lhs - lhs_mean) * (rhs - rhs_mean)).mean(dim=0, keepdim=True) / (lhs_std * rhs_std)


Operators: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log, CSRank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min,
    Med, Mad, Rank, Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]
