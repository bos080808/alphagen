"""
# 表达式令牌模块 (Expression Token Module)
#
# 本文件定义了表达式解析和构建过程中使用的令牌（Token）类型。主要内容包括：
#
# 1. 基础令牌类型：
#    - Token：所有令牌的基类
#    - ConstantToken：常量令牌
#    - DeltaTimeToken：时间间隔令牌
#    - FeatureToken：特征令牌
#    - OperatorToken：操作符令牌
#
# 2. 特殊令牌：
#    - SequenceIndicatorToken：序列指示器令牌
#    - BEG_TOKEN、SEP_TOKEN：开始和分隔标记
#
# 与其他组件的关系：
# - 与alphagen/data/tree.py配合，用于构建表达式树
# - 被alphagen/data/parser.py使用，表示解析过程中的中间结果
# - 为alphagen/rl中的策略提供表达式序列化支持
"""
from enum import IntEnum
from typing import Type
from alphagen_qlib.gold_data import FeatureType
from alphagen.data.expression import Operator, Expression


class SequenceIndicatorType(IntEnum):
    BEG = 0
    SEP = 1


class Token:
    def __repr__(self):
        return str(self)


class ConstantToken(Token):
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __str__(self): return str(self.constant)


class DeltaTimeToken(Token):
    def __init__(self, delta_time: int) -> None:
        self.delta_time = delta_time

    def __str__(self): return str(self.delta_time)


class FeatureToken(Token):
    def __init__(self, feature: FeatureType) -> None:
        self.feature = feature

    def __str__(self): return '$' + self.feature.name.lower()


class OperatorToken(Token):
    def __init__(self, operator: Type[Operator]) -> None:
        self.operator = operator

    def __str__(self): return self.operator.__name__


class SequenceIndicatorToken(Token):
    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self): return self.indicator.name


class ExpressionToken(Token):
    def __init__(self, expr: Expression) -> None:
        self.expression = expr

    def __str__(self): return str(self.expression)


BEG_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEG)
SEP_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.SEP)
