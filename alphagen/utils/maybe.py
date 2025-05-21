"""
# Maybe单子模块 (Maybe Monad Module)
#
# 本文件实现了Maybe单子模式，用于优雅地处理可能为空的值。主要内容包括：
#
# 1. Maybe类：实现Maybe单子
#    - 安全处理可能为None的值
#    - 提供链式调用接口
#    - 支持映射和条件转换
#
# 2. 辅助函数：
#    - some：创建包含值的Maybe实例
#    - none：创建空的Maybe实例
#
# 与其他组件的关系：
# - 被alphagen/data/parser.py使用，处理表达式解析中的可选值
# - 被alphagen/data/expression.py使用，验证表达式参数
# - 提供函数式编程风格的错误处理机制
"""
from typing import Optional, TypeVar, Generic, Type, Callable, cast


_T = TypeVar("_T")
_TRes = TypeVar("_TRes")


class Maybe(Generic[_T]):
    def __init__(self, value: Optional[_T]) -> None:
        self._value = value

    @property
    def is_some(self) -> bool: return self._value is not None

    @property
    def is_none(self) -> bool: return self._value is None

    @property
    def value(self) -> Optional[_T]: return self._value

    def value_or(self, other: _T) -> _T:
        return cast(_T, self.value) if self.is_some else other

    def and_then(self, func: Callable[[_T], "Maybe[_TRes]"]) -> "Maybe[_TRes]":
        return func(cast(_T, self._value)) if self.is_some else Maybe(None)

    def map(self, func: Callable[[_T], _TRes]) -> "Maybe[_TRes]":
        return some(func(cast(_T, self._value))) if self.is_some else Maybe(None)

    def or_else(self, func: Callable[[], "Maybe[_T]"]) -> "Maybe[_T]":
        return self if self.is_some else func()


def some(value: _T) -> Maybe[_T]: return Maybe(value)
def none(_: Type[_T]) -> Maybe[_T]: return Maybe(None)
