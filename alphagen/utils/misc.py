"""
# 杂项工具模块 (Miscellaneous Utilities Module)
#
# 本文件提供了各种通用辅助函数。主要内容包括：
#
# 1. 列表和索引工具：
#    - reverse_enumerate：反向枚举列表元素
#    - find_last_if：查找满足条件的最后一个元素
#
# 2. 函数参数处理：
#    - get_arguments_as_dict：获取函数参数字典
#    - pprint_arguments：格式化打印函数参数
#
# 与其他组件的关系：
# - 被alphagen/data/parser.py使用，支持表达式解析
# - 为整个项目提供通用工具函数
# - 简化调试和参数处理
"""
from typing import TypeVar, List, Iterable, Tuple, Callable, Optional
from types import FrameType
import inspect


_T = TypeVar("_T")


def reverse_enumerate(lst: List[_T]) -> Iterable[Tuple[int, _T]]:
    for i in range(len(lst) - 1, -1, -1):
        yield i, lst[i]


def find_last_if(lst: List[_T], predicate: Callable[[_T], bool]) -> int:
    for i in range(len(lst) - 1, -1, -1):
        if predicate(lst[i]):
            return i
    return -1


def get_arguments_as_dict(frame: Optional[FrameType] = None) -> dict:
    if frame is None:
        frame = inspect.currentframe().f_back   # type: ignore
    keys, _, _, values = inspect.getargvalues(frame)    # type: ignore
    res = {}
    for k in keys:
        if k != "self":
            res[k] = values[k]
    return res


def pprint_arguments(frame: Optional[FrameType] = None) -> dict:
    if frame is None:
        frame = inspect.currentframe().f_back   # type: ignore
    args = get_arguments_as_dict(frame)
    formatted_args = '\n'.join(f"    {k}: {v}" for k, v in args.items())
    print(f"[Parameters]\n{formatted_args}")
    return args
