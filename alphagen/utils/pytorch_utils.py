"""
# PyTorch工具模块 (PyTorch Utilities Module)
#
# 本文件提供了PyTorch相关的工具函数，用于处理张量数据。主要内容包括：
#
# 1. masked_mean_std：计算带掩码的均值和标准差
#    - 处理NaN值和缺失数据
#    - 支持批量计算
#
# 2. normalize_by_day：按日期标准化数据
#    - 实现金融数据的每日截面标准化
#    - 处理异常值
#
# 与其他组件的关系：
# - 被alphagen/utils/correlation.py使用，支持相关性计算
# - 为alphagen/data中的数据处理提供基础工具
# - 支持alphagen/models中的因子计算和处理
"""
from typing import Tuple, Optional
import torch
from torch import Tensor


def masked_mean_std(
    x: Tensor,
    n: Optional[Tensor] = None,
    mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    `x`: [days, stocks], input data
    `n`: [days], should be `(~mask).sum(dim=1)`, provide this to avoid unnecessary computations
    `mask`: [days, stocks], data masked as `True` will not participate in the computation, \
    defaults to `torch.isnan(x)`
    """
    if mask is None:
        mask = torch.isnan(x)
    if n is None:
        n = (~mask).sum(dim=1)
    x = x.clone()
    x[mask] = 0.
    mean = x.sum(dim=1) / n
    std = ((((x - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()
    return mean, std


def normalize_by_day(value: Tensor) -> Tensor:
    "The shape of the input and the output is (days, stocks)"
    mean, std = masked_mean_std(value)
    value = (value - mean[:, None]) / std[:, None]
    nan_mask = torch.isnan(value)
    value[nan_mask] = 0.
    return value
