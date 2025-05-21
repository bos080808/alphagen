"""
# 交易策略模块 (Trading Strategy Module)
#
# 本文件定义了交易策略的基类接口。主要内容包括：
#
# 1. Strategy：交易策略基类
#    - 抽象方法step_decision定义了交易决策接口
#    - 策略根据市场状态和当前持仓做出买入/卖出决策
#
# 与其他组件的关系：
# - 使用alphagen/trade/base.py中的基础数据类型
# - 接收来自alphagen/models中因子池生成的信号
# - 可用于回测和实盘交易系统
"""
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import pandas as pd

from alphagen.trade.base import AssetCode


class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def step_decision(self,
                      status_df: pd.DataFrame,
                      position_df: Optional[pd.DataFrame] = None
                     ) -> Tuple[List[AssetCode], List[AssetCode]]:
        pass
