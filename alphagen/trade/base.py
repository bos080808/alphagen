"""
# 交易基础模块 (Trading Base Module)
#
# 本文件定义了交易系统的基础数据类型和结构。主要内容包括：
#
# 1. 基础数据类型：
#    - Amount, Price: 交易金额和价格类型
#    - AssetCode, TradeSignal: 资产代码和信号类型
#
# 2. 数据结构：
#    - Position: 持仓结构
#    - TradeStatus: 交易状态结构
#    - TradeOrder: 交易订单类
#    - OrderDirection: 交易方向枚举
#
# 与其他组件的关系：
# - 被alphagen/trade/strategy.py使用，提供交易决策所需的数据结构
# - 接收来自alphagen/models中因子池生成的信号作为TradeSignal
# - 为回测和实盘交易系统提供基础数据类型
"""
from enum import IntEnum
from typing import Dict, NamedTuple, Optional, Type

import pandera as pa


Amount = float
Price = float

AssetCode = str
TradeSignal = float

Position = pa.DataFrameSchema({
    'code': pa.Column(AssetCode),
    'amount': pa.Column(Amount),
    'days_holded': pa.Column(int),
})

TradeStatus = pa.DataFrameSchema({
    'code': pa.Column(AssetCode),
    'buyable': pa.Column(bool),
    'sellable': pa.Column(bool),
    'signal': pa.Column(TradeSignal, nullable=True),
})


class OrderDirection(IntEnum):
    BUY = 1
    SELL = 2


class TradeOrder:
    code: AssetCode
    amount: Amount
    direction: Optional[OrderDirection]

    def __init__(self,
                 code: AssetCode,
                 amount: Amount):
        self.code = code
        self.amount = amount
        self.direction = None

    def to_buy(self):
        self.direction = OrderDirection.BUY

    def to_sell(self):
        self.direction = OrderDirection.SELL

    def set_direction(self, direction: OrderDirection):
        self.direction = direction
