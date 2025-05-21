"""
# 黄金交易决策生成模块 (Gold Trading Decision Generation Module)
#
# 本文件实现了基于生成因子的黄金交易决策流程。主要内容包括：
#
# 1. 加载预训练的黄金因子池
# 2. 获取最新黄金市场数据并评估因子
# 3. 基于因子信号生成做多/做空交易决策
# 4. 输出交易指令（开仓/平仓、做多/做空）
#
# 与其他组件的关系：
# - 使用alphagen/trade中的交易策略和头寸管理
# - 使用alphagen_qlib/gold_data的黄金数据处理功能
# - 从保存的因子池文件加载训练好的表达式
"""
from math import isnan

import pandas as pd
import numpy as np
from alphagen.trade.base import Position, TradeStatus
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen_qlib.gold_data import GoldData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


POOL_PATH = './out/gold_factors/gold_factor_pool.json'  # 黄金因子池路径


class GoldTradingStrategy:
    """黄金交易策略类"""
    
    def __init__(self, threshold=0.5, stop_loss=0.02, take_profit=0.05):
        """
        初始化黄金交易策略
        
        Args:
            threshold: 信号阈值，超过此值做多，低于-此值做空
            stop_loss: 止损比例
            take_profit: 止盈比例
        """
        self.threshold = threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def generate_decision(self, signal_value, current_position=None):
        """
        根据信号值生成交易决策
        
        Args:
            signal_value: 因子信号值
            current_position: 当前持仓状态，None表示无持仓
            
        Returns:
            决策: "BUY"(做多), "SELL"(做空), "CLOSE"(平仓), "HOLD"(持有)
        """
        # 无持仓时的开仓决策
        if current_position is None:
            if signal_value > self.threshold:
                return "BUY"
            elif signal_value < -self.threshold:
                return "SELL"
            else:
                return "HOLD"
        
        # 有持仓时的平仓或持有决策
        if current_position == "LONG":
            if signal_value < 0:
                return "CLOSE"
            return "HOLD"
        elif current_position == "SHORT":
            if signal_value > 0:
                return "CLOSE"
            return "HOLD"


if __name__ == '__main__':
    # 初始化qlib数据
    initialize_qlib("~/.qlib/qlib_data/gold_data_2024")
    
    # 加载黄金数据
    data = GoldData(
        start_time="2023-01-01",  # 使用最近一年的数据
        end_time="2023-12-31",
        device="cpu"
    )
    
    latest_date = data._dates[-1].strftime("%Y-%m-%d")
    calculator = QLibGoldDataCalculator(data=data, target=None)
    
    # 加载预训练的因子池
    try:
        exprs, weights = load_alpha_pool_by_path(POOL_PATH)
        print(f"成功加载因子池: {len(exprs)}个因子")
    except Exception as e:
        print(f"加载因子池失败: {str(e)}")
        # 如果加载失败，使用示例因子和权重
        from alphagen.data.expression import *
        exprs = [
            Div(Sub(Close(), Shift(Close(), 1)), Shift(Close(), 1)),  # 日收益率
            Div(Sub(Close(), Min(Low(), 5)), Min(Low(), 5)),         # 相对低点距离
            Div(Sub(Max(High(), 5), Close()), Close())               # 相对高点距离
        ]
        weights = np.array([0.4, 0.3, 0.3])
        print("使用默认因子")

    # 计算综合信号
    ensemble_alpha = calculator.make_ensemble_alpha(exprs, weights)
    signal_df = data.make_dataframe(ensemble_alpha)

    # 获取最新信号
    latest_signal = signal_df.iloc[-1].values[0]
    
    # 初始化交易策略
    strategy = GoldTradingStrategy(threshold=0.2)
    
    # 生成交易决策（假设当前无持仓）
    decision = strategy.generate_decision(latest_signal)
    
    # 打印交易决策
    print(f"日期: {latest_date}")
    print(f"信号值: {latest_signal:.4f}")
    print(f"交易决策: {decision}")
    
    # 如果有实际交易记录，可以添加持仓分析
    print("\n策略分析:")
    if decision == "BUY":
        print("市场看涨信号较强，建议做多")
    elif decision == "SELL":
        print("市场看跌信号较强，建议做空")
    elif decision == "HOLD":
        print("市场信号不明确，建议观望")
    elif decision == "CLOSE":
        print("信号反转，建议平仓")
