from typing import Optional, TypeVar, Callable, Tuple, Dict, List
import os
import pickle
import warnings
import pandas as pd
import numpy as np
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

from alphagen.data.expression import Expression
from alphagen.data.parser import parse_expression
from alphagen_generic.features import *
from alphagen_qlib.gold_data import GoldData, FeatureType
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen.utils.parallel import ParallelExecutor, ChunkProcessor
from alphagen.models.factor_evaluator import FactorEvaluator, LLMFactorEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_T = TypeVar("_T")

def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

@dataclass
class Position:
    """Trading position information"""
    size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class BacktestConfig(DataClassJsonMixin):
    """Configuration for gold trading backtest"""
    start_time: str
    end_time: str
    alpha_path: str
    max_backtrack_days: int = 100
    max_future_days: int = 30
    initial_capital: float = 1000000.0
    position_size: float = 0.1  # Maximum position size as fraction of capital
    leverage: float = 1.0  # Maximum leverage
    transaction_cost: float = 0.0001  # Transaction cost as fraction
    stop_loss: float = 0.02  # Stop loss as fraction of entry price
    take_profit: float = 0.05  # Take profit as fraction of entry price
    trailing_stop: float = 0.01  # Trailing stop as fraction of highest price
    risk_per_trade: float = 0.02  # Maximum risk per trade as fraction of capital
    max_positions: int = 1  # Maximum number of concurrent positions
    chunk_size: int = 252  # Process one year of trading days at a time
    n_workers: Optional[int] = None
    data: Optional[GoldData] = None

class GoldTrader:
    """Gold trading system with risk management"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.capital = config.initial_capital
        self.positions: List[Position] = []
        self.trades_history: List[Dict] = []
        
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management rules"""
        # Calculate maximum position size based on risk per trade
        risk_amount = self.capital * self.config.risk_per_trade
        stop_distance = self.config.stop_loss * price
        max_units = risk_amount / stop_distance
        
        # Apply position size and leverage limits
        max_capital_units = (self.capital * self.config.position_size * 
                           self.config.leverage / price)
        
        return min(max_units, max_capital_units)
    
    def update_positions(
        self,
        current_time: datetime,
        current_price: float,
        high_price: float,
        low_price: float
    ) -> List[Dict]:
        """Update positions and execute risk management rules"""
        closed_trades = []
        
        for position in self.positions[:]:
            # Initialize tracking variables
            highest_price = max(high_price, position.entry_price)
            trailing_stop_price = highest_price * (1 - self.config.trailing_stop)
            
            # Check stop loss
            if low_price <= position.stop_loss:
                trade_result = self._close_position(
                    position, position.stop_loss, current_time, 'stop_loss'
                )
                closed_trades.append(trade_result)
                continue
            
            # Check take profit
            if high_price >= position.take_profit:
                trade_result = self._close_position(
                    position, position.take_profit, current_time, 'take_profit'
                )
                closed_trades.append(trade_result)
                continue
            
            # Check trailing stop
            if low_price <= trailing_stop_price:
                trade_result = self._close_position(
                    position, trailing_stop_price, current_time, 'trailing_stop'
                )
                closed_trades.append(trade_result)
                continue
        
        return closed_trades
    
    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> Dict:
        """Close a position and record the trade"""
        self.positions.remove(position)
        
        # Calculate trade metrics
        entry_value = position.size * position.entry_price
        exit_value = position.size * exit_price
        gross_pnl = exit_value - entry_value
        transaction_cost = (entry_value + exit_value) * self.config.transaction_cost
        net_pnl = gross_pnl - transaction_cost
        
        # Update capital
        self.capital += net_pnl
        
        # Record trade
        trade_record = {
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'transaction_cost': transaction_cost,
            'exit_reason': exit_reason
        }
        
        self.trades_history.append(trade_record)
        return trade_record
    
    def open_position(
        self,
        current_time: datetime,
        current_price: float,
        signal: float
    ) -> Optional[Dict]:
        """Open a new position based on signal and risk management rules"""
        if len(self.positions) >= self.config.max_positions:
            return None
        
        # Calculate position size
        size = self.calculate_position_size(current_price)
        
        # Set stop loss and take profit levels
        stop_loss = current_price * (1 - self.config.stop_loss)
        take_profit = current_price * (1 + self.config.take_profit)
        
        # Create new position
        position = Position(
            size=size,
            entry_price=current_price,
            entry_time=current_time,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        
        # Record entry
        entry_value = size * current_price
        transaction_cost = entry_value * self.config.transaction_cost
        self.capital -= transaction_cost
        
        return {
            'time': current_time,
            'price': current_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'transaction_cost': transaction_cost
        }

class Backtest:
    """Backtesting system for gold trading strategies"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trader = GoldTrader(config)
        self.data = self._load_data()
        self.factor_evaluator = self._setup_evaluator()
    
    def _load_data(self) -> GoldData:
        """Load and prepare gold trading data"""
        return GoldData(
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            data_path=self.config.data.csv_data_path if self.config.data.data_source_type == 'csv' else None,
            max_backtrack_days=self.config.max_backtrack_days,
            max_future_days=self.config.max_future_days,
            n_workers=self.config.n_workers
        )
    
    def _setup_evaluator(self) -> FactorEvaluator:
        """Set up factor evaluation system"""
        # Get price data
        price_data = self.data.make_dataframe(
            self.data.data[:, [self.data.feature_names.index('close')]],
            ['price']
        )
        price_data['returns'] = price_data['price'].pct_change()
        
        return FactorEvaluator(
            price_data=price_data,
            factor_data=pd.DataFrame(),  # Will be updated during backtest
            llm_evaluator=LLMFactorEvaluator(),
            n_workers=self.config.n_workers
        )
    
    def run(self) -> pd.DataFrame:
        """Run backtest with parallel processing"""
        # Load alpha expressions
        alpha_pool = load_alpha_pool_by_path(self.config.alpha_path)
        calculator = QLibGoldDataCalculator(self.data)
        signals = calculator.evaluate_alpha(alpha_pool.make_alpha())
        
        # Convert to DataFrame for analysis
        signal_df = self.data.make_dataframe(signals, ['signal'])
        price_data = self.data.make_dataframe(
            self.data.data[:, [FeatureType.OPEN, FeatureType.HIGH, 
                              FeatureType.LOW, FeatureType.CLOSE]],
            ['open', 'high', 'low', 'close']
        )
        
        # Update factor evaluator
        self.factor_evaluator.factor_data = signal_df
        
        # Process in chunks for memory efficiency
        chunk_processor = ChunkProcessor(
            chunk_size=self.config.chunk_size,
            n_workers=self.config.n_workers
        )
        
        def process_chunk(chunk_data: pd.DataFrame) -> List[Dict]:
            trades = []
            for idx, row in chunk_data.iterrows():
                # Update existing positions
                closed_trades = self.trader.update_positions(
                    current_time=idx,
                    current_price=row['close'],
                    high_price=row['high'],
                    low_price=row['low']
                )
                trades.extend(closed_trades)
                
                # Open new position if signal is strong enough
                if abs(row['signal']) > 0.5:  # Signal threshold
                    entry = self.trader.open_position(
                        current_time=idx,
                        current_price=row['close'],
                        signal=row['signal']
                    )
                    if entry:
                        trades.append(entry)
            
            return trades
        
        # Run backtest in chunks
        all_trades = []
        for chunk_start in range(0, len(price_data), self.config.chunk_size):
            chunk_end = chunk_start + self.config.chunk_size
            chunk = pd.concat([
                price_data.iloc[chunk_start:chunk_end],
                signal_df.iloc[chunk_start:chunk_end]
            ], axis=1)
            
            chunk_trades = process_chunk(chunk)
            all_trades.extend(chunk_trades)
        
        # Convert results to DataFrame
        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty:
            trades_df.set_index('time', inplace=True)
        
        return trades_df
    
    def generate_report(self, trades_df: pd.DataFrame) -> Dict:
        """Generate comprehensive backtest report"""
        if trades_df.empty:
            return {'error': 'No trades executed'}
        
        # Calculate performance metrics
        total_pnl = trades_df['net_pnl'].sum()
        win_rate = (trades_df['net_pnl'] > 0).mean()
        profit_factor = (
            abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()) /
            abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        )
        
        # Calculate returns
        trades_df['returns'] = trades_df['net_pnl'] / self.config.initial_capital
        annual_return = trades_df['returns'].mean() * 252
        annual_volatility = trades_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + trades_df['returns']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        return {
            'total_trades': len(trades_df),
            'total_pnl': total_pnl,
            'final_capital': self.trader.capital,
            'return_pct': (self.trader.capital / self.config.initial_capital - 1) * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility
        }

if __name__ == "__main__":
    # Example usage
    config = BacktestConfig(
        start_time="2020-01-01",
        end_time="2023-12-31",
        alpha_path="path/to/alpha/expressions.json",
        initial_capital=1000000.0,
        transaction_cost=0.0001,
        position_size=0.1,
        leverage=1.0,
        stop_loss=0.02,
        take_profit=0.05,
        trailing_stop=0.01,
        risk_per_trade=0.02,
        max_positions=1,
        chunk_size=252,
        n_workers=os.cpu_count()
    )
    
    try:
        backtest = Backtest(config)
        results = backtest.run()
        print("\nBacktest Results:")
        print(results.to_string())
        report = backtest.generate_report(results)
        print("\nBacktest Report:")
        print(report)
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
