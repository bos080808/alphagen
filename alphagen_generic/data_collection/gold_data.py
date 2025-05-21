"""
# 黄金数据收集模块 (Gold Data Collection Module)
#
# 本文件实现了黄金交易数据的收集和处理功能。主要内容包括：
#
# 1. GoldDataCollector：黄金数据收集器类
#    - 加载原始价格数据
#    - 计算收益率
#    - 生成技术指标特征
#
# 主要特征包括：
# - 价格特征：开盘价、最高价、最低价、收盘价、交易量
# - 技术指标：移动平均线、RSI、MACD、布林带等
#
# 与其他组件的关系：
# - 被./scripts/train_gold_factor.py使用，提供训练数据
# - 为强化学习环境提供特征和收益数据
# - 影响生成因子的质量和性能
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GoldDataCollector:
    """Collector for gold trading data"""
    
    def __init__(
        self,
        data_path: str,
        features_config: Optional[Dict] = None,
        sample_size: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        self.features_config = features_config or {
            'price_features': ['open', 'high', 'low', 'close', 'volume'],
            'technical_features': ['ma', 'rsi', 'macd', 'bollinger']
        }
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw gold trading data"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # If sample_size is specified, only take that many rows
        if self.sample_size is not None:
            logger.info(f"Using first {self.sample_size} rows for testing")
            df = df.head(self.sample_size)
            
        # Convert datetime column
        if 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime'])
            df.drop('datetime', axis=1, inplace=True)
        else:
            df['date'] = pd.to_datetime(df['date'])
            
        df.set_index('date', inplace=True)
        
        # Drop any unused columns
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[columns_to_keep]
        
        return df
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change()
    
    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators as features"""
        features = df.copy()
        
        # Setup progress bar for feature generation
        feature_types = [ft for ft in self.features_config['technical_features']]
        pbar = tqdm(feature_types, desc="Generating features")
        
        for feature_type in pbar:
            pbar.set_postfix({"Feature": feature_type})
            
            if feature_type == 'ma':
                for window in [5, 10, 20, 60]:
                    features[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            
            elif feature_type == 'rsi':
                for window in [14, 28]:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    rs = gain / loss
                    features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            elif feature_type == 'macd':
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                features['macd'] = exp1 - exp2
                features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            
            elif feature_type == 'bollinger':
                for window in [20]:
                    mid = df['close'].rolling(window=window).mean()
                    std = df['close'].rolling(window=window).std()
                    features[f'bb_upper_{window}'] = mid + 2 * std
                    features[f'bb_lower_{window}'] = mid - 2 * std
        
        return features
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for factor generation"""
        # Load and process data
        df = self.load_raw_data()
        
        # Calculate returns
        logger.info("Calculating returns...")
        df['returns'] = self.calculate_returns(df['close'])
        
        # Generate features
        logger.info("Generating technical features...")
        features_df = self.generate_technical_features(df)
        
        # Drop any rows with NaN values
        initial_rows = len(features_df)
        features_df = features_df.dropna()
        final_rows = len(features_df)
        logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values")
        
        # Log data info
        logger.info(f"Data shape after processing: {features_df.shape}")
        
        # Separate features and returns
        feature_columns = [col for col in features_df.columns 
                         if col not in ['returns']]
        
        features = features_df[feature_columns].values
        returns = features_df['returns'].values
        
        return features, returns 