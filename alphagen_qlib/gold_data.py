from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import yfinance as yf
import fredapi as fred
import logging
import os
import qlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from alphagen.utils.parallel import ParallelExecutor, retry_on_error, parallel_cache
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_qlib(provider_uri: str, region: str = 'cn'):
    """Initialize the qlib library for gold data
    
    Args:
        provider_uri: The path to the gold data directory
        region: The region of the data
    """
    try:
        qlib.init(provider_uri=provider_uri, region=region)
        logger.info(f"QLib initialized with provider_uri={provider_uri}")
    except Exception as e:
        logger.error(f"Failed to initialize QLib: {str(e)}")
        raise

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    INTEREST_RATE = 6
    USD_INDEX = 7
    INFLATION = 8
    SILVER_PRICE = 9      # Silver often correlates with gold
    OIL_PRICE = 10        # Oil can impact gold prices
    VIX = 11             # Market volatility indicator
    REAL_RATES = 12      # Real interest rates (nominal - inflation)
    OI = 13              # Open Interest for gold futures

class GoldData:
    # FRED API series IDs
    FRED_SERIES = {
        'INTEREST_RATE': 'DFF',    # Federal Funds Rate
        'USD_INDEX': 'DTWEXB',     # USD Index
        'INFLATION': 'CPIAUCSL',   # CPI
        'REAL_RATES': 'DFII10',    # 10-Year Treasury Inflation-Indexed Security
    }
    
    # Yahoo Finance symbols
    YAHOO_SYMBOLS = {
        'GOLD': 'GC=F',           # Gold Futures
        'SILVER': 'SI=F',         # Silver Futures
        'OIL': 'CL=F',           # Crude Oil Futures
        'VIX': '^VIX',           # Volatility Index
    }

    def __init__(
        self,
        start_time: str,
        end_time: str,
        data_path: Optional[str] = None,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cuda:0"),
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index]] = None,
        fred_api_key: Optional[str] = None,
        cache_dir: str = ".cache/gold_data",
        n_workers: Optional[int] = None
    ) -> None:
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self._fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self._cache_dir = cache_dir
        self._parallel_executor = ParallelExecutor(
            use_processes=False,  # Use threads for I/O-bound operations
            max_workers=n_workers
        )
        
        if not self._fred_api_key:
            logger.warning("FRED API key not provided. Economic indicators will not be available.")
        
        if preloaded_data is not None:
            self.data, self._dates = preloaded_data
        elif data_path:
            self.data, self._dates = self._load_from_csv(data_path)
        else:
            self.data, self._dates = self._get_data()

    @retry_on_error(max_retries=3, retry_delay=1.0)
    @parallel_cache(cache_dir=".cache/yahoo_finance")
    def _get_yahoo_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with caching and retry"""
        data = yf.download(
            symbol,
            start=self._adjust_date(self.start_time, -self.max_backtrack_days),
            end=self._adjust_date(self.end_time, self.max_future_days),
            progress=False
        )
        if data.empty:
            raise ValueError(f"No data returned for symbol {symbol}")
        return data

    def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch all market data in parallel"""
        return self._parallel_executor.map_dict(
            lambda name, symbol: self._get_yahoo_data(symbol),
            self.YAHOO_SYMBOLS
        )

    @retry_on_error(max_retries=3, retry_delay=1.0)
    @parallel_cache(cache_dir=".cache/fred")
    def _get_fred_data(self, series_id: str) -> pd.Series:
        """Fetch data from FRED with caching and retry"""
        if not self._fred_api_key:
            return pd.Series()
        
        fred_client = fred.Fred(api_key=self._fred_api_key)
        data = fred_client.get_series(
            series_id,
            observation_start=self._adjust_date(self.start_time, -self.max_backtrack_days),
            observation_end=self._adjust_date(self.end_time, self.max_future_days)
        )
        return data

    def _get_economic_indicators(self) -> Dict[str, pd.Series]:
        """Fetch all economic indicators in parallel"""
        return self._parallel_executor.map_dict(
            lambda name, series_id: self._get_fred_data(series_id),
            self.FRED_SERIES
        )

    def _adjust_date(self, date: pd.Timestamp, days: int) -> pd.Timestamp:
        """Adjust date by given number of days"""
        adjusted = date + pd.Timedelta(days=days)
        return adjusted

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index]:
        # Get all market data
        market_data = self._get_market_data()
        gold_data = market_data.get('GOLD')
        
        if gold_data is None or gold_data.empty:
            raise ValueError("Failed to fetch gold price data")

        # Get economic indicators
        indicators = self._get_economic_indicators()
        
        # Process features in parallel
        def process_feature(feature: FeatureType) -> np.ndarray:
            try:
                if feature in [FeatureType.OPEN, FeatureType.HIGH, FeatureType.LOW, 
                             FeatureType.CLOSE, FeatureType.VOLUME]:
                    return gold_data[feature.name.capitalize()].values
                elif feature == FeatureType.VWAP:
                    return ((gold_data['High'] + gold_data['Low']) / 2).values
                elif feature == FeatureType.SILVER_PRICE:
                    return market_data['SILVER']['Close'].reindex(gold_data.index).ffill().values
                elif feature == FeatureType.OIL_PRICE:
                    return market_data['OIL']['Close'].reindex(gold_data.index).ffill().values
                elif feature == FeatureType.VIX:
                    return market_data['VIX']['Close'].reindex(gold_data.index).ffill().values
                elif feature == FeatureType.REAL_RATES:
                    if 'INTEREST_RATE' in indicators and 'INFLATION' in indicators:
                        nominal = indicators['INTEREST_RATE'].reindex(gold_data.index).ffill()
                        inflation = indicators['INFLATION'].reindex(gold_data.index).ffill()
                        return (nominal - inflation.pct_change(12)).values
                    else:
                        return indicators['REAL_RATES'].reindex(gold_data.index).ffill().values
                else:
                    indicator_name = feature.name
                    return indicators[indicator_name].reindex(gold_data.index).ffill().values
            except Exception as e:
                logger.error(f"Error processing feature {feature.name}: {str(e)}")
                return np.zeros(len(gold_data))

        features_list = self._parallel_executor.map(
            process_feature,
            self._features,
            show_progress=True
        )

        # Convert to tensor
        data = np.stack(features_list, axis=1)
        tensor_data = torch.tensor(data, dtype=torch.float, device=self.device)
        
        return tensor_data, gold_data.index

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert tensor data to DataFrame"""
        if isinstance(data, list):
            data = torch.stack(data, dim=1)
        if len(data.shape) == 1:
            data = data.unsqueeze(1)
        if columns is None:
            columns = [str(i) for i in range(data.shape[1])]
            
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
            
        return pd.DataFrame(
            data.detach().cpu().numpy(),
            index=date_index,
            columns=columns
        )

    def _load_from_csv(self, data_path: str) -> Tuple[torch.Tensor, pd.Index]:
        """Load data from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(data_path)
            
            # Ensure date column exists and convert to datetime
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            if not date_col:
                raise ValueError("No date column found in CSV file")
            
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Filter data based on date range
            df = df[(df.index >= self.start_time) & (df.index <= self.end_time)]
            
            # Validate required columns
            required_columns = {
                'open': FeatureType.OPEN,
                'high': FeatureType.HIGH,
                'low': FeatureType.LOW,
                'close': FeatureType.CLOSE,
                'volume': FeatureType.VOLUME
            }
            
            for col, feature in required_columns.items():
                if not any(c.lower() == col for c in df.columns):
                    raise ValueError(f"Required column {col} not found in CSV file")
            
            # Create feature mapping
            feature_mapping = {}
            for i, feature in enumerate(self._features):
                if feature == FeatureType.OPEN:
                    col = next((c for c in df.columns if c.lower() == 'open'), None)
                    feature_mapping[col] = i
                elif feature == FeatureType.HIGH:
                    col = next((c for c in df.columns if c.lower() == 'high'), None)
                    feature_mapping[col] = i
                elif feature == FeatureType.LOW:
                    col = next((c for c in df.columns if c.lower() == 'low'), None)
                    feature_mapping[col] = i
                elif feature == FeatureType.CLOSE:
                    col = next((c for c in df.columns if c.lower() == 'close'), None)
                    feature_mapping[col] = i
                elif feature == FeatureType.VOLUME:
                    col = next((c for c in df.columns if c.lower() == 'volume'), None)
                    feature_mapping[col] = i
                elif feature == FeatureType.OI:
                    # Look for open_interest column
                    col = next((c for c in df.columns if 'open_interest' in c.lower()), None)
                    if col:
                        feature_mapping[col] = i
                elif feature == FeatureType.VWAP:
                    # Calculate VWAP from other columns if not present
                    col = next((c for c in df.columns if c.lower() == 'vwap'), None)
                    if not col:
                        logger.info("VWAP not found in data, calculating from high and low")
                        df['vwap'] = (df['high'] + df['low']) / 2
                        feature_mapping['vwap'] = i
                    else:
                        feature_mapping[col] = i
            
            # Handle optional columns
            optional_features = {
                'interest_rate': FeatureType.INTEREST_RATE,
                'usd_index': FeatureType.USD_INDEX,
                'inflation': FeatureType.INFLATION,
                'vix': FeatureType.VIX
            }
            
            for col, feature in optional_features.items():
                matching_cols = [c for c in df.columns if col.lower() in c.lower()]
                if matching_cols and feature in self._features:
                    feature_index = self._features.index(feature)
                    feature_mapping[matching_cols[0]] = feature_index
            
            # Create tensor data
            self.feature_names = [f.name for f in self._features]
            tensor_data = torch.zeros((len(df), len(self._features)), dtype=torch.float32, device=self.device)
            
            for col, feature_idx in feature_mapping.items():
                tensor_data[:, feature_idx] = torch.tensor(df[col].values, dtype=torch.float32, device=self.device)
            
            return tensor_data, df.index
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def get_features(self) -> np.ndarray:
        """Get feature data"""
        return self.data

    def get_feature_names(self) -> List[str]:
        """Get list of available feature names"""
        return self.feature_names 