import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import torch
import logging

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings"""
    fred_api_key: str = os.getenv('FRED_API_KEY', '')
    huggingface_api_key: str = os.getenv('HUGGINGFACE_API_KEY', '')

@dataclass
class DataConfig:
    """Data source configuration"""
    data_source_type: str = os.getenv('DATA_SOURCE_TYPE', 'csv')  # 'csv' or 'api'
    csv_data_path: str = os.getenv('CSV_DATA_PATH', '')
    gold_data_source: str = os.getenv('GOLD_DATA_SOURCE', 'yfinance')
    economic_data_source: str = os.getenv('ECONOMIC_DATA_SOURCE', 'fred')
    max_backtrack_days: int = int(os.getenv('MAX_BACKTRACK_DAYS', '100'))
    max_future_days: int = int(os.getenv('MAX_FUTURE_DAYS', '30'))
    
    def validate(self):
        """Validate data configuration"""
        if self.data_source_type == 'csv':
            if not self.csv_data_path:
                raise ValueError("CSV data path must be provided when using CSV data source")
            if not os.path.exists(self.csv_data_path):
                raise ValueError(f"CSV file not found: {self.csv_data_path}")
        elif self.data_source_type == 'api':
            if not self.gold_data_source:
                raise ValueError("Gold data source must be specified when using API data source")
        else:
            raise ValueError(f"Invalid data source type: {self.data_source_type}")

@dataclass
class ModelConfig:
    """Model configuration settings"""
    llm_model_name: str = os.getenv('LLM_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2')
    device: torch.device = torch.device(os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))

@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    initial_capital: float = float(os.getenv('INITIAL_CAPITAL', '1000000.0'))
    position_size: float = float(os.getenv('POSITION_SIZE', '0.1'))
    leverage: float = float(os.getenv('LEVERAGE', '1.0'))
    transaction_cost: float = float(os.getenv('TRANSACTION_COST', '0.0001'))
    stop_loss: float = float(os.getenv('STOP_LOSS', '0.02'))
    take_profit: float = float(os.getenv('TAKE_PROFIT', '0.05'))
    trailing_stop: float = float(os.getenv('TRAILING_STOP', '0.01'))
    risk_per_trade: float = float(os.getenv('RISK_PER_TRADE', '0.02'))
    max_positions: int = int(os.getenv('MAX_POSITIONS', '1'))

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '252'))
    max_workers: Optional[int] = int(os.getenv('MAX_WORKERS', '0')) or None
    cache_dir: str = os.getenv('CACHE_DIR', '.cache')
    data_cache_ttl: int = int(os.getenv('DATA_CACHE_TTL', '86400'))
    model_cache_ttl: int = int(os.getenv('MODEL_CACHE_TTL', '604800'))

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    file: str = os.getenv('LOG_FILE', 'alphagen.log')

    def setup(self):
        """Configure logging settings"""
        numeric_level = getattr(logging, self.level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.file),
                logging.StreamHandler()
            ]
        )

class Config:
    """Global configuration manager"""
    
    def __init__(self):
        self.api = APIConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.trading = TradingConfig()
        self.backtest = BacktestConfig()
        self.logging = LoggingConfig()
        
        # Setup logging
        self.logging.setup()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'api': {
                'fred_api_key': '***' if self.api.fred_api_key else None,
                'huggingface_api_key': '***' if self.api.huggingface_api_key else None
            },
            'data': {
                'data_source_type': self.data.data_source_type,
                'csv_data_path': self.data.csv_data_path,
                'gold_data_source': self.data.gold_data_source,
                'economic_data_source': self.data.economic_data_source,
                'max_backtrack_days': self.data.max_backtrack_days,
                'max_future_days': self.data.max_future_days
            },
            'model': {
                'llm_model_name': self.model.llm_model_name,
                'device': str(self.model.device)
            },
            'trading': {
                'initial_capital': self.trading.initial_capital,
                'position_size': self.trading.position_size,
                'leverage': self.trading.leverage,
                'transaction_cost': self.trading.transaction_cost,
                'stop_loss': self.trading.stop_loss,
                'take_profit': self.trading.take_profit,
                'trailing_stop': self.trading.trailing_stop,
                'risk_per_trade': self.trading.risk_per_trade,
                'max_positions': self.trading.max_positions
            },
            'backtest': {
                'chunk_size': self.backtest.chunk_size,
                'max_workers': self.backtest.max_workers,
                'cache_dir': self.backtest.cache_dir,
                'data_cache_ttl': self.backtest.data_cache_ttl,
                'model_cache_ttl': self.backtest.model_cache_ttl
            },
            'logging': {
                'level': self.logging.level,
                'file': self.logging.file
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        config = cls()
        
        if 'api' in config_dict:
            api = config_dict['api']
            if 'fred_api_key' in api:
                config.api.fred_api_key = api['fred_api_key']
            if 'huggingface_api_key' in api:
                config.api.huggingface_api_key = api['huggingface_api_key']
        
        if 'data' in config_dict:
            data = config_dict['data']
            if 'data_source_type' in data:
                config.data.data_source_type = data['data_source_type']
            if 'csv_data_path' in data:
                config.data.csv_data_path = data['csv_data_path']
            if 'gold_data_source' in data:
                config.data.gold_data_source = data['gold_data_source']
            if 'economic_data_source' in data:
                config.data.economic_data_source = data['economic_data_source']
            if 'max_backtrack_days' in data:
                config.data.max_backtrack_days = int(data['max_backtrack_days'])
            if 'max_future_days' in data:
                config.data.max_future_days = int(data['max_future_days'])
        
        if 'model' in config_dict:
            model = config_dict['model']
            if 'llm_model_name' in model:
                config.model.llm_model_name = model['llm_model_name']
            if 'device' in model:
                config.model.device = torch.device(model['device'])
        
        if 'trading' in config_dict:
            trading = config_dict['trading']
            if 'initial_capital' in trading:
                config.trading.initial_capital = float(trading['initial_capital'])
            if 'position_size' in trading:
                config.trading.position_size = float(trading['position_size'])
            if 'leverage' in trading:
                config.trading.leverage = float(trading['leverage'])
            if 'transaction_cost' in trading:
                config.trading.transaction_cost = float(trading['transaction_cost'])
            if 'stop_loss' in trading:
                config.trading.stop_loss = float(trading['stop_loss'])
            if 'take_profit' in trading:
                config.trading.take_profit = float(trading['take_profit'])
            if 'trailing_stop' in trading:
                config.trading.trailing_stop = float(trading['trailing_stop'])
            if 'risk_per_trade' in trading:
                config.trading.risk_per_trade = float(trading['risk_per_trade'])
            if 'max_positions' in trading:
                config.trading.max_positions = int(trading['max_positions'])
        
        if 'backtest' in config_dict:
            backtest = config_dict['backtest']
            if 'chunk_size' in backtest:
                config.backtest.chunk_size = int(backtest['chunk_size'])
            if 'max_workers' in backtest:
                config.backtest.max_workers = int(backtest['max_workers'])
            if 'cache_dir' in backtest:
                config.backtest.cache_dir = backtest['cache_dir']
            if 'data_cache_ttl' in backtest:
                config.backtest.data_cache_ttl = int(backtest['data_cache_ttl'])
            if 'model_cache_ttl' in backtest:
                config.backtest.model_cache_ttl = int(backtest['model_cache_ttl'])
        
        if 'logging' in config_dict:
            logging = config_dict['logging']
            if 'level' in logging:
                config.logging.level = logging['level']
            if 'file' in logging:
                config.logging.file = logging['file']
        
        return config

# Global configuration instance
config = Config() 