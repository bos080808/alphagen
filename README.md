# AlphaGen Gold - Gold Factor Mining System

AlphaGen Gold is a sophisticated factor mining system specifically designed for gold trading. It combines traditional financial analysis with modern machine learning techniques to discover and evaluate trading factors for the gold market.

## Features

### 1. Data Integration
- Gold futures data from Yahoo Finance
- Economic indicators from FRED API (Federal Reserve Economic Data)
- Support for multiple data sources and custom data providers
- Comprehensive feature set including:
  - Price and volume data
  - Interest rates
  - USD Index
  - Inflation rates
  - Market volatility (VIX)
  - Correlated assets (Silver, Oil)

### 2. Factor Generation and Evaluation
- Advanced factor generation using genetic programming
- LLM-powered factor optimization and analysis
- Comprehensive factor evaluation metrics:
  - Information Coefficient (IC)
  - Rank Information Coefficient
  - Information Ratio (IR)
  - Sharpe Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit/Loss Ratio
  - NDCG (Normalized Discounted Cumulative Gain)

### 3. Risk Management
- Position sizing based on risk parameters
- Stop-loss and take-profit mechanisms
- Trailing stop implementation
- Maximum drawdown control
- Leverage management
- Portfolio exposure limits

### 4. Performance Optimization
- Parallel processing framework
- Data and computation caching
- Memory-efficient chunk processing
- Error handling and retry mechanisms
- Configurable worker pools

### 5. Machine Learning Integration
- Deep learning factor ranking model
- LLM-based factor analysis
- Semantic similarity scoring
- Factor complexity evaluation
- Economic soundness assessment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alphagen-gold.git
cd alphagen-gold
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

The system is highly configurable through environment variables or the `config.py` file. Key configuration areas include:

- API credentials
- Data sources
- Model parameters
- Trading rules
- Risk management settings
- Performance optimization
- Logging preferences

See `.env.example` for all available configuration options.

## Usage

### 1. Data Collection
```python
from alphagen_qlib.gold_data import GoldData

# Initialize data collector
data = GoldData(
    start_time="2020-01-01",
    end_time="2023-12-31",
    max_backtrack_days=100,
    max_future_days=30
)

# Get feature data
features = data.get_features()
```

### 2. Factor Generation
```python
from alphagen.models.factor_evaluator import FactorEvaluator
from alphagen_qlib.calculator import QLibGoldDataCalculator

# Initialize evaluator
evaluator = FactorEvaluator(
    price_data=price_data,
    factor_data=factor_data,
    llm_evaluator=LLMFactorEvaluator()
)

# Evaluate factors
results = evaluator.evaluate_factors(factors)
```

### 3. Backtesting
```python
from backtest import Backtest, BacktestConfig

# Configure backtest
config = BacktestConfig(
    start_time="2020-01-01",
    end_time="2023-12-31",
    alpha_path="factors/gold_alphas.json",
    initial_capital=1000000.0
)

# Run backtest
backtest = Backtest(config)
results = backtest.run()
report = backtest.generate_report(results)
```

## Project Structure

```
alphagen-gold/
├── alphagen/
│   ├── data/
│   │   ├── expression.py
│   │   └── parser.py
│   ├── models/
│   │   ├── factor_evaluator.py
│   │   └── linear_alpha_pool.py
│   └── utils/
│       ├── parallel.py
│       └── pytorch_utils.py
├── alphagen_qlib/
│   ├── gold_data.py
│   ├── calculator.py
│   └── utils.py
├── scripts/
│   ├── train_model.py
│   └── generate_factors.py
├── tests/
│   └── ...
├── .env.example
├── config.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FRED API for economic data
- Yahoo Finance for market data
- Hugging Face for transformer models
- The Qlib team for inspiration and some code structure

## Contact

For questions and feedback, please open an issue on GitHub.
