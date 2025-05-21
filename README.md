# AlphaGen - 黄金量化投资因子生成框架

AlphaGen是一个专为黄金市场设计的量化投资因子生成框架，支持多种因子生成方法，包括强化学习（PPO）、遗传规划（GP）和基于大语言模型（LLM）的方法。该框架专注于发现和评估可用于黄金交易的有效量化因子，帮助投资者在贵金属市场获取超额收益。

## 中文介绍

AlphaGen框架提供了一套针对黄金市场的完整工具，用于量化投资因子的生成、评估和应用。框架的主要特点包括：

### 1. 黄金市场专属因子生成方法
- 基于PPO的强化学习方法，适配黄金价格波动特性
- 遗传规划（GP）方法，挖掘黄金与宏观经济的关系
- 大语言模型（LLM）辅助生成，结合专家知识和市场规律

### 2. 完整的黄金因子评估体系
- 信息系数（IC）
- 排序信息系数（Rank IC）
- 信息比率（IR）
- 夏普比率
- 最大回撤
- 胜率
- 盈亏比
- NDCG (归一化折扣累积增益)

### 3. 黄金市场数据处理
- 支持多种黄金市场数据源
- 黄金价格与宏观经济指标的特征工程
- 高效的数据计算和缓存机制
- 整合黄金、美元指数、利率等关键影响因素

### 4. 性能优化
- 并行处理框架
- 数据和计算缓存
- 内存高效的分块处理
- 错误处理和重试机制
- 可配置的工作线程池

## Features (English)

### 1. Gold Market-Specific Factor Generation
- PPO-based Reinforcement Learning optimized for gold price fluctuations
- Genetic Programming (GP) for discovering relationships between gold and macro indicators
- LLM-assisted generation incorporating expert knowledge and market patterns

### 2. Comprehensive Factor Evaluation
- Information Coefficient (IC)
- Rank Information Coefficient
- Information Ratio (IR)
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit/Loss Ratio
- NDCG (Normalized Discounted Cumulative Gain)

### 3. Flexible Data Processing
- Support for multiple data sources
- Customizable feature engineering
- Efficient data computation and caching

### 4. Performance Optimization
- Parallel processing framework
- Data and computation caching
- Memory-efficient chunk processing
- Error handling and retry mechanisms
- Configurable worker pools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bos080808/alphagen.git
cd alphagen
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

## Project Structure

```
alphagen/
├── alphagen/                  # 核心框架
│   ├── data/                  # 数据处理和表达式计算
│   │   ├── alpha_calculator_base.py
│   │   ├── expression.py      # 表达式处理
│   │   ├── parser.py          # 表达式解析
│   │   ├── pool_update.py     # 因子池更新
│   │   ├── tokens.py          # 表达式token处理
│   │   └── tree.py            # 表达式树
│   ├── models/                # 模型组件
│   │   ├── alpha_pool.py      # 因子池基类
│   │   └── linear_alpha_pool.py # 线性因子池
│   ├── rl/                    # 强化学习组件
│   │   ├── env/               # RL环境
│   │   │   ├── core.py        # 环境核心
│   │   │   ├── factor_env.py  # 因子生成环境
│   │   │   └── wrapper.py     # 环境包装器
│   │   └── policy.py          # RL策略
│   ├── trade/                 # 交易相关
│   │   ├── base.py            # 交易基类
│   │   └── strategy.py        # 交易策略
│   └── utils/                 # 工具函数
│       ├── correlation.py     # 相关性计算
│       ├── logging.py         # 日志功能
│       ├── maybe.py           # 可选值处理
│       ├── misc.py            # 杂项功能
│       ├── parallel.py        # 并行处理
│       ├── pytorch_utils.py   # PyTorch工具
│       └── random.py          # 随机数生成
├── alphagen_generic/          # 通用数据处理
│   ├── data_collection/       # 数据收集
│   │   └── gold_data.py       # 黄金数据处理
│   └── features.py            # 特征工程
├── alphagen_llm/              # LLM因子生成
│   ├── client.py              # LLM客户端
│   ├── generator.py           # 因子生成器
│   └── prompts/               # 提示模板
│       ├── interaction.py     # 交互提示
│       └── system_prompt.py   # 系统提示
├── alphagen_qlib/             # QLib集成
│   ├── gold_data.py           # 黄金数据
│   ├── qlib_alpha_calculator.py # QLib因子计算器
│   ├── strategy.py            # QLib策略
│   └── utils.py               # QLib工具
├── configs/                   # 配置文件
│   └── gold_factor_config.yaml # 黄金因子配置
├── scripts/                   # 脚本文件
│   ├── dso.py                 # DSO算法脚本
│   ├── gold_factor_generation.py # 黄金因子生成
│   ├── gp.py                  # 遗传规划脚本
│   ├── llm_only.py            # 仅LLM脚本
│   ├── llm_test_validity.py   # LLM有效性测试
│   ├── rl.py                  # RL训练脚本
│   └── train_gold_factor.py   # 黄金因子训练
├── backtest.py                # 回测功能
├── config.py                  # 配置管理
├── requirements.txt
├── trade_decision.py          # 交易决策
└── README.md
```

## Usage Examples

### 1. 使用强化学习生成黄金因子

```python
from alphagen.rl.env.factor_env import FactorEnv
from alphagen.rl.policy import Policy
import alphagen.scripts.rl as rl

# 初始化黄金市场环境
env = FactorEnv(
    data_calculator=gold_calculator,  
    max_depth=3
)

# 初始化策略
policy = Policy(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# 训练
rl.train(
    env=env,
    policy=policy,
    num_episodes=1000,
    learning_rate=1e-4
)

# 获取生成的黄金因子
gold_factors = policy.generate_factors(env)
```

### 2. 使用LLM生成黄金因子

```python
from alphagen_llm.client import OpenAIClient
from alphagen_llm.generator import LLMFactorGenerator

# 初始化LLM客户端
client = OpenAIClient(api_key="your_api_key")

# 初始化黄金因子生成器
generator = LLMFactorGenerator(
    client=client,
    data_calculator=gold_calculator
)

# 生成黄金因子
gold_factors = generator.generate_factors(
    num_factors=10,
    market_context="黄金市场在过去一周受美联储加息影响出现大幅波动"
)
```

### 3. 黄金因子评估与回测

```python
from backtest import Backtest, BacktestConfig

# 配置黄金回测
config = BacktestConfig(
    start_time="2020-01-01",
    end_time="2023-12-31",
    alpha_path="factors/gold_alphas.json",
    initial_capital=1000000.0,
    trading_fee=0.0005  # 黄金交易费率
)

# 运行回测
backtest = Backtest(config)
results = backtest.run()
report = backtest.generate_report(results)
```

## 黄金市场特性

AlphaGen框架特别考虑了黄金市场的独特特性：

1. **宏观经济敏感性** - 黄金价格对利率、通胀和地缘政治事件高度敏感
2. **避险特性** - 市场恐慌情绪下的价格行为模式
3. **季节性模式** - 传统节日和季节性需求对价格的影响
4. **美元相关性** - 黄金与美元指数的负相关关系
5. **供需平衡** - 黄金生产和消费数据对价格的长期影响

## Contributing

欢迎通过GitHub Issues和Pull Requests贡献代码和提出问题。

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FRED API for economic data
- Yahoo Finance for gold market data
- World Gold Council for industry insights
- Hugging Face for transformer models
- The Qlib team for inspiration and some code structure

## Contact

For questions and feedback, please open an issue on GitHub.
