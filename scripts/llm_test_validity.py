#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# LLM有效因子率测试脚本 (LLM Factor Validity Test Script)
#
# 本脚本测试不同系统提示对LLM生成的因子有效率的影响。主要功能包括：
#
# 1. 测试不同系统提示模板
# 2. 生成多组因子并计算有效率
# 3. 分析不同提示对因子语法正确性和金融逻辑的影响
# 4. 输出综合比较报告
#
# 使用方法：
# python -m scripts.llm_test_validity --help
"""
import sys
import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from alphagen.utils import get_logger
from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser, ParseError
from alphagen_qlib.gold_data import GoldData, initialize_qlib
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.generator import LLMAlphaGenerator


# 定义不同的系统提示模板
PROMPT_TEMPLATES = {
    "basic": """
你是一个量化交易专家，请为黄金市场设计alpha因子。

Alpha因子是预测未来黄金价格走势的数学表达式。请生成5个不同的alpha因子。
每个因子必须有清晰的数学表达式。
    """,
    
    "detailed": """
你是一个量化交易专家，擅长设计黄金市场的alpha因子。

Alpha因子是指通过对黄金市场数据的分析，预测未来黄金价格走势的数学表达式。
这些因子通常通过对市场数据（如价格、交易量、未平仓合约等）进行数学运算和变换来创建。

请在回答中遵循以下准则：
1. 设计符合金融逻辑的黄金alpha因子，解释其背后的经济或市场原理
2. 考虑因子的时效性、稳定性和对黄金市场行为的反映
3. 使用具体的函数和数学表达式描述你的因子
4. 评估因子可能的优势、局限性和适用市场条件

可用的基础数据包括：
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量
- oi: 未平仓合约数量

可用的数学操作包括：
- 基本算术: +, -, *, /
- 滚动窗口函数: mean, std, min, max, sum, skew, kurt
- 时间序列操作: delta, delay
- 数学函数: log, abs, sign, power

请生成5个不同的黄金alpha因子。
    """,
    
    "syntax_focused": """
你是一个量化交易专家，擅长设计黄金市场的alpha因子。

请生成5个黄金alpha因子，使用以下严格的语法规则：

1. 基本特征:
   - open: 开盘价
   - high: 最高价
   - low: 最低价
   - close: 收盘价
   - volume: 成交量
   - oi: 未平仓合约数量

2. 单目运算符函数，格式为 func(arg):
   - log(x): 自然对数
   - abs(x): 绝对值
   - sign(x): 符号函数，返回-1,0,1
   - power(x, n): x的n次方

3. 双目运算符，格式为 func(arg1, arg2):
   - add(x, y): x + y
   - sub(x, y): x - y
   - mul(x, y): x * y
   - div(x, y): x / y

4. 滚动窗口操作，格式为 func(arg, n):
   - mean(x, n): n天均值
   - std(x, n): n天标准差
   - min(x, n): n天最小值
   - max(x, n): n天最大值
   - sum(x, n): n天求和
   - skew(x, n): n天偏度
   - kurt(x, n): n天峰度

5. 时序操作:
   - delta(x, n): x_t - x_(t-n)
   - delay(x, n): x_(t-n)

请确保每个因子的语法完全正确，并提供简要解释其在黄金市场的应用。
    """,
    
    "example_based": """
你是一个量化交易专家，擅长设计黄金市场的alpha因子。

下面是几个有效黄金alpha因子的例子：

1. `rank(decay_linear(delta(close, 2), 5))`
   这个因子计算黄金收盘价的2日变化，然后进行5日线性衰减加权，捕捉短期价格动量。

2. `div(mul(power(high, 2), low), mul(sum(close, 5), abs(delta(close, 3))))`
   这个因子结合了高价的平方与低价，除以5日收盘价总和与3日价格变化的绝对值的乘积，反映价格波动性。

3. `div(mul(volume, close), mean(volume, 10))`
   这个因子计算当日成交量与收盘价的乘积，除以10日平均成交量，识别成交量异常导致的价格变动。

请按照这些例子的语法格式，生成5个新的适用于黄金市场的alpha因子。确保使用正确的函数名和参数顺序。
    """,
    
    "financial_logic": """
你是一个量化交易专家，擅长设计基于金融理论的黄金alpha因子。

请生成5个具有坚实金融逻辑基础的黄金alpha因子，每个因子应当:

1. 基于以下至少一种黄金市场异象或效应:
   - 避险效应: 风险事件发生时黄金往往表现出强势
   - 美元相关性: 黄金与美元往往呈负相关关系
   - 实际利率敏感性: 黄金对实际利率变化高度敏感
   - 技术突破: 突破关键价格水平后可能继续延续趋势
   - 季节性模式: 黄金在特定季节或月份可能有特定表现模式

2. 使用可用的基础数据:
   - open: 开盘价
   - high: 最高价
   - low: 最低价
   - close: 收盘价
   - volume: 成交量
   - oi: 未平仓合约数量

3. 使用清晰的数学表达式，采用函数嵌套格式

请对每个因子进行详细解释，说明其在黄金市场的金融逻辑，以及为何这种异象可能会持续存在。
    """
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test LLM factor generation validity")
    
    # 数据配置
    parser.add_argument("--qlib_data", type=str, default="~/.qlib/qlib_data/cn_data_2024h1",
                        help="Path to qlib data directory")
    parser.add_argument("--instruments", type=str, default="csi300",
                        help="Instruments to use")
    parser.add_argument("--data_start", type=str, default="2022-01-01",
                        help="Data start date")
    parser.add_argument("--data_end", type=str, default="2023-06-30",
                        help="Data end date")
                        
    # LLM配置
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM temperature")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (default: use environment variable)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="OpenAI API base URL (default: use environment variable)")
                        
    # 测试配置
    parser.add_argument("--n_attempts", type=int, default=3,
                        help="Number of attempts per prompt template")
    parser.add_argument("--custom_prompts", type=str, default=None,
                        help="Path to custom prompts JSON file")
    
    # 输出配置
    parser.add_argument("--save_path", type=str, default="outputs/llm_validity",
                        help="Path to save results")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def setup_logger(args):
    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, f"validity_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger("llm_validity_test", log_file)
    logger.setLevel(getattr(logging, args.log_level))
    return logger


def load_custom_prompts(path: Optional[str]) -> Dict[str, str]:
    """加载自定义提示模板"""
    if not path:
        return {}
        
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_prompt_template(
    template_name: str,
    template: str,
    client: ChatClient,
    parser: ExpressionParser,
    calculator: Optional[QLibGoldDataCalculator] = None,
    n_attempts: int = 3
) -> Dict[str, Any]:
    """测试单个提示模板的有效性"""
    factors_generated = 0
    factors_valid_syntax = 0
    factors_valid_execution = 0
    factors_with_ic = 0
    
    all_expressions = []
    valid_expressions = []
    execution_errors = []
    
    for attempt in range(n_attempts):
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": "请生成5个不同的alpha因子，确保语法正确。"}
        ]
        
        response = client.chat(messages)
        content = response.get("content", "")
        
        # 提取表达式
        expression_patterns = [
            r'`([^`]+)`',  # Markdown代码块
            r'"([^"]+)"',  # 双引号表达式
            r'表达式[:：]\s*([^\n]+)',  # 中文标记
            r'Factor[:：]\s*([^\n]+)',  # 英文标记
            r'Alpha[:：]\s*([^\n]+)',   # Alpha标记
            r'([a-zA-Z0-9_\(\)\+\-\*\/\.\,\s]+\([a-zA-Z0-9_\(\)\+\-\*\/\.\,\s]+\))'  # 函数式表达式
        ]
        
        potential_expressions = []
        for pattern in expression_patterns:
            matches = re.findall(pattern, content)
            potential_expressions.extend(matches)
        
        # 去重并验证
        unique_expressions = list(set(potential_expressions))
        factors_generated += len(unique_expressions)
        
        for expr_text in unique_expressions:
            all_expressions.append(expr_text)
            
            # 验证语法
            try:
                expr = parser.parse(expr_text.strip())
                factors_valid_syntax += 1
                valid_expressions.append(expr_text)
                
                # 验证执行
                if calculator:
                    try:
                        ic = calculator.calc_single_IC_ret(expr)
                        factors_valid_execution += 1
                        
                        if not np.isnan(ic):
                            factors_with_ic += 1
                    except Exception as e:
                        execution_errors.append(f"{expr_text}: {str(e)}")
                
            except Exception as e:
                pass
    
    # 计算统计信息
    if factors_generated == 0:
        syntax_valid_rate = 0.0
        execution_valid_rate = 0.0
        ic_valid_rate = 0.0
    else:
        syntax_valid_rate = factors_valid_syntax / factors_generated
        execution_valid_rate = factors_valid_execution / factors_generated if calculator else None
        ic_valid_rate = factors_with_ic / factors_generated if calculator else None
    
    return {
        "template_name": template_name,
        "factors_generated": factors_generated,
        "factors_valid_syntax": factors_valid_syntax,
        "factors_valid_execution": factors_valid_execution if calculator else None,
        "factors_with_ic": factors_with_ic if calculator else None,
        "syntax_valid_rate": syntax_valid_rate,
        "execution_valid_rate": execution_valid_rate,
        "ic_valid_rate": ic_valid_rate,
        "all_expressions": all_expressions,
        "valid_expressions": valid_expressions,
        "execution_errors": execution_errors
    }


def main():
    args = parse_args()
    logger = setup_logger(args)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载自定义提示
    custom_prompts = load_custom_prompts(args.custom_prompts)
    all_prompts = {**PROMPT_TEMPLATES, **custom_prompts}
    
    logger.info(f"Testing {len(all_prompts)} prompt templates")
    
    # 初始化qlib数据
    initialize_qlib(args.qlib_data)
    
    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据
    data = GoldData(
        args.instruments,
        args.data_start,
        args.data_end,
        device=device
    )
    
    # 创建计算器
    calculator = QLibGoldDataCalculator(data, target="20d")
    
    # 设置LLM客户端
    logger.info(f"Setting up LLM client with model: {args.model}")
    config = ChatConfig(
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
        api_base=args.api_base
    )
    client = OpenAIClient(config)
    
    # 创建表达式解析器
    parser = ExpressionParser()
    
    # 测试所有提示模板
    results = []
    for name, template in all_prompts.items():
        logger.info(f"Testing template: {name}")
        result = test_prompt_template(
            template_name=name,
            template=template,
            client=client,
            parser=parser,
            calculator=calculator,
            n_attempts=args.n_attempts
        )
        results.append(result)
        
        logger.info(f"  Generated: {result['factors_generated']}")
        logger.info(f"  Valid syntax: {result['factors_valid_syntax']} ({result['syntax_valid_rate']:.2%})")
        if result['execution_valid_rate'] is not None:
            logger.info(f"  Valid execution: {result['factors_valid_execution']} ({result['execution_valid_rate']:.2%})")
            logger.info(f"  With IC: {result['factors_with_ic']} ({result['ic_valid_rate']:.2%})")
    
    # 保存结果
    result_dir = os.path.join(args.save_path, f"validity_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存原始结果
    with open(os.path.join(result_dir, "raw_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存摘要数据
    summary_csv = os.path.join(result_dir, "summary.csv")
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Template", "Generated", "Valid Syntax", "Valid Execution", "With IC",
            "Syntax Rate", "Execution Rate", "IC Rate"
        ])
        
        for result in results:
            writer.writerow([
                result["template_name"],
                result["factors_generated"],
                result["factors_valid_syntax"],
                result["factors_valid_execution"] if result["factors_valid_execution"] is not None else "N/A",
                result["factors_with_ic"] if result["factors_with_ic"] is not None else "N/A",
                f"{result['syntax_valid_rate']:.2%}",
                f"{result['execution_valid_rate']:.2%}" if result["execution_valid_rate"] is not None else "N/A",
                f"{result['ic_valid_rate']:.2%}" if result["ic_valid_rate"] is not None else "N/A"
            ])
    
    # 创建可视化图表
    plt.figure(figsize=(10, 6))
    template_names = [r["template_name"] for r in results]
    syntax_rates = [r["syntax_valid_rate"] * 100 for r in results]
    
    execution_rates = []
    ic_rates = []
    for r in results:
        if r["execution_valid_rate"] is not None:
            execution_rates.append(r["execution_valid_rate"] * 100)
            ic_rates.append(r["ic_valid_rate"] * 100)
        else:
            execution_rates.append(0)
            ic_rates.append(0)
    
    x = np.arange(len(template_names))
    width = 0.25
    
    plt.bar(x - width, syntax_rates, width, label='Valid Syntax %')
    plt.bar(x, execution_rates, width, label='Valid Execution %')
    plt.bar(x + width, ic_rates, width, label='With IC %')
    
    plt.ylabel('Percentage (%)')
    plt.title('Effect of Different Prompt Templates on Factor Validity')
    plt.xticks(x, template_names, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "validity_comparison.png"), dpi=300)
    
    logger.info(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main() 