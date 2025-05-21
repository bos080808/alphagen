#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# LLM因子生成脚本 (LLM Factor Generation Script)
#
# 本脚本实现了纯基于LLM交互式生成量化因子的流程。主要功能包括：
#
# 1. 初始化市场数据环境
# 2. 设置LLM客户端
# 3. 通过多轮对话生成因子
# 4. 评估生成的因子表现
# 5. 保存结果到因子池
#
# 使用方法：
# python -m scripts.llm_only --help
"""
import sys
import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any

import torch
import numpy as np

from alphagen.utils import get_logger
from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool
from alphagen_qlib.gold_data import GoldData, initialize_qlib
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.generator import LLMAlphaGenerator
from alphagen_llm.prompts.system_prompt import ALPHA_GENERATOR_PROMPT, EXPLAIN_WITH_TEXT_DESC


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based alpha factor generation")
    
    # 数据配置
    parser.add_argument("--qlib_data", type=str, default="~/.qlib/qlib_data/cn_data_2024h1",
                        help="Path to qlib data directory")
    parser.add_argument("--instruments", type=str, default="csi300",
                        help="Instruments to use")
    parser.add_argument("--train_start", type=str, default="2012-01-01",
                        help="Training data start date")
    parser.add_argument("--train_end", type=str, default="2021-12-31",
                        help="Training data end date")
    parser.add_argument("--test_start", type=str, default="2022-01-01",
                        help="Test data start date")
    parser.add_argument("--test_end", type=str, default="2023-06-30",
                        help="Test data end date")
                        
    # LLM配置
    parser.add_argument("--model", type=str, default="gpt-4",
                        help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM temperature")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (default: use environment variable)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="OpenAI API base URL (default: use environment variable)")
                        
    # 生成配置
    parser.add_argument("--n_factors", type=int, default=10,
                        help="Number of factors to generate")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of improvement iterations per factor")
    parser.add_argument("--market_context", type=str, default=None,
                        help="Path to market context file")
    parser.add_argument("--pool_capacity", type=int, default=20,
                        help="Alpha pool capacity")
    parser.add_argument("--ic_lower_bound", type=float, default=None,
                        help="Minimum IC for factors")
    
    # 输出配置
    parser.add_argument("--save_path", type=str, default="outputs/llm",
                        help="Path to save results")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def setup_logger(args):
    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, f"llm_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger("llm_generation", log_file)
    logger.setLevel(getattr(logging, args.log_level))
    return logger


def load_market_context(context_path: Optional[str]) -> str:
    """加载市场背景信息，用于提供给LLM"""
    if not context_path:
        return "生成适用于黄金期货市场的alpha因子，考虑黄金的避险属性、与宏观经济的关系以及周期性波动特点。"
        
    with open(context_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    args = parse_args()
    logger = setup_logger(args)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("Initializing data environment...")
    
    # 初始化qlib数据
    initialize_qlib(args.qlib_data)
    
    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载数据
    data_train = GoldData(
        args.instruments, 
        args.train_start, 
        args.train_end, 
        device=device
    )
    data_test = GoldData(
        args.instruments, 
        args.test_start, 
        args.test_end, 
        device=device
    )
    
    # 创建计算器和因子池
    calculator_train = QLibGoldDataCalculator(data_train, target="20d")
    calculator_test = QLibGoldDataCalculator(data_test, target="20d")
    
    pool = LinearAlphaPool(
        capacity=args.pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=args.ic_lower_bound,
        device=device
    )
    
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
    
    # 创建LLM因子生成器
    generator = LLMAlphaGenerator(
        client=client,
        pool=pool,
        parser=parser,
        system_prompt=ALPHA_GENERATOR_PROMPT
    )
    
    # 加载市场背景信息
    market_context = load_market_context(args.market_context)
    
    # 生成因子
    logger.info(f"Starting factor generation, aiming for {args.n_factors} factors...")
    
    generated_factors = []
    initial_prompt = f"""
请设计用于量化交易的黄金市场alpha因子。

市场背景:
{market_context}

请生成3个不同的alpha因子，每个因子需要:
1. 清晰的数学表达式
2. 背后的金融/市场逻辑
3. 预期的表现特点

可用的特征包括:
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量
- oi: 未平仓合约数量
"""
    
    # 初始生成
    factors = generator.generate_factors(
        prompt=initial_prompt,
        n_attempts=max(2, (args.n_factors + 2) // 3)
    )
    
    # 评估和改进
    factors = generator.evaluate_factors(factors)
    valid_factors = [f for f in factors if f.valid]
    
    logger.info(f"Initial generation produced {len(valid_factors)} valid factors out of {len(factors)}")
    
    # 迭代改进
    if valid_factors:
        improved_factors = generator.iterative_improvement(
            valid_factors, 
            n_iterations=args.iterations
        )
        generated_factors.extend(improved_factors)
    
    # 如果因子数量不够，尝试更多的生成
    while len(generated_factors) < args.n_factors:
        remaining = args.n_factors - len(generated_factors)
        logger.info(f"Need {remaining} more factors. Generating...")
        
        feedback_prompt = f"""
前面生成的因子不够或表现不佳。请生成更多不同的黄金市场alpha因子，注意:
1. 避免过于复杂的表达式
2. 考虑黄金特有的市场异象和交易信号（如避险需求、与美元和实际利率的关系）
3. 利用不同的时间窗口捕捉黄金的周期性和波动特性
4. 确保表达式语法正确
"""
        new_factors = generator.generate_factors(
            prompt=feedback_prompt,
            n_attempts=max(1, (remaining + 1) // 2)
        )
        
        new_factors = generator.evaluate_factors(new_factors)
        valid_new_factors = [f for f in new_factors if f.valid]
        
        if valid_new_factors:
            improved_new_factors = generator.iterative_improvement(
                valid_new_factors, 
                n_iterations=args.iterations
            )
            generated_factors.extend(improved_new_factors)
        
        # 防止无限循环
        if not valid_new_factors:
            logger.warning("Failed to generate more valid factors, breaking loop")
            break
    
    # 保存结果
    result_dir = os.path.join(args.save_path, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存因子池
    pool_file = os.path.join(result_dir, "alpha_pool.json")
    with open(pool_file, 'w', encoding='utf-8') as f:
        json.dump(pool.to_json_dict(), f, indent=2, ensure_ascii=False)
    
    # 保存生成的因子详情
    factors_file = os.path.join(result_dir, "generated_factors.json")
    factors_json = []
    for i, factor in enumerate(generated_factors):
        factor_data = {
            "id": i,
            "expression": factor.expression_text,
            "description": factor.description,
            "metrics": factor.metrics
        }
        factors_json.append(factor_data)
    
    with open(factors_file, 'w', encoding='utf-8') as f:
        json.dump(factors_json, f, indent=2, ensure_ascii=False)
    
    # 测试表现
    logger.info("Testing performance on test dataset...")
    ic_test, ric_test = pool.test_ensemble(calculator_test)
    
    test_results = {
        "ic_test": float(ic_test),
        "ric_test": float(ric_test),
        "n_factors": len(generated_factors)
    }
    
    test_file = os.path.join(result_dir, "test_results.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Generation complete. Results saved to {result_dir}")
    logger.info(f"Test IC: {ic_test:.4f}, Test Rank IC: {ric_test:.4f}")


if __name__ == "__main__":
    main() 