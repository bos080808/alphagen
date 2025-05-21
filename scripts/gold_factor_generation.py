"""
# 黄金因子生成脚本 (Gold Factor Generation Script)
#
# 本脚本演示如何使用AlphaGen项目生成黄金市场数据的因子，
# 采用三种不同的方法：
#
# 1. 基于PPO的因子生成
# 2. 遗传规划 (GPLearn)
# 3. 深度符号优化 (DSO)
#
# 脚本从CSV文件加载黄金市场数据，使用各种方法训练模型，
# 并评估生成的因子。
"""

import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# AlphaGen imports
from alphagen.data.expression import *
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen_qlib.gold_data import GoldData, initialize_qlib, FeatureType
from alphagen.models.linear_alpha_pool import MseAlphaPool

# For GPLearn approach
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

# For DSO approach
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions

# For PPO approach
import gymnasium as gym
from stable_baselines3 import PPO
from alphagen.rl.env.factor_env import FactorEnv
from alphagen.rl.policy import PPOPolicy, MultiMetricReward


def parse_args():
    parser = argparse.ArgumentParser(description='Gold Factor Generation')
    parser.add_argument('--data_path', type=str, default='data/au888.csv',
                        help='Path to gold market data CSV file')
    parser.add_argument('--start_date', type=str, default='2012-01-01',
                        help='Start date for training data')
    parser.add_argument('--end_date', type=str, default='2022-12-31',
                        help='End date for training data')
    parser.add_argument('--test_start_date', type=str, default='2023-01-01', 
                        help='Start date for test data')
    parser.add_argument('--test_end_date', type=str, default='2023-12-31',
                        help='End date for test data')
    parser.add_argument('--method', type=str, choices=['ppo', 'gp', 'dso', 'all'], 
                        default='all', help='Factor generation method to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='out/gold_factors',
                        help='Directory to save results')
    return parser.parse_args()


def setup_env(args):
    """Set up the environment, data, and models"""
    # Set random seed for reproducibility
    reseed_everything(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load gold data
    features = [
        FeatureType.OPEN, 
        FeatureType.CLOSE, 
        FeatureType.HIGH, 
        FeatureType.LOW, 
        FeatureType.VOLUME, 
        FeatureType.OI
    ]
    
    data_train = GoldData(
        start_time=args.start_date,
        end_time=args.end_date,
        data_path=args.data_path,
        device=device,
        features=features
    )
    
    data_test = GoldData(
        start_time=args.test_start_date,
        end_time=args.test_end_date,
        data_path=args.data_path,
        device=device,
        features=features
    )
    
    # Define target expression (next day's return)
    target = Shift(Div(Sub(Close(), Shift(Close(), 1)), Shift(Close(), 1)), -1)
    
    # Create calculators
    calculator_train = QLibGoldDataCalculator(data_train, target)
    calculator_test = QLibGoldDataCalculator(data_test, target)
    
    return data_train, data_test, calculator_train, calculator_test, device


def run_gplearn(data_train, data_test, calculator_train, calculator_test, args, device):
    """Run factor generation using GPLearn (Genetic Programming)"""
    print("\n=== Running GPLearn Factor Generation ===")
    
    # Set up functions for gplearn
    funcs = [make_function(**func._asdict()) for func in generic_funcs]
    
    # Create cache for evaluated expressions
    cache = {}
    
    # Define fitness function
    def _metric(x, y, w):
        key = y[0]
        
        if key in cache:
            return cache[key]
        token_len = key.count('(') + key.count(')')
        if token_len > 20:
            return -1.
            
        expr = eval(key)
        try:
            ic = calculator_train.calc_single_IC_ret(expr)
        except Exception:
            ic = -1.
        if np.isnan(ic):
            ic = -1.
        cache[key] = ic
        return ic
    
    Metric = make_fitness(function=_metric, greater_is_better=True)
    
    # Set up feature terminals for GPLearn
    features = ['open_', 'close', 'high', 'low', 'volume', 'oi']
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    terminals = features + constants
    
    X_train = np.array([terminals])
    y_train = np.array([[1]])
    
    # Create and train the model
    est_gp = SymbolicRegressor(
        population_size=1000,
        generations=30,
        init_depth=(2, 6),
        tournament_size=200,
        stopping_criteria=0.95,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.01,
        random_state=args.seed,
        function_set=funcs,
        metric=Metric,
        const_range=None,
        n_jobs=1
    )
    
    print("Training GP model...")
    est_gp.fit(X_train, y_train)
    
    # Get best programs
    print("\nTop factors generated:")
    top_exprs = []
    
    for key, value in sorted(cache.items(), key=lambda x: x[1], reverse=True)[:10]:
        expr = eval(key)
        ic_test, rank_ic_test = calculator_test.calc_single_all_ret(expr)
        print(f"Expression: {key}")
        print(f"Train IC: {cache[key]:.4f}, Test IC: {ic_test:.4f}, Test Rank IC: {rank_ic_test:.4f}")
        top_exprs.append((expr, key, ic_test))
    
    # Create alpha pool from top expressions
    pool = MseAlphaPool(
        capacity=10,
        calculator=calculator_train,
        ic_lower_bound=None
    )
    
    for expr, _, _ in top_exprs[:10]:
        pool.try_new_expr(expr)
    
    # Test the ensemble
    ic_train, ric_train = pool.test_ensemble(calculator_train)
    ic_test, ric_test = pool.test_ensemble(calculator_test)
    
    print("\nEnsemble performance:")
    print(f"Train IC: {ic_train:.4f}, Train Rank IC: {ric_train:.4f}")
    print(f"Test IC: {ic_test:.4f}, Test Rank IC: {ric_test:.4f}")
    
    # Save results
    results = {
        "method": "gplearn",
        "top_expressions": [(str(expr), key, float(ic)) for expr, key, ic in top_exprs[:10]],
        "ensemble_train_ic": float(ic_train),
        "ensemble_train_ric": float(ric_train),
        "ensemble_test_ic": float(ic_test),
        "ensemble_test_ric": float(ric_test)
    }
    
    return results


def run_dso(data_train, data_test, calculator_train, calculator_test, args, device):
    """Run factor generation using DSO (Deep Symbolic Optimization)"""
    print("\n=== Running DSO Factor Generation ===")
    
    # Set up functions for DSO
    funcs_dict = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
    for i, feature in enumerate(['open_', 'close', 'high', 'low', 'volume', 'oi']):
        funcs_dict[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
    for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]:
        funcs_dict[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)
    
    functions.function_map = funcs_dict
    
    # Create pool for evaluating expressions
    pool = MseAlphaPool(
        capacity=10,
        calculator=calculator_train,
        ic_lower_bound=None
    )
    
    results = {}
    
    class Evaluator:
        def __init__(self, pool):
            self.cnt = 0
            self.pool = pool
            self.results = {}
            self.expressions = []
            
        def alpha_ev_fn(self, key):
            expr = eval(key)
            try:
                ret = self.pool.try_new_expr(expr)
                self.expressions.append((expr, key, ret))
            except Exception:
                ret = -1.
            finally:
                self.cnt += 1
                if self.cnt % 100 == 0:
                    test_ic = pool.test_ensemble(calculator_test)[0]
                    self.results[self.cnt] = test_ic
                    print(f"Evaluated {self.cnt} expressions, current test IC: {test_ic:.4f}")
                return ret
    
    evaluator = Evaluator(pool)
    
    # DSO configuration
    config = dict(
        task=dict(
            task_type='regression',
            function_set=list(funcs_dict.keys()),
            metric='alphagen',
            metric_params=[lambda key: evaluator.alpha_ev_fn(key)],
        ),
        training={'n_samples': 10000, 'batch_size': 128, 'epsilon': 0.05},
        prior={'length': {'min_': 2, 'max_': 20, 'on': True}}
    )
    
    # Create input data for DSO
    X = np.array([['open_', 'close', 'high', 'low', 'volume', 'oi']])
    y = np.array([[1]])
    
    # Create and train the model
    print("Training DSO model...")
    model = DeepSymbolicRegressor(config=config)
    model.fit(X, y)
    
    # Print and save top expressions
    print("\nTop factors generated:")
    top_exprs = sorted(evaluator.expressions, key=lambda x: x[2], reverse=True)[:10]
    
    for expr, key, train_ic in top_exprs:
        ic_test, rank_ic_test = calculator_test.calc_single_all_ret(expr)
        print(f"Expression: {key}")
        print(f"Train IC: {train_ic:.4f}, Test IC: {ic_test:.4f}, Test Rank IC: {rank_ic_test:.4f}")
    
    # Test ensemble performance
    ic_train, ric_train = pool.test_ensemble(calculator_train)
    ic_test, ric_test = pool.test_ensemble(calculator_test)
    
    print("\nEnsemble performance:")
    print(f"Train IC: {ic_train:.4f}, Train Rank IC: {ric_train:.4f}")
    print(f"Test IC: {ic_test:.4f}, Test Rank IC: {ric_test:.4f}")
    
    # Save results
    results = {
        "method": "dso",
        "top_expressions": [(str(expr), key, float(train_ic)) for expr, key, train_ic in top_exprs],
        "ensemble_train_ic": float(ic_train),
        "ensemble_train_ric": float(ric_train),
        "ensemble_test_ic": float(ic_test),
        "ensemble_test_ric": float(ric_test)
    }
    
    return results


def run_ppo(data_train, data_test, calculator_train, calculator_test, args, device):
    """Run factor generation using PPO (Proximal Policy Optimization)"""
    print("\n=== Running PPO Factor Generation ===")
    
    # Create factor environment
    env = FactorEnv(
        calculator=calculator_train,
        max_steps=20,  # Maximum factor length
        reward_function=MultiMetricReward(
            weights={'ic': 0.4, 'rank_ic': 0.4, 'sharpe': 0.2}
        )
    )
    
    # Create PPO policy
    policy = PPOPolicy(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train the policy
    print("Training PPO model...")
    policy.train(env, total_timesteps=100000)
    
    # Generate factors using the trained policy
    print("\nGenerating factors with trained policy...")
    pool = MseAlphaPool(
        capacity=10,
        calculator=calculator_train,
        ic_lower_bound=None
    )
    
    # Generate multiple factors
    factors = []
    for i in range(50):
        obs, _ = env.reset()
        done = False
        actions = []
        
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action)
            
        if 'expr' in info:
            expr = info['expr']
            try:
                ic_train = calculator_train.calc_single_IC_ret(expr)
                ic_test, rank_ic_test = calculator_test.calc_single_all_ret(expr)
                factors.append((expr, str(expr), ic_train, ic_test, rank_ic_test))
                pool.try_new_expr(expr)
            except Exception as e:
                print(f"Error evaluating expression: {e}")
    
    # Sort factors by test IC
    factors.sort(key=lambda x: x[3], reverse=True)
    
    # Print top factors
    print("\nTop factors generated:")
    for expr, expr_str, ic_train, ic_test, rank_ic_test in factors[:10]:
        print(f"Expression: {expr_str}")
        print(f"Train IC: {ic_train:.4f}, Test IC: {ic_test:.4f}, Test Rank IC: {rank_ic_test:.4f}")
    
    # Test ensemble performance
    ic_train, ric_train = pool.test_ensemble(calculator_train)
    ic_test, ric_test = pool.test_ensemble(calculator_test)
    
    print("\nEnsemble performance:")
    print(f"Train IC: {ic_train:.4f}, Train Rank IC: {ric_train:.4f}")
    print(f"Test IC: {ic_test:.4f}, Test Rank IC: {ric_test:.4f}")
    
    # Save results
    results = {
        "method": "ppo",
        "top_expressions": [(str(expr), expr_str, float(ic_train), float(ic_test), float(rank_ic_test)) 
                           for expr, expr_str, ic_train, ic_test, rank_ic_test in factors[:10]],
        "ensemble_train_ic": float(ic_train),
        "ensemble_train_ric": float(ric_train),
        "ensemble_test_ic": float(ic_test),
        "ensemble_test_ric": float(ric_test)
    }
    
    return results


def main():
    args = parse_args()
    data_train, data_test, calculator_train, calculator_test, device = setup_env(args)
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run selected method(s)
    if args.method == 'ppo' or args.method == 'all':
        ppo_results = run_ppo(data_train, data_test, calculator_train, calculator_test, args, device)
        all_results['ppo'] = ppo_results
        
        # Save PPO results
        with open(os.path.join(args.output_dir, f"ppo_results_{timestamp}.json"), 'w') as f:
            json.dump(ppo_results, f, indent=2)
    
    if args.method == 'gp' or args.method == 'all':
        gp_results = run_gplearn(data_train, data_test, calculator_train, calculator_test, args, device)
        all_results['gp'] = gp_results
        
        # Save GP results
        with open(os.path.join(args.output_dir, f"gp_results_{timestamp}.json"), 'w') as f:
            json.dump(gp_results, f, indent=2)
    
    if args.method == 'dso' or args.method == 'all':
        dso_results = run_dso(data_train, data_test, calculator_train, calculator_test, args, device)
        all_results['dso'] = dso_results
        
        # Save DSO results
        with open(os.path.join(args.output_dir, f"dso_results_{timestamp}.json"), 'w') as f:
            json.dump(dso_results, f, indent=2)
    
    # Save combined results
    if args.method == 'all':
        with open(os.path.join(args.output_dir, f"all_results_{timestamp}.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main() 