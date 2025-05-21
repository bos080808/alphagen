"""
# 遗传规划因子生成模块 (Genetic Programming Factor Generation)
#
# 本文件实现了使用遗传规划（GP）方法生成量化因子。主要内容包括：
#
# 1. 配置和初始化gplearn遗传规划器
#    - 定义适应度函数和操作符
#    - 设置训练参数
#    - 评估生成的因子
#
# 2. 因子池操作：
#    - 尝试单个因子评估
#    - 构建多因子组合
#    - 处理因子间相关性
#
# 与其他组件的关系：
# - 使用alphagen/models中的因子池存储生成的因子
# - 使用alphagen/data中的表达式表示因子
# - 使用gplearn库实现遗传规划算法
# - 作为PPO强化学习算法的替代方案，提供对比实验
"""
import json
import os
from collections import Counter
from typing import Optional

import numpy as np

from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen_qlib.gold_data import initialize_qlib
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


funcs = [make_function(**func._asdict()) for func in generic_funcs]

# Gold market specific setup
seed = 2
reseed_everything(seed)

cache = {}
device = torch.device("cuda:0")
initialize_qlib("~/.qlib/qlib_data/gold_data_2024")
data_train = GoldData("2012-01-01", "2021-12-31", device=device)
data_test = GoldData("2022-01-01", "2023-06-30", device=device)
calculator_train = QLibGoldDataCalculator(data_train, target)
calculator_test = QLibGoldDataCalculator(data_test, target)

pool = MseAlphaPool(
    capacity=20,
    calculator=calculator_train,
    ic_lower_bound=None,
    l1_alpha=5e-3,
    device=device
)


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
    except OutOfDataRangeError:
        ic = -1.
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic


Metric = make_fitness(function=_metric, greater_is_better=True)


def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    expr = eval(top_key)
    ic_test, ric_test = calculator_test.calc_single_all_ret(expr)
    return {
        'ic_test': ic_test,
        'ric_test': ric_test
    }


def try_pool(capacity: int, mutual_ic_thres: Optional[float] = None):
    pool = MseAlphaPool(
        capacity=capacity,
        calculator=calculator_train,
        ic_lower_bound=None
    )
    exprs = []

    def acceptable(expr: str) -> bool:
        if mutual_ic_thres is None:
            return True
        return all(abs(pool.calculator.calc_mutual_IC(e, eval(expr))) <= mutual_ic_thres
                   for e in exprs)

    most_common = dict(Counter(cache).most_common(capacity if mutual_ic_thres is None else None))
    for key in most_common:
        if acceptable(key):
            exprs.append(eval(key))
            if len(exprs) >= capacity:
                break
    pool.force_load_exprs(exprs)

    ic_train, ric_train = pool.test_ensemble(calculator_train)
    ic_test, ric_test = pool.test_ensemble(calculator_test)
    return {
        "ic_train": ic_train,
        "ric_train": ric_train,
        "ic_test": ic_test,
        "ric_test": ric_test,
        "pool_state": pool.to_json_dict()
    }


generation = 0

def ev():
    global generation
    generation += 1
    directory = f"out/gp/{seed}"
    os.makedirs(directory, exist_ok=True)
    if generation % 4 != 0:
        return
    capacity = 20
    res = {"pool": capacity, "res": try_pool(capacity, mutual_ic_thres=0.7)}
    with open(f'{directory}/{generation}.json', 'w') as f:
        json.dump({'res': res, 'cache': cache}, f, indent=4)


if __name__ == '__main__':
    # Updated for gold market features - replaced vwap with oi (open interest)
    features = ['open_', 'close', 'high', 'low', 'volume', 'oi']
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    terminals = features + constants

    X_train = np.array([terminals])
    y_train = np.array([[1]])

    est_gp = SymbolicRegressor(
        population_size=1000,
        generations=40,
        init_depth=(2, 6),
        tournament_size=600,
        stopping_criteria=1.,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.,
        random_state=seed,
        function_set=funcs,
        metric=Metric,  # type: ignore
        const_range=None,
        n_jobs=1
    )
    est_gp.fit(X_train, y_train, callback=ev)
    print(est_gp._program.execute(X_train)) 