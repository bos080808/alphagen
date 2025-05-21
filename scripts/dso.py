"""
# 深度符号优化因子生成模块 (Deep Symbolic Optimization Factor Module)
#
# 本文件实现了基于DSO（深度符号优化）方法的量化因子生成。主要内容包括：
#
# 1. 配置DSO模型和函数集
# 2. 定义特征和常量映射
# 3. 设置评估函数和因子池
# 4. 训练DSO模型生成黄金市场因子
#
# 与其他组件的关系：
# - 使用alphagen/data/expression.py中的表达式表示生成的因子
# - 使用alphagen/models中的因子池存储和评估生成的因子
# - 作为gold_factor_generation.py中DSO方法的核心实现
"""
import numpy as np

from alphagen.data.expression import *
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *



funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(['open', 'close', 'high', 'low', 'volume', 'oi']):
    funcs[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)


instruments = 'csi300'
import sys
seed = int(sys.argv[1])
reseed_everything(seed)

cache = {}
device = torch.device('cuda:0')
data_train = GoldData('2009-01-01', '2018-12-31', device=device)
data_valid = GoldData('2019-01-01', '2019-12-31', device=device)
data_test = GoldData('2020-01-01', '2021-12-31', device=device)
calculator_train = QLibGoldDataCalculator(data_train, target)
calculator_valid = QLibGoldDataCalculator(data_valid, target)
calculator_test = QLibGoldDataCalculator(data_test, target)


if __name__ == '__main__':
    X = np.array([['open_', 'close', 'high', 'low', 'volume', 'oi']])
    y = np.array([[1]])
    functions.function_map = funcs

    pool = MseAlphaPool(
        capacity=10,
        calculator=calculator_train,
        ic_lower_bound=None
    )

    class Ev:
        def __init__(self, pool):
            self.cnt = 0
            self.pool: MseAlphaPool = pool
            self.results = {}

        def alpha_ev_fn(self, key):
            expr = eval(key)
            try:
                ret = self.pool.try_new_expr(expr)
            except OutOfDataRangeError:
                ret = -1.
            finally:
                self.cnt += 1
                if self.cnt % 100 == 0:
                    test_ic = pool.test_ensemble(calculator_test)[0]
                    self.results[self.cnt] = test_ic
                    print(self.cnt, test_ic)
                return ret

    ev = Ev(pool)

    config = dict(
        task=dict(
            task_type='regression',
            function_set=list(funcs.keys()),
            metric='alphagen',
            metric_params=[lambda key: ev.alpha_ev_fn(key)],
        ),
        training={'n_samples': 20000, 'batch_size': 128, 'epsilon': 0.05},
        prior={'length': {'min_': 2, 'max_': 20, 'on': True}}
    )

    # Create the model
    model = DeepSymbolicRegressor(config=config)
    model.fit(X, y)

    print(ev.results) 