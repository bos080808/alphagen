"""
# 强化学习训练脚本 (Reinforcement Learning Training Script)
#
# 本文件是使用PPO算法训练因子生成模型的主执行脚本。主要功能包括：
#
# 1. 配置和初始化训练环境
# 2. 构建因子池和强化学习模型
# 3. 训练过程管理和模型保存
# 4. 因子评估和测试
#
# 与其他组件的关系：
# - 使用alphagen/rl/policy.py中的PPO策略
# - 使用alphagen/rl/env中的强化学习环境
# - 使用alphagen/models中的因子池存储生成的因子
# - 使用alphagen/data中的表达式和计算器
# - 训练结果保存在./out/results目录
"""
import json
import os
from typing import Optional, Tuple, List, Union
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.qlib_alpha_calculator import QLibGoldDataCalculator
from alphagen_qlib.gold_data import GoldData, initialize_qlib, FeatureType
from alphagen_llm.client import ChatClient, OpenAIClient, ChatConfig
from alphagen_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from alphagen_llm.prompts.interaction import InterativeSession, DefaultInteraction


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibGoldDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.chat_session is not None:
            self._try_use_llm()

        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_ic_ret:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    @property
    def pool(self) -> LinearAlphaPool:
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def run_single_experiment(
    seed: int = 0,
    data_path: str = "data/au888.csv",  # 黄金数据路径
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3
):
    reseed_everything(seed)
    
    # 黄金数据不需要初始化qlib
    # initialize_qlib("~/.qlib/qlib_data/cn_data")

    llm_replace_n = 0 if not use_llm else llm_replace_n
    print(f"""[Main] Starting training process
    Seed: {seed}
    Data Path: {data_path}
    Pool capacity: {pool_capacity}
    Total Iteration Steps: {steps}
    AlphaGPT-Like Init-Only LLM Usage: {alphagpt_init}
    Use LLM: {use_llm}
    Invoke LLM every N steps: {llm_every_n_steps}
    Replace N alphas with LLM: {llm_replace_n}
    Drop N alphas before LLM: {drop_rl_n}""")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = f"gold_{pool_capacity}_{seed}_{timestamp}_{tag}"
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 定义黄金数据的特征
    features = [
        FeatureType.OPEN, 
        FeatureType.CLOSE, 
        FeatureType.HIGH, 
        FeatureType.LOW, 
        FeatureType.VOLUME
    ]
    
    # 定义目标表达式 - 使用20天后的价格变化作为目标
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> GoldData:
        return GoldData(
            start_time=start,
            end_time=end,
            data_path=data_path,
            features=features,
            device=device
        )

    segments = [
        ("2012-01-01", "2021-12-31"),  # 训练集
        ("2022-01-01", "2022-06-30"),  # 测试集1
        ("2022-07-01", "2022-12-31"),  # 测试集2
        ("2023-01-01", "2023-06-30")   # 测试集3
    ]
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibGoldDataCalculator(d, target) for d in datasets]

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([])
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            get_logger('interaction', os.path.join(save_path, "interact.log"))
        )
        inter.update_pool(pool)

    cb = CustomCallback(
        save_path, calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )

    env = AlphaEnv(
        pool=pool,
        max_steps=20,
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
            n_layers=2, d_model=128,
            dropout=0.2, device=device
        ),
        device=device
    )
    model = MaskablePPO('MlpPolicy', env, verbose=1, device=device)
    model.learn(total_timesteps=steps, callback=cb)

    path = os.path.join(save_path, 'final_model')
    model.save(path)
    cb.show_pool_state()
    return cb


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    data_path: str = "data/au888.csv",  # 黄金数据路径
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = None,
    llm_every_n_steps: int = 25000
):
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds,)
    for seed in random_seeds:
        run_single_experiment(
            seed=seed,
            data_path=data_path,  # 传递黄金数据路径
            pool_capacity=pool_capacity,
            steps=steps or 200_000,
            alphagpt_init=alphagpt_init,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps,
            drop_rl_n=drop_rl_n,
        )


if __name__ == '__main__':
    fire.Fire(main)
