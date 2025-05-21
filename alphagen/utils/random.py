"""
# 随机数控制模块 (Random Control Module)
#
# 本文件提供了随机数生成的控制和设置功能。主要内容包括：
#
# 1. reseed_everything：重新设置所有随机数种子
#    - 设置Python内置random、numpy和PyTorch的随机种子
#    - 确保实验的可重复性
#    - 控制CUDA相关的随机行为
#
# 与其他组件的关系：
# - 被alphagen/rl/env/core.py使用，确保环境重置的一致性
# - 在训练脚本中用于设置实验种子
# - 支持结果的可复现性和调试
"""
from typing import Optional
import random
import os
import numpy as np
import torch
from torch.backends import cudnn


def reseed_everything(seed: Optional[int]):
    if seed is None:
        return

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
