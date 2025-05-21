"""
# 日志工具模块 (Logging Utilities Module)
#
# 本文件提供了日志记录和配置的功能。主要内容包括：
#
# 1. get_logger：创建和配置日志记录器
#    - 支持同时输出到文件和控制台
#    - 自动创建日志目录
#    - 配置格式化输出
#
# 2. get_null_logger：创建空日志记录器
#    - 用于禁用日志输出
#    - 避免日志传播
#
# 与其他组件的关系：
# - 被scripts中的训练脚本使用，记录训练进程
# - 为alphagen/models中的模型提供日志支持
# - 记录系统运行状态和错误信息
"""
import logging
import os
from typing import Optional


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    logger = logging.getLogger(name)
    while logger.hasHandlers():
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")

    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_null_logger() -> logging.Logger:
    logger = logging.getLogger("null_logger")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger
