"""
# 表达式异常模块 (Expression Exception Module)
#
# 本文件定义了表达式处理过程中可能出现的异常类型。主要内容包括：
#
# 1. InvalidExpressionException：表达式无效异常
#    - 用于表示表达式语法或结构不正确的情况
#    - 继承自ValueError，提供更具体的异常类型
#
# 与其他组件的关系：
# - 被alphagen/data/tree.py使用，标识表达式构建过程中的错误
# - 被alphagen/data/parser.py间接使用，处理表达式解析错误
# - 为表达式处理提供统一的错误处理机制
"""
class InvalidExpressionException(ValueError):
    pass
