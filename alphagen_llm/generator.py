"""
# LLM因子生成器模块 (LLM Factor Generator Module)
#
# 本文件实现了使用LLM生成和改进量化因子的工具。主要内容包括：
#
# 1. LLMAlphaGenerator：LLM因子生成器
#    - 与LLM进行交互式因子生成
#    - 解析LLM输出为结构化表达式
#    - 评估和改进生成的因子
#
# 2. 自动迭代改进流程：
#    - 分析因子表现
#    - 基于表现指标提供反馈
#    - 引导LLM进行定向改进
#
# 与其他组件的关系：
# - 使用alphagen_llm/client.py与LLM交互
# - 使用alphagen_llm/prompts中的提示模板
# - 生成的因子被保存到alphagen/models中的因子池
# - 与alphagen/data/expression.py配合表示因子
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import logging
from dataclasses import dataclass

from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen_llm.client import ChatClient
from alphagen_llm.prompts.system_prompt import ALPHA_GENERATOR_PROMPT, FACTOR_EVALUATOR_PROMPT
from alphagen_llm.prompts.interaction import InterativeSession

logger = logging.getLogger(__name__)

@dataclass
class GeneratedFactor:
    """生成的因子信息"""
    expression_text: str  # 原始文本表达式
    parsed_expression: Optional[Expression]  # 解析后的表达式对象
    description: str  # 因子描述
    metrics: Optional[Dict[str, float]] = None  # 评估指标
    valid: bool = False  # 是否有效


class LLMAlphaGenerator:
    """基于LLM的因子生成器"""
    
    def __init__(
        self,
        client: ChatClient,
        pool: AlphaPoolBase,
        parser: ExpressionParser,
        system_prompt: Optional[str] = None
    ):
        self.client = client
        self.pool = pool
        self.parser = parser
        self.system_prompt = system_prompt or ALPHA_GENERATOR_PROMPT
        self.session = InterativeSession(
            client=client,
            system_prompt=self.system_prompt
        )
    
    def generate_factors(
        self,
        prompt: str,
        n_attempts: int = 3
    ) -> List[GeneratedFactor]:
        """生成多个因子"""
        factors = []
        
        for _ in range(n_attempts):
            response = self.session.submit_and_get_response(prompt)
            new_factors = self._extract_factors_from_response(response)
            factors.extend(new_factors)
            
            # 如果已经有足够的有效因子，提前结束
            valid_factors = [f for f in factors if f.valid]
            if len(valid_factors) >= n_attempts:
                break
                
            # 如果需要继续尝试，提供更明确的指导
            prompt = """
前一轮生成的因子不足或无效。请生成更多不同的alpha因子，注意：
1. 确保表达式的语法正确
2. 每个因子应有清晰的数学表达式
3. 请使用支持的特征和操作符
4. 提供简洁的表达式，避免过度复杂
            """
        
        return factors
    
    def _extract_factors_from_response(self, response: str) -> List[GeneratedFactor]:
        """从LLM响应中提取因子表达式"""
        # 提取表达式模式: 尝试多种格式
        patterns = [
            r'`([^`]+)`',  # Markdown代码块
            r'"([^"]+)"',  # 双引号表达式
            r'表达式[:：]\s*([^\n]+)',  # 中文标记
            r'Factor[:：]\s*([^\n]+)',  # 英文标记
            r'Alpha[:：]\s*([^\n]+)',   # Alpha标记
            r'([a-zA-Z0-9_\(\)\+\-\*\/\.\,\s]+\([a-zA-Z0-9_\(\)\+\-\*\/\.\,\s]+\))'  # 函数式表达式
        ]
        
        potential_expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, response)
            potential_expressions.extend(matches)
        
        # 分析响应以获取描述
        factors = []
        for expr_text in set(potential_expressions):
            # 尝试解析表达式
            parsed_expr = None
            try:
                parsed_expr = self.parser.parse(expr_text.strip())
                valid = True
            except Exception as e:
                logger.warning(f"解析表达式失败: {expr_text} - {str(e)}")
                valid = False
            
            # 提取该表达式周围的描述
            expr_index = response.find(expr_text)
            if expr_index >= 0:
                # 提取上下文
                start = max(0, expr_index - 200)
                end = min(len(response), expr_index + len(expr_text) + 200)
                context = response[start:end]
                
                # 查找段落分隔
                paragraphs = re.split(r'\n\s*\n', context)
                for p in paragraphs:
                    if expr_text in p:
                        description = p.replace(expr_text, f"**{expr_text}**")
                        break
                else:
                    description = context
            else:
                description = "无描述"
            
            factors.append(GeneratedFactor(
                expression_text=expr_text,
                parsed_expression=parsed_expr,
                description=description,
                valid=valid
            ))
                
        return factors
    
    def evaluate_factors(self, factors: List[GeneratedFactor]) -> List[GeneratedFactor]:
        """评估因子性能"""
        for factor in factors:
            if not factor.valid or factor.parsed_expression is None:
                factor.metrics = {"error": -1.0}
                continue
                
            try:
                # 尝试添加到因子池并获取评估指标
                metrics = self.pool.try_new_expr(factor.parsed_expression)
                factor.metrics = {
                    "score": metrics
                }
            except Exception as e:
                logger.warning(f"评估因子失败: {factor.expression_text} - {str(e)}")
                factor.metrics = {"error": -1.0}
                
        return factors
    
    def refine_factor(
        self,
        factor: GeneratedFactor,
        feedback: str
    ) -> GeneratedFactor:
        """根据反馈改进因子"""
        prompt = f"""
请改进以下Alpha因子:

```
{factor.expression_text}
```

原因子描述:
{factor.description}

评估反馈:
{feedback}

请提供:
1. 改进后的因子表达式
2. 改进的理由和预期效果
        """
        
        response = self.session.submit_and_get_response(prompt)
        new_factors = self._extract_factors_from_response(response)
        
        if not new_factors:
            return factor
            
        # 选择第一个有效因子
        for new_factor in new_factors:
            if new_factor.valid:
                # 保留原始指标用于比较
                new_factor.metrics = {"original": factor.metrics}
                return new_factor
                
        return factor
    
    def iterative_improvement(
        self,
        initial_factors: List[GeneratedFactor],
        n_iterations: int = 3,
        metric_threshold: float = 0.1
    ) -> List[GeneratedFactor]:
        """迭代改进因子"""
        current_factors = self.evaluate_factors(initial_factors)
        improved_factors = []
        
        for factor in current_factors:
            if not factor.valid or factor.metrics is None:
                continue
                
            best_factor = factor
            best_score = factor.metrics.get("score", -float("inf"))
            
            for iteration in range(n_iterations):
                # 生成反馈
                if best_score < metric_threshold:
                    feedback = f"当前因子得分较低: {best_score:.4f}，需要显著改进。"
                else:
                    feedback = f"当前因子得分: {best_score:.4f}，尝试进一步优化。"
                    
                if factor.metrics.get("error", 0) != 0:
                    feedback += " 当前因子可能存在计算错误，请检查表达式的有效性。"
                
                # 改进因子
                new_factor = self.refine_factor(best_factor, feedback)
                
                # 评估新因子
                if new_factor.valid and new_factor.parsed_expression is not None:
                    try:
                        metrics = self.pool.try_new_expr(new_factor.parsed_expression)
                        new_factor.metrics = {"score": metrics}
                        
                        # 比较并更新最佳因子
                        new_score = new_factor.metrics.get("score", -float("inf"))
                        if new_score > best_score:
                            best_factor = new_factor
                            best_score = new_score
                            
                    except Exception as e:
                        logger.warning(f"评估改进因子失败: {new_factor.expression_text} - {str(e)}")
            
            improved_factors.append(best_factor)
            
        return improved_factors 