"""
# LLM客户端模块 (LLM Client Module)
#
# 本文件实现了与大型语言模型交互的客户端抽象和实现，主要内容包括：
#
# 1. ChatConfig：聊天配置类，定义模型参数
# 2. ChatClient：抽象基类，定义LLM交互接口
# 3. OpenAIClient：OpenAI API实现
#
# 与其他组件的关系：
# - 被alphagen_llm中的迭代生成流程使用
# - 提供与外部LLM服务的统一接口
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import os
import time
import logging
from abc import ABC, abstractmethod

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

logger = logging.getLogger(__name__)

@dataclass
class ChatConfig:
    """配置LLM交互参数"""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # API相关配置
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_type: Optional[str] = None
    api_version: Optional[str] = None
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为API参数字典"""
        result = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
            
        return result


class ChatClient(ABC):
    """LLM聊天客户端抽象基类"""
    
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.setup()
    
    @abstractmethod
    def setup(self):
        """设置客户端环境和配置"""
        pass
    
    @abstractmethod
    def chat(self, 
             messages: List[Dict[str, str]], 
             config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """与LLM进行聊天交互
        
        Args:
            messages: 消息列表，每个消息包含角色和内容
            config_override: 覆盖默认配置的参数
            
        Returns:
            LLM响应
        """
        pass
    
    def simple_completion(self, prompt: str) -> str:
        """简单的文本完成
        
        Args:
            prompt: 提示文本
            
        Returns:
            LLM生成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages)
        return response.get("content", "")


class OpenAIClient(ChatClient):
    """OpenAI API的客户端实现"""
    
    def setup(self):
        """设置OpenAI API环境"""
        # 设置API密钥，优先使用配置中的密钥，其次使用环境变量
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it in config or set OPENAI_API_KEY env var.")
        
        # 设置API基础URL（如果需要自定义端点）
        api_base = self.config.api_base or os.environ.get("OPENAI_API_BASE")
        
        # 创建客户端
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base
        )
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """通过OpenAI API进行聊天交互"""
        # 合并配置
        params = self.config.to_dict()
        if config_override:
            params.update(config_override)
        
        # 重试逻辑
        retries = 0
        while retries <= self.config.max_retries:
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **params
                )
                
                # 提取响应内容
                choice: Choice = response.choices[0]
                message: ChatCompletionMessage = choice.message
                
                return {
                    "content": message.content,
                    "role": message.role,
                    "finish_reason": choice.finish_reason,
                    "full_response": response
                }
                
            except Exception as e:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error(f"Failed after {self.config.max_retries} retries: {str(e)}")
                    raise
                
                logger.warning(f"Attempt {retries} failed, retrying in {self.config.retry_delay}s: {str(e)}")
                time.sleep(self.config.retry_delay * retries)  # 指数退避
        
        # 这应该永远不会执行，但为了类型安全
        return {"content": "", "role": "assistant", "finish_reason": "error"} 