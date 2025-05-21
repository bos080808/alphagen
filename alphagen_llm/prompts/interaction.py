"""
# 交互式会话模块 (Interactive Session Module)
#
# 本文件实现了与LLM进行交互式因子生成的会话管理。主要内容包括：
#
# 1. InterativeSession：交互式会话类
#    - 管理与LLM的多轮对话
#    - 记录会话历史
#    - 提供会话状态保存和恢复
#
# 2. DefaultInteraction：默认交互实现
#    - 预设的因子生成交互流程
#    - 包含提示模板和响应处理
#
# 与其他组件的关系：
# - 使用client.py中的ChatClient与LLM交互
# - 使用system_prompt.py中的提示模板
# - 生成的因子可以被导入到alphagen的因子池中
"""
from typing import List, Dict, Any, Optional, Union, Callable
import json
import os
from datetime import datetime

from alphagen_llm.client import ChatClient


class InterativeSession:
    """交互式LLM会话管理类"""
    
    def __init__(
        self,
        client: ChatClient,
        system_prompt: str,
        session_name: Optional[str] = None,
        save_dir: str = "sessions"
    ):
        self.client = client
        self.messages: List[Dict[str, str]] = []
        
        # 添加系统提示
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 会话管理
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.messages.append({
            "role": "assistant",
            "content": content
        })
    
    def get_response(self, config_override: Optional[Dict[str, Any]] = None) -> str:
        """获取LLM响应"""
        response = self.client.chat(self.messages, config_override)
        content = response.get("content", "")
        
        # 添加助手回复到历史记录
        self.add_assistant_message(content)
        
        return content
    
    def submit_and_get_response(self, user_message: str, config_override: Optional[Dict[str, Any]] = None) -> str:
        """提交用户消息并获取响应"""
        self.add_user_message(user_message)
        return self.get_response(config_override)
    
    def save_session(self) -> str:
        """保存会话历史到文件"""
        filename = os.path.join(self.save_dir, f"{self.session_name}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "session_name": self.session_name,
                "timestamp": datetime.now().isoformat(),
                "messages": self.messages
            }, f, ensure_ascii=False, indent=2)
        return filename
    
    @classmethod
    def load_session(
        cls,
        client: ChatClient,
        filename: str
    ) -> "InterativeSession":
        """从文件加载会话"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session = cls(
            client=client,
            system_prompt="",  # 不添加系统提示，因为会从历史记录加载
            session_name=data.get("session_name")
        )
        
        session.messages = data.get("messages", [])
        return session
    
    def get_last_user_message(self) -> Optional[str]:
        """获取最后一条用户消息"""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """获取最后一条助手消息"""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return None


class DefaultInteraction:
    """默认因子生成交互实现"""
    
    def __init__(
        self,
        client: ChatClient,
        system_prompt: str,
        session_name: Optional[str] = None,
        save_dir: str = "sessions",
        auto_save: bool = True
    ):
        self.session = InterativeSession(
            client=client,
            system_prompt=system_prompt,
            session_name=session_name,
            save_dir=save_dir
        )
        self.auto_save = auto_save
    
    def start_with_context(self, context: str) -> str:
        """开始会话并提供市场背景信息"""
        prompt = f"""
请设计适合当前市场环境的alpha因子。

市场背景信息:
{context}

请基于这一背景，设计2-3个不同的alpha因子，并解释它们的原理和预期效果。
        """
        response = self.session.submit_and_get_response(prompt)
        
        if self.auto_save:
            self.session.save_session()
            
        return response
    
    def request_factor_refinement(self, feedback: str) -> str:
        """请求根据反馈改进因子"""
        prompt = f"""
基于以下反馈，请改进你提出的因子：

{feedback}

请提供改进后的因子表达式，并解释你的修改如何解决了反馈中提到的问题。
        """
        response = self.session.submit_and_get_response(prompt)
        
        if self.auto_save:
            self.session.save_session()
            
        return response
    
    def provide_backtest_results(self, results: str) -> str:
        """提供回测结果并请求分析"""
        prompt = f"""
以下是你提出的因子的回测结果：

{results}

请分析这些结果，并提出:
1. 结果的关键亮点和不足
2. 因子表现的可能原因
3. 进一步改进因子的方向
        """
        response = self.session.submit_and_get_response(prompt)
        
        if self.auto_save:
            self.session.save_session()
            
        return response
    
    def custom_interaction(self, prompt: str) -> str:
        """自定义交互"""
        response = self.session.submit_and_get_response(prompt)
        
        if self.auto_save:
            self.session.save_session()
            
        return response
    
    def end_session(self) -> str:
        """结束会话，总结和保存"""
        prompt = """
请总结我们的讨论，列出最终推荐的alpha因子表达式，并概述它们的主要特点和预期应用场景。
        """
        response = self.session.submit_and_get_response(prompt)
        
        filename = self.session.save_session()
        return f"会话已保存到 {filename}\n\n{response}" 