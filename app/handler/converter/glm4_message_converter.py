"""
GLM4消息格式转换器

GLM4模型的聊天模板对工具调用格式有特殊要求，
此模块负责将OpenAI API格式的消息转换为GLM4兼容格式。
"""

import json
from typing import Dict, List, Any


class GLM4MessageConverter:
    """GLM4专用的消息格式转换器"""
    
    def convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换消息格式以兼容GLM4聊天模板"""
        converted_messages = []
        
        for message in messages:
            converted_message = self._convert_single_message(message)
            if converted_message:
                converted_messages.append(converted_message)
        
        return converted_messages
    
    def _convert_single_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """转换单条消息"""
        if not isinstance(message, dict):
            return message
        
        # 复制消息
        converted_message = message.copy()
        
        # 转换tool_calls中的function.arguments从字符串到对象
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "function" in tool_call:
                    function = tool_call["function"]
                    if isinstance(function, dict) and "arguments" in function:
                        arguments = function["arguments"]
                        if isinstance(arguments, str):
                            try:
                                function["arguments"] = json.loads(arguments)
                            except json.JSONDecodeError:
                                # 如果解析失败，保持原样
                                pass
        
        return converted_message