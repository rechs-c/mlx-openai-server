"""
Base Message Converter

Provides generic conversion from OpenAI API message format to model-compatible format.
"""

import json
from typing import Dict, List, Any


class BaseMessageConverter:
    """Base message format converter class"""
    
    def convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert message format to be compatible with specific model chat templates"""
        converted_messages = []
        
        for message in messages:
            converted_message = self._convert_single_message(message)
            if converted_message:
                converted_messages.append(converted_message)
        
        return converted_messages
    
    def _convert_single_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single message"""
        if not isinstance(message, dict):
            return message

        # Convert function.arguments from string to object in tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            self._convert_tool_calls(tool_calls)
        
        return message
    
    def _convert_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Convert arguments format in tool calls"""
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and "function" in tool_call:
                function = tool_call["function"]
                if isinstance(function, dict) and "arguments" in function:
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        function["arguments"] = self._parse_arguments_string(arguments)
    
    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse arguments string to object, can be overridden by subclasses"""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str