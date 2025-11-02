"""
MiniMax Message Converter

MiniMax model's chat template has special requirements for tool call format.
This module is responsible for converting OpenAI API format messages to MiniMax-compatible format.
"""

import json
from typing import Dict, List, Any
from app.handler.converter.base import BaseMessageConverter


class MiniMaxMessageConverter(BaseMessageConverter):
    """MiniMax-specific message format converter"""
    
    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse MiniMax-specific argument string format"""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str