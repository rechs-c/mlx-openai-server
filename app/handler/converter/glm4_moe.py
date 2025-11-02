"""
GLM4 MoE Message Converter

GLM4 MoE model's chat template has special requirements for tool call format.
This module is responsible for converting OpenAI API format messages to GLM4 MoE-compatible format.
"""

import json
from typing import Dict, List, Any
from app.handler.converter.base import BaseMessageConverter


class Glm4MoEMessageConverter(BaseMessageConverter):
    """GLM4 MoE-specific message format converter"""
    
    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse GLM4 MoE-specific argument string format"""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str