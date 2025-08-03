from app.handler.parser.base import BaseToolParser, BaseThinkingParser
from app.handler.parser.qwen3 import Qwen3ToolParser, Qwen3ThinkingParser
from app.handler.parser.glm4_moe import Glm4MoeToolParser, Glm4MoeThinkingParser
from typing import Tuple
__all__ = ['BaseToolParser', 'BaseThinkingParser', 'Qwen3ToolParser', 'Qwen3ThinkingParser', 'Glm4MoeToolParser', 'Glm4MoeThinkingParser']

parser_map = {
    'qwen3': {
        "tool_parser": Qwen3ToolParser,
        "thinking_parser": Qwen3ThinkingParser
    },
    'qwen3_moe': {
        "tool_parser": Qwen3ToolParser,
        "thinking_parser": Qwen3ThinkingParser
    },
    'glm4_moe': {
        "tool_parser": Glm4MoeToolParser,
        "thinking_parser": Glm4MoeThinkingParser
    }
}

def get_parser(model_name: str) -> Tuple[BaseToolParser, BaseThinkingParser]:
    if model_name not in parser_map:
        return None, None
        
    model_parsers = parser_map[model_name]
    tool_parser = model_parsers.get("tool_parser")
    thinking_parser = model_parsers.get("thinking_parser")
    
    return (tool_parser() if tool_parser else None, 
            thinking_parser() if thinking_parser else None)
