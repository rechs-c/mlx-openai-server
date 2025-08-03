import json
from typing import Any, Dict, List, Tuple
from app.handler.parser.base import BaseToolParser, BaseThinkingParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
ARG_KEY_OPEN = "<arg_key>"
ARG_KEY_CLOSE = "</arg_key>"
ARG_VALUE_OPEN = "<arg_value>"
ARG_VALUE_CLOSE = "</arg_value>"

THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"

class ParseState:
    NORMAL = 0
    IN_TOOL_CALL = 1
    PARSING_FUNC_NAME = 2
    PARSING_ARG_KEY = 3
    PARSING_ARG_VALUE = 4

class Glm4MoeToolParser(BaseToolParser):
    """Parser for GLM4-MoE model's tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE   
        )
        self.buffer = ""

    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        res = []
        remaining_content = content
        while True:
            start_tool = remaining_content.find(self.tool_open)
            if start_tool == -1:
                break
            
            end_tool = remaining_content.find(self.tool_close, start_tool)
            if end_tool == -1:
                # Incomplete tool call, maybe handle as error or wait for more content
                break

            # Extract the full tool call block
            tool_content_full = remaining_content[start_tool:end_tool + len(self.tool_close)]
            
            # Extract content within the tool_call tags
            inner_content = tool_content_full[len(self.tool_open):-len(self.tool_close)].strip()
            
            # Find the first occurrence of a tag to correctly split function name
            first_tag_pos = inner_content.find(ARG_KEY_OPEN)
            if first_tag_pos != -1:
                func_name = inner_content[:first_tag_pos].strip()
                arg_content = inner_content[first_tag_pos:]
            else:
                func_name = inner_content.strip()
                arg_content = ""

            arguments = {}
            
            key_start = 0
            while True:
                start_key_tag = arg_content.find(ARG_KEY_OPEN, key_start)
                if start_key_tag == -1:
                    break
                end_key_tag = arg_content.find(ARG_KEY_CLOSE, start_key_tag)
                key = arg_content[start_key_tag + len(ARG_KEY_OPEN):end_key_tag].strip()

                start_value_tag = arg_content.find(ARG_VALUE_OPEN, end_key_tag)
                end_value_tag = arg_content.find(ARG_VALUE_CLOSE, start_value_tag)
                value = arg_content[start_value_tag + len(ARG_VALUE_OPEN):end_value_tag].strip()
                
                arguments[key] = value
                key_start = end_value_tag + len(ARG_VALUE_CLOSE)

            res.append({
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
            })
            
            # Move past the processed tool call block
            remaining_content = remaining_content[end_tool + len(self.tool_close):]

        return res, remaining_content.strip()

    def parse_stream(self, chunk: str) -> List[any]:
        self.buffer += chunk
        outputs = []
        
        while True:
            start_idx = self.buffer.find(self.tool_open)
            end_idx = self.buffer.find(self.tool_close)

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                # A full tool call is present
                
                # 1. Yield any text before the tool call
                if start_idx > 0:
                    outputs.append(self.buffer[:start_idx])
                
                # 2. Parse and yield the tool call
                end_of_tool_close = end_idx + len(self.tool_close)
                tool_call_block = self.buffer[start_idx:end_of_tool_close]
                
                parsed_tools, _ = self.parse(tool_call_block)
                if parsed_tools:
                    outputs.extend(parsed_tools)
                
                # 3. Update buffer to what's left and continue loop
                self.buffer = self.buffer[end_of_tool_close:]
                continue
            
            # If no full tool call is found, break the loop
            break
        
        # After the loop, the buffer might contain the start of a tool call,
        # or just plain text. We should only yield the text part that is certain.
        start_idx = self.buffer.find(self.tool_open)
        if start_idx != -1:
            # We have an incomplete tool call. Yield text before it.
            if start_idx > 0:
                outputs.append(self.buffer[:start_idx])
                self.buffer = self.buffer[start_idx:]
        else:
            # No sign of a tool call, so it's all plain text.
            if self.buffer:
                outputs.append(self.buffer)
                self.buffer = ""
                
        return outputs if outputs else None


class Glm4MoeThinkingParser(BaseThinkingParser):
    """Parser for GLM4-MoE model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )
