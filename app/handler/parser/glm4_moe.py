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
        self.state = ParseState.NORMAL
        self.buffer = ""
        self.current_func_name = None
        self.current_arg_key = None
        self.current_arguments = {}

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
            
            lines = inner_content.split('\n')
            func_name = lines[0].strip()
            
            arguments = {}
            # The rest of the lines contain arguments
            arg_content = "\n".join(lines[1:])
            
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

    def _reset_state(self):
        self.state = ParseState.NORMAL
        self.buffer = ""
        self.current_func_name = None
        self.current_arg_key = None
        self.current_arguments = {}

    def parse_stream(self, chunk: str):
        self.buffer += chunk
        
        if self.state == ParseState.NORMAL:
            if self.tool_open in self.buffer:
                self.state = ParseState.IN_TOOL_CALL
                # Discard content before tool_open
                self.buffer = self.buffer[self.buffer.find(self.tool_open):]
            else:
                # Not in a tool call, return the chunk as is
                self.buffer = ""
                return chunk

        if self.state == ParseState.IN_TOOL_CALL:
            # remove tool_open tag
            self.buffer = self.buffer.replace(self.tool_open, "", 1)
            self.state = ParseState.PARSING_FUNC_NAME
        
        if self.state == ParseState.PARSING_FUNC_NAME:
            if '\n' in self.buffer or ARG_KEY_OPEN in self.buffer:
                # Determine the split token based on which appears first
                pos_newline = self.buffer.find('\n')
                pos_arg_key = self.buffer.find(ARG_KEY_OPEN)

                # -1 means not found, so we treat it as infinity
                if pos_newline == -1: pos_newline = float('inf')
                if pos_arg_key == -1: pos_arg_key = float('inf')

                if pos_newline < pos_arg_key:
                    split_token = '\n'
                elif pos_arg_key < pos_newline:
                    split_token = ARG_KEY_OPEN
                else: # Neither is found, or buffer is empty
                    return None

                func_name_part, rest = self.buffer.split(split_token, 1)
                self.current_func_name = func_name_part.strip()
                
                # Prepend the split_token back if it was a tag
                self.buffer = rest if split_token == '\n' else split_token + rest
                
                self.state = ParseState.PARSING_ARG_KEY
                return {
                    "name": self.current_func_name,
                    "arguments": ""
                }
        
        if self.state == ParseState.PARSING_ARG_KEY:
            if ARG_KEY_OPEN in self.buffer and ARG_KEY_CLOSE in self.buffer:
                start_key_idx = self.buffer.find(ARG_KEY_OPEN)
                end_key_idx = self.buffer.find(ARG_KEY_CLOSE)
                self.current_arg_key = self.buffer[start_key_idx + len(ARG_KEY_OPEN):end_key_idx].strip()
                self.buffer = self.buffer[end_key_idx + len(ARG_KEY_CLOSE):]
                self.state = ParseState.PARSING_ARG_VALUE

        if self.state == ParseState.PARSING_ARG_VALUE:
            # Stream out argument chunks as they arrive
            # This part is tricky because we need to form valid JSON chunks
            # A simplified approach: buffer until a full value is parsed
            if ARG_VALUE_OPEN in self.buffer and ARG_VALUE_CLOSE in self.buffer:
                start_val_idx = self.buffer.find(ARG_VALUE_OPEN)
                end_val_idx = self.buffer.find(ARG_VALUE_CLOSE)
                
                # Ensure we have the full value tag pair
                if start_val_idx < end_val_idx:
                    value_part = self.buffer[start_val_idx + len(ARG_VALUE_OPEN):end_val_idx]
                    self.current_arguments[self.current_arg_key] = value_part
                    
                    # Yield the argument as a JSON string part
                    # To simplify, we yield the full argument dict each time
                    arg_str = json.dumps({self.current_arg_key: value_part}, ensure_ascii=False)[1:-1] # remove {}
                    
                    self.buffer = self.buffer[end_val_idx + len(ARG_VALUE_CLOSE):]
                    self.state = ParseState.PARSING_ARG_KEY # Ready for next key
                    self.current_arg_key = None
                    
                    # This is a simplification. In reality, streaming JSON arguments is more complex.
                    # We return a chunk of the arguments string.
                    return {
                        "name": None,
                        "arguments": arg_str + ", " # Add comma for next arg
                    }

        if self.tool_close in self.buffer:
            # End of tool call
            # Clean up the buffer from the close tag
            self.buffer = self.buffer[:self.buffer.find(self.tool_close)]
            
            # Potentially process any remaining buffer content if needed
            # For now, we just reset
            
            final_args_str = ""
            if self.buffer.strip() and self.current_arg_key:
                 # This handles the final part of an argument value before the close tag
                 final_args_str = self.buffer.strip()

            self._reset_state()
            # Return the final piece of argument and signal completion
            return {
                "name": None,
                "arguments": final_args_str
            }

        # If nothing complete is parsed, return None to signal waiting for more chunks
        return None


class Glm4MoeThinkingParser(BaseThinkingParser):
    """Parser for GLM4-MoE model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )
