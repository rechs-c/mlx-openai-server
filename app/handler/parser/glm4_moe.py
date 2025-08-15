import re
import json
from typing import Any, Dict, List, Tuple
from app.handler.parser.base import BaseToolParser, ParseState

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
ARG_KEY_OPEN = "<arg_key>"
ARG_KEY_CLOSE = "</arg_key>"
ARG_VALUE_OPEN = "<arg_value>"
ARG_VALUE_CLOSE = "</arg_value>"

class Glm4MoeToolParser(BaseToolParser):
    """Parser for GLM-4.5 model's tool response format."""

    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE
        )
        # Override regex from parent for non-streaming parse
        self.func_detail_regex = r"([^\n]*)\n(.*)"
        self.func_arg_regex = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"
        
        # Streaming-specific state
        self.current_func_name = None
        self.current_arguments = {}
        self.current_arg_key = None
        self.temp_buffer = ""

    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        res = []
        remaining_content = content
        
        tool_calls_str = re.findall(f"{self.tool_open}(.*?){self.tool_close}", content, re.DOTALL)

        for tool_content in tool_calls_str:
            tool_content = tool_content.strip()
            try:
                func_detail = re.search(self.func_detail_regex, tool_content, re.DOTALL)
                if not func_detail:
                    continue

                func_name = func_detail.group(1).strip()
                func_args_str = func_detail.group(2)

                pairs = re.findall(self.func_arg_regex, func_args_str, re.DOTALL)
                
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    try:
                        if (arg_value.startswith('{') and arg_value.endswith('}')) or \
                           (arg_value.startswith('[') and arg_value.endswith(']')):
                            arguments[arg_key] = json.loads(arg_value)
                        else:
                            arguments[arg_key] = arg_value
                    except json.JSONDecodeError:
                        arguments[arg_key] = arg_value

                json_output = {
                    "name": func_name,
                    "arguments": arguments
                }
                res.append(json_output)

            except Exception as e:
                print(f"Error parsing tool call: {tool_content}, error: {e}")

        final_content_start = content.find(self.tool_open)
        final_content = content[:final_content_start].strip() if final_content_start != -1 else content.strip()

        return res, final_content

    def _reset_streaming_state(self):
        self.state = ParseState.NORMAL
        self.current_func_name = None
        self.current_arguments = {}
        self.current_arg_key = None
        self.temp_buffer = ""

    def parse_stream(self, chunk: str):
        self.buffer += chunk

        if self.state == ParseState.NORMAL:
            start_index = self.buffer.find(self.tool_open)
            if start_index == -1:
                text_to_yield = self.buffer
                self.buffer = ""
                return text_to_yield if text_to_yield else None
            
            text_to_yield = self.buffer[:start_index]
            self.buffer = self.buffer[start_index + len(self.tool_open):]
            self.state = ParseState.FOUND_PREFIX
            if text_to_yield:
                return text_to_yield
        
        if self.state == ParseState.FOUND_PREFIX:
            # First line is the function name
            newline_index = self.buffer.find('\n')
            if newline_index != -1:
                self.current_func_name = self.buffer[:newline_index].strip()
                self.buffer = self.buffer[newline_index + 1:]
                self.state = ParseState.FOUND_FUNC_NAME
                return {
                    "name": self.current_func_name,
                    "arguments": ""
                }
        
        if self.state == ParseState.FOUND_FUNC_NAME:
            # Now we are parsing arguments
            # This is a simplified parser that accumulates arguments and returns them in one go at the end.
            # A true token-by-token streaming of arguments is much more complex.
            
            end_index = self.buffer.find(self.tool_close)
            if end_index != -1:
                # Found the end of the tool call
                args_str = self.buffer[:end_index]
                self.buffer = self.buffer[end_index + len(self.tool_close):]

                # Use the regex from non-streaming parse to get all args at once
                pairs = re.findall(self.func_arg_regex, args_str, re.DOTALL)
                arguments = {}
                for arg_key, arg_value in pairs:
                    arguments[arg_key.strip()] = arg_value.strip()

                # Reset for the next potential tool call
                self._reset_streaming_state()
                
                # Return the arguments chunk
                return {
                    "name": None,
                    "arguments": json.dumps(arguments)
                }

        # If we are in the middle of parsing, or if there's trailing text, handle it.
        if self.state == ParseState.NORMAL and self.buffer:
            return self.parse_stream("")

        return None
