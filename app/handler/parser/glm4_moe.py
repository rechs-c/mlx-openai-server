import re
import json
from typing import Any, Dict, List, Tuple
from app.handler.parser.base import BaseToolParser, ParseState

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

class Glm4MoeToolParser(BaseToolParser):
    """Parser for GLM-4.5 model's tool response format."""

    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE
        )
        self.func_detail_regex = r"([^\n]*)\n(.*)"
        self.func_arg_regex = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"

    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        res = []
        remaining_content = content
        
        # Find all tool calls
        tool_calls_str = re.findall(f"{self.tool_open}(.*?){self.tool_close}", content, re.DOTALL)

        for tool_content in tool_calls_str:
            tool_content = tool_content.strip()
            try:
                # Get function name
                func_detail = re.search(self.func_detail_regex, tool_content, re.DOTALL)
                if not func_detail:
                    continue

                func_name = func_detail.group(1).strip()
                func_args_str = func_detail.group(2)

                # Find all argument key-value pairs
                pairs = re.findall(self.func_arg_regex, func_args_str, re.DOTALL)
                
                arguments = {}
                for arg_key, arg_value in pairs:
                    arg_key = arg_key.strip()
                    arg_value = arg_value.strip()
                    try:
                        # Attempt to parse the value as JSON if it looks like a dict or list
                        if (arg_value.startswith('{') and arg_value.endswith('}')) or \
                           (arg_value.startswith('[') and arg_value.endswith(']')):
                            arguments[arg_key] = json.loads(arg_value)
                        else:
                            arguments[arg_key] = arg_value
                    except json.JSONDecodeError:
                        arguments[arg_key] = arg_value # Keep as string if JSON parsing fails

                # Construct the final JSON object
                json_output = {
                    "name": func_name,
                    "arguments": arguments
                }
                res.append(json_output)

            except Exception as e:
                print(f"Error parsing tool call: {tool_content}, error: {e}")

        # The text before the first tool call is the remaining content
        final_content_start = content.find(self.tool_open)
        final_content = content[:final_content_start].strip() if final_content_start != -1 else content.strip()

        return res, final_content

    def parse_stream(self, chunk: str):
        self.buffer += chunk
        
        if self.state == ParseState.NORMAL:
            start_index = self.buffer.find(self.tool_open)
            if start_index != -1:
                text_to_yield = self.buffer[:start_index]
                self.buffer = self.buffer[start_index:]
                self.state = ParseState.FOUND_PREFIX
                if text_to_yield:
                    return text_to_yield
                else:
                    # Continue to next state if no text to yield
                    return self.parse_stream("")

        if self.state == ParseState.FOUND_PREFIX:
            end_index = self.buffer.find(self.tool_close)
            if end_index != -1:
                tool_call_str = self.buffer[:end_index + len(self.tool_close)]
                self.buffer = self.buffer[end_index + len(self.tool_close):]
                
                parsed_tools, _ = self.parse(tool_call_str)
                self.state = ParseState.NORMAL
                
                if parsed_tools:
                    return parsed_tools[0]

        return None