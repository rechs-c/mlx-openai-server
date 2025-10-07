import json
import re
from typing import Any, Dict, List, Optional, Tuple
from app.handler.parser.base import BaseToolParser, BaseThinkingParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"

class Glm4MoEThinkingParser(BaseThinkingParser):
    """Parser for GLM4 model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )

class Glm4MoEToolParser(BaseToolParser):
    """Parser for GLM4 model's tool response format with XML-style arguments."""
    
    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE
        )
        # Regex patterns for parsing GLM4 XML-style tool calls
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>([^\n]*)\n(.*)</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )
        # State for streaming parsing
        self.stream_buffer = ""
        self.current_func_name = None
        self.current_args = {}
        self.parsing_tool = False
        
    def _deserialize_value(self, value: str) -> Any:
        """Try to deserialize a value from string to appropriate Python type."""
        value = value.strip()
        
        # Try JSON parsing first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try literal eval for Python literals
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        
        # Return as string if all else fails
        return value
    
    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse complete content for GLM4 tool calls.
        
        Returns:
            Tuple of (list of tool calls, remaining content)
        """
        tool_calls = []
        matched_calls = self.func_call_regex.findall(content)
        
        try:
            for match in matched_calls:
                # Extract function name and arguments section
                detail_match = self.func_detail_regex.search(match)
                if not detail_match:
                    continue
                
                func_name = detail_match.group(1).strip()
                args_section = detail_match.group(2)
                
                # Extract all key-value pairs
                arg_pairs = self.func_arg_regex.findall(args_section)
                arguments = {}
                for key, value in arg_pairs:
                    arg_key = key.strip()
                    arg_value = self._deserialize_value(value)
                    arguments[arg_key] = arg_value
                
                # Build tool call object
                tool_calls.append({
                    "name": func_name,
                    "arguments": json.dumps(arguments)
                })
        except Exception as e:
            print(f"Error parsing GLM4 tool call: {e}")
        
        # Find content before first tool call
        first_tool_idx = content.find(self.tool_open)
        if first_tool_idx != -1:
            remaining_content = content[:first_tool_idx].strip()
        else:
            remaining_content = content.strip()
        
        return tool_calls, remaining_content
    
    def parse_stream(self, chunk: str) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for GLM4 tool calls.
        
        This handles the XML-style format incrementally.
        
        Returns:
            Tuple[parsed_content, is_complete]:
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if tool call is complete
        """
        if chunk is None:
            return None, False
        
        self.stream_buffer += chunk
        
        # Check if we're starting a tool call
        if not self.parsing_tool:
            if self.tool_open in self.stream_buffer:
                tool_start_idx = self.stream_buffer.find(self.tool_open)
                # Return any content before the tool call
                content_before = self.stream_buffer[:tool_start_idx]
                self.stream_buffer = self.stream_buffer[tool_start_idx + len(self.tool_open):]
                self.parsing_tool = True
                self.current_func_name = None
                self.current_args = {}
                
                if content_before:
                    return content_before, False
                return None, False
            else:
                # No tool call found yet, return the content (except last few chars as buffer)
                if len(self.stream_buffer) > len(self.tool_open):
                    content_to_return = self.stream_buffer[:-len(self.tool_open)]
                    self.stream_buffer = self.stream_buffer[-len(self.tool_open):]
                    if content_to_return:
                        return content_to_return, False
                return None, False
        
        # We're inside a tool call
        if self.tool_close in self.stream_buffer:
            tool_end_idx = self.stream_buffer.find(self.tool_close)
            tool_content = self.stream_buffer[:tool_end_idx]
            self.stream_buffer = self.stream_buffer[tool_end_idx + len(self.tool_close):]
            
            # Parse the complete tool call
            full_tool = f"{self.tool_open}{tool_content}{self.tool_close}"
            parsed_tools, _ = self.parse(full_tool)
            
            self.parsing_tool = False
            self.current_func_name = None
            self.current_args = {}
            
            if parsed_tools:
                tool = parsed_tools[0]
                # Return the complete tool call information
                return {
                    "name": tool["name"],
                    "arguments": tool["arguments"]
                }, True  # Tool call complete
            return None, True
        
        # Still accumulating the tool call
        # Try to extract function name if we haven't yet
        if self.current_func_name is None:
            if '\n' in self.stream_buffer or len(self.stream_buffer) > 50:
                # Extract function name (first line)
                newline_idx = self.stream_buffer.find('\n')
                if newline_idx != -1:
                    self.current_func_name = self.stream_buffer[:newline_idx].strip()
                    self.stream_buffer = self.stream_buffer[newline_idx + 1:]
                    # Return function name
                    return {
                        "name": self.current_func_name,
                        "arguments": ""
                    }, False
        
        # Check if we can parse any complete argument pairs
        arg_matches = list(self.func_arg_regex.finditer(self.stream_buffer))
        if arg_matches:
            last_match = arg_matches[-1]
            # Only process if we have the complete closing tag
            if last_match.end() < len(self.stream_buffer):
                for match in arg_matches:
                    arg_key = match.group(1).strip()
                    arg_value = self._deserialize_value(match.group(2))
                    if arg_key not in self.current_args:
                        self.current_args[arg_key] = arg_value
                
                # Remove processed content from buffer
                self.stream_buffer = self.stream_buffer[last_match.end():]
                
                # Return incremental arguments
                if self.current_args:
                    return {
                        "name": None,
                        "arguments": json.dumps(self.current_args)
                    }, False
        
        return None, False