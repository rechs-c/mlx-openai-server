import json
from typing import Any, Dict, List, Tuple
from app.handler.parser.base import BaseToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

class Glm4ToolParser(BaseToolParser):
    def __init__(self):
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        self.reset()

    def reset(self):
        self.buffer = ""
        self.parsed_chunks = []

    def _parse_tool_content(self, tool_content: str) -> Tuple[str, Dict]:
        """Helper function to parse the content of a tool call."""
        lines = tool_content.strip().split('\n')
        tool_name = lines[0]
        
        args = {}
        i = 1
        while i < len(lines):
            line = lines[i]
            if '<arg_key>' in line:
                key_start = line.find('<arg_key>') + len('<arg_key>')
                key_end = line.find('</arg_key>')
                if key_end == -1:
                    i += 1
                    continue
                key = line[key_start:key_end]
                
                if i + 1 < len(lines):
                    value_line = lines[i+1]
                    if '<arg_value>' in value_line:
                        value_start = value_line.find('<arg_value>') + len('<arg_value>')
                        value_end = value_line.find('</arg_value>')
                        if value_end == -1:
                            i += 1
                            continue
                        value_str = value_line[value_start:value_end]
                        
                        try:
                            args[key] = int(value_str)
                        except ValueError:
                            try:
                                args[key] = float(value_str)
                            except ValueError:
                                args[key] = value_str
                        
                        i += 2
                        continue
            i += 1
        return tool_name, args

    def parse_stream(self, chunk: str) -> Dict[str, Any]:
        self.buffer += chunk
        
        start_tag_idx = self.buffer.find(self.tool_open)
        if start_tag_idx == -1:
            return None

        end_tag_idx = self.buffer.find(self.tool_close, start_tag_idx)
        if end_tag_idx == -1:
            return None # Incomplete tool call, wait for more chunks

        # Extract complete tool call content
        tool_content_start = start_tag_idx + len(self.tool_open)
        tool_content = self.buffer[tool_content_start:end_tag_idx].strip()
        
        # Update buffer to remove the processed tool call
        self.buffer = self.buffer[end_tag_idx + len(self.tool_close):]

        # Parse the content
        tool_name, args = self._parse_tool_content(tool_content)

        # Return a single, complete tool call object
        return {
            "name": tool_name,
            "arguments": json.dumps(args),
            "is_full_tool_call": True # Flag for the handler
        }

    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        res = []
        remaining_content = content
        while True:
            start_tool = remaining_content.find(self.tool_open)
            if start_tool == -1:
                break
            
            end_tool = remaining_content.find(self.tool_close, start_tool)
            if end_tool == -1:
                break
            
            tool_content_start = start_tool + len(self.tool_open)
            tool_content = remaining_content[tool_content_start:end_tool].strip()
            
            tool_name, args = self._parse_tool_content(tool_content)
            
            res.append({
                "name": tool_name,
                "arguments": args
            })
            
            remaining_content = remaining_content[end_tool + len(self.tool_close):]
        
        return res, remaining_content.strip()
