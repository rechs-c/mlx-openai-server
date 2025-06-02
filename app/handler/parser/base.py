import json
import uuid
from typing import Any, Dict, List, Tuple


class BaseThinkingParser:
    def __init__(self, thinking_open: str, thinking_close: str):
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close
        self.is_thinking = False

    def parse(self, content: str) -> str:
        if self.thinking_open in content:
            start_thinking = content.find(self.thinking_open)
            end_thinking = content.find(self.thinking_close)
            if end_thinking != -1:
                return content[start_thinking + len(self.thinking_open):end_thinking].strip(), content[end_thinking + len(self.thinking_close):].strip()
        return None, content
    
    def parse_stream(self, chunk: str) -> Tuple[str, bool]:
        if not self.is_thinking:
            if chunk == self.thinking_open:
                self.is_thinking = True
                return None, False
            return chunk, False
        if chunk == self.thinking_close:
            self.is_thinking = False
            return None, True
        
        return {
            "reasoning_content": chunk
        }, False

class ParseState:
    NORMAL = 0
    FOUND_PREFIX = 1
    FOUND_FUNC_NAME = 2
    FOUND_FUNC_ARGS = 3
    PROCESS_FUNC_ARGS = 4
    @staticmethod
    def next_state(state):
        return (state + 1) % 5

class BaseToolParser:
    def __init__(self, tool_open: str, tool_close: str):
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.buffer = ""
        self.state = ParseState.NORMAL

    def get_tool_open(self):
        return self.tool_open
    
    def get_tool_close(self):
        return self.tool_close
    
    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        res = []
        start = 0
        while True:
            start_tool = content.find(self.tool_open, start)
            if start_tool == -1:
                break
            end_tool = content.find(self.tool_close, start_tool + len(self.tool_open))
            if end_tool == -1:
                break
            tool_content = content[start_tool + len(self.tool_open):end_tool].strip()

            try:
                json_output = json.loads(tool_content)  
                res.append(json_output)
            except json.JSONDecodeError:
                print("Error parsing tool call: ", tool_content)
                break
            start = end_tool + len(self.tool_close)
        return res, content[start:].strip()
    
    def parse_stream(self, chunk: str):
        if self.state == ParseState.NORMAL:
            if chunk.strip() == self.tool_open:
                self.state = ParseState.next_state(self.state)
                self.buffer = ""
                self.current_func = None
                return None
            return chunk

        if self.state == ParseState.FOUND_PREFIX:
            self.buffer += chunk
            # Try to parse function name
            if self.buffer.count('"') >= 4:
                # try parse json
                try:
                    json_output = json.loads(self.buffer.rstrip(',') + "}")
                    self.current_func = {
                        "name": None
                    }
                    self.state = ParseState.next_state(self.state)
                    return {
                        "name": json_output["name"],
                        "arguments": ""
                    }
                except json.JSONDecodeError:
                    return None
            return None

        if self.state == ParseState.FOUND_FUNC_NAME:
            # Try to parse function arguments
            if chunk.strip() == "arguments":
                self.state = ParseState.next_state(self.state)
                return None
            return None
        
        if self.state == ParseState.FOUND_FUNC_ARGS:
            if ":" in chunk:
                chunk = chunk[:chunk.find(":") + 1: ].lstrip()
                self.state = ParseState.next_state(self.state)
                if not chunk:
                    return None
            return None

        if '}\n' in chunk:
            chunk = chunk[:chunk.find('}\n')]

        if chunk == self.tool_close:
            # end of arguments
            # reset
            self.state = ParseState.NORMAL
            self.buffer = ""
            self.current_func = None
            return None

        return {
            "name": None,
            "arguments": chunk
        }
       
        
        
            
                
            
        
                    
        

        
            