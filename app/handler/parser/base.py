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

class BaseToolParser:
    def __init__(self, tool_open: str, tool_close: str):
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.buffer = ""

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
                
                # 2. Parse the tool call
                end_of_tool_close = end_idx + len(self.tool_close)
                tool_call_block = self.buffer[start_idx:end_of_tool_close]
                
                parsed_tools, _ = self.parse(tool_call_block)
                
                if parsed_tools:
                    # For each complete tool, split it into OpenAI-compatible delta chunks
                    for tool in parsed_tools:
                        # First chunk with the name
                        outputs.append({
                            "name": tool.get("name"),
                            "arguments": ""
                        })
                        # Second chunk with the arguments
                        # Note: The arguments from parse() should be a dict
                        arguments_str = json.dumps(tool.get("arguments", {}), ensure_ascii=False)
                        outputs.append({
                            "name": None,
                            "arguments": arguments_str
                        })

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
