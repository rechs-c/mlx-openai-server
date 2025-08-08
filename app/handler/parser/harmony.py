from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    StreamableParser,
    Role
)    
from typing import Tuple

# Harmony Parsing Helper Functions
class HarmonyParser:
    """Helper class for parsing GPT-OSS model responses using harmony encoding."""

    def __init__(self):
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.parser = StreamableParser(self.enc, role=Role.ASSISTANT)
        self.end_tool_chunk = "<|call|>"
        self.tool_state = False
        self.end_stream = False
    
    def parse_stream(self, text: str | None = None) -> Tuple[bool, dict | str | None]:   
        if text == self.end_tool_chunk:
            self.end_stream = True
            return self.end_stream, None

        if text:
            text_token = self.enc.encode(text, allowed_special="all")
            text_token = text_token[0]
            stream_text = self.parser.process(text_token)
            channel = stream_text.current_channel
            content = stream_text.last_content_delta
            if channel == "analysis":
                if content:
                    return self.end_stream, {
                        "reasoning_content": content
                    }
            elif channel == "commentary":
                if self.tool_state:
                    return self.end_stream, {
                        "name": None,
                        "arguments": content.replace("functions.", "")
                    }
                self.tool_state = True
                return self.end_stream, {
                    "name": stream_text.current_recipient.replace("function.", ""),
                    "arguments": ""
                }
            return self.end_stream, content
        return self.end_stream, None

    def parse(self, text: str) -> dict:
        res = {
            "reasoning_content": None,
            "tool_calls": None,
            "content": None
        }
        if self.end_tool_chunk in text:
            text = text.split(self.end_tool_chunk)[0]          
        tokens = self.enc.encode(text, allowed_special="all")
        parsed_messages = self.enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT)
        for message in parsed_messages:
            if message.channel == "analysis":
                res["reasoning_content"] = message.content[0].text
            elif message.channel == "commentary":
                res["tool_calls"] = [{
                    "name": message.recipient.replace("function.", ""),
                    "arguments": message.content[0].text
                }]
            elif message.channel == "final":
                res["content"] = message.content[0].text
        return res  