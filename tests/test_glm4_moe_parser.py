import unittest
import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.handler.parser.glm4_moe import Glm4MoeToolParser

class TestGlm4MoeToolParser(unittest.TestCase):

    def setUp(self):
        self.parser = Glm4MoeToolParser()

    def test_single_tool_call(self):
        content = """我来帮您计算 50 乘以 3，并查询今天是星期几。
<tool_call>calculate_math_operation
<arg_key>operation</arg_key>
<arg_value>multiply</arg_value>
<arg_key>a</arg_key>
<arg_value>50</arg_value>
<arg_key>b</arg_key>
<arg_value>3</arg_value>
</tool_call>"""
        
        expected_tool_calls = [{
            "name": "calculate_math_operation",
            "arguments": {
                "operation": "multiply",
                "a": "50",
                "b": "3"
            }
        }]
        
        tool_calls, remaining_content = self.parser.parse(content)
        
        self.assertEqual(tool_calls, expected_tool_calls)
        self.assertEqual(remaining_content, "我来帮您计算 50 乘以 3，并查询今天是星期几。")

    def test_multiple_tool_calls(self):
        content = """我来帮您计算 50 乘以 3，并查询今天是星期几。
<tool_call>calculate_math_operation
<arg_key>operation</arg_key>
<arg_value>multiply</arg_value>
<arg_key>a</arg_key>
<arg_value>50</arg_value>
<arg_key>b</arg_key>
<arg_value>3</arg_value>
</tool_call>
<tool_call>get_current_time
<arg_key>format_str</arg_key>
<arg_value>%Y-%m-%d</arg_value>
</tool_call>"""
        
        expected_tool_calls = [
            {
                "name": "calculate_math_operation",
                "arguments": {
                    "operation": "multiply",
                    "a": "50",
                    "b": "3"
                }
            },
            {
                "name": "get_current_time",
                "arguments": {
                    "format_str": "%Y-%m-%d"
                }
            }
        ]
        
        tool_calls, remaining_content = self.parser.parse(content)
        
        self.assertEqual(tool_calls, expected_tool_calls)
        self.assertEqual(remaining_content, "我来帮您计算 50 乘以 3，并查询今天是星期几。")

    def test_no_tool_call(self):
        content = "Hello, how are you?"
        tool_calls, remaining_content = self.parser.parse(content)
        self.assertEqual(tool_calls, [])
        self.assertEqual(remaining_content, "Hello, how are you?")

    def test_json_in_args(self):
        content = """
<tool_call>some_function
<arg_key>data</arg_key>
<arg_value>{"key": "value"}</arg_value>
</tool_call>"""
        expected_tool_calls = [{
            "name": "some_function",
            "arguments": {
                "data": {"key": "value"}
            }
        }]
        tool_calls, remaining_content = self.parser.parse(content)
        self.assertEqual(tool_calls, expected_tool_calls)

    def test_stream_single_tool_call(self):
        content = """我来帮您计算 50 乘以 3，并查询今天是星期几。
<tool_call>calculate_math_operation
<arg_key>operation</arg_key>
<arg_value>multiply</arg_value>
<arg_key>a</arg_key>
<arg_value>50</arg_value>
<arg_key>b</arg_key>
<arg_value>3</arg_value>
</tool_call>"""
        
        expected_tool_calls = [{
            "name": "calculate_math_operation",
            "arguments": {
                "operation": "multiply",
                "a": "50",
                "b": "3"
            }
        }]
        
        chunks = [content[i:i+5] for i in range(0, len(content), 5)]
        
        parsed_items = []
        # Process the content chunk by chunk
        for chunk in chunks:
            result = self.parser.parse_stream(chunk)
            if result:
                if isinstance(result, list):
                    parsed_items.extend(result)
                else:
                    parsed_items.append(result)

        # After all chunks are processed, there might be remaining content in the buffer
        remaining = self.parser.parse_stream("") # Pass empty string to flush buffer
        if remaining:
            if isinstance(remaining, list):
                parsed_items.extend(remaining)
            else:
                parsed_items.append(remaining)

        self.assertTrue(any(isinstance(item, str) for item in parsed_items))
        self.assertTrue(any(isinstance(item, dict) for item in parsed_items))
        
        # Find the tool call and assert its correctness
        tool_call = next((item for item in parsed_items if isinstance(item, dict)), None)
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call, expected_tool_calls[0])

if __name__ == '__main__':
    unittest.main()