import unittest
import json

from app.handler.parser.glm4_moe import GLM4MoeToolParser


class TestGLM4MoeToolParser(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "name": "glm4_moe function call",
                "chunks": '''<tool_call>#record_intent#
#<arg_key>#intent#</arg_key>#
#<arg_value>#用户询问我的功能和能力范围，希望了解我能够提供哪些服务和支持#</arg_value>#
#<arg_key>#keywords#</arg_key>#
#<arg_value>#功能,能力,服务,支持#</arg_value>#
#<arg_key>#is_follow_up#</arg_key>#
#<arg_value>#false#</arg_value>#
#<arg_key>#is_query#</arg_key>#
#<arg_value>#false#</arg_value>#
#<arg_key>#analysis#</arg_key>#
#<arg_value>#用户进行一般性询问，了解我的功能范围和服务能力，属于元信息查询性质#</arg_value>#
#<arg_key>#knowledge_keywords#</arg_key>#
#<arg_value>#</arg_value>#
#</tool_call>'''.split('#'),
                "expected_outputs": [
                    {'name': 'record_intent', 'arguments': ''},
                    {'name': None, 'arguments': '{"intent": "用户询问我的功能和能力范围，希望了解我能够提供哪些服务和支持", '},
                    {'name': None, 'arguments': '"keywords": "功能,能力,服务,支持", '},
                    {'name': None, 'arguments': '"is_follow_up": "false", '},
                    {'name': None, 'arguments': '"is_query": "false", '},
                    {'name': None, 'arguments': '"analysis": "用户进行一般性询问，了解我的功能范围和服务能力，属于元信息查询性质", '},
                    {'name': None, 'arguments': '"knowledge_keywords": ""}'},
                ]
            }
        ]

    def test_parse_stream(self):
        for test_case in self.test_cases:
            with self.subTest(msg=test_case["name"]):
                parser = GLM4MoeToolParser()
                outputs = []
                
                for chunk in test_case["chunks"]:
                    result = parser.parse_stream(chunk)
                    if result:
                        outputs.append(result)

                # Combine arguments for comparison
                full_output = {}
                if outputs:
                    full_output['name'] = outputs[0]['name']
                    arguments_str = "".join(o['arguments'] for o in outputs)
                    # Post-process the combined string to be valid JSON
                    arguments_str = arguments_str.strip()
                    if arguments_str.endswith(','):
                        arguments_str = arguments_str[:-1]
                    if not arguments_str.startswith('{'):
                        arguments_str = '{' + arguments_str
                    if not arguments_str.endswith('}'):
                        arguments_str = arguments_str + '}'
                    
                    try:
                        full_output['arguments'] = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        self.fail(f"Failed to decode arguments JSON: {arguments_str}")


                full_expected = {}
                if test_case["expected_outputs"]:
                    full_expected['name'] = test_case["expected_outputs"][0]['name']
                    expected_args_str = "".join(o['arguments'] for o in test_case["expected_outputs"])
                    try:
                        full_expected['arguments'] = json.loads(expected_args_str)
                    except json.JSONDecodeError:
                         self.fail(f"Failed to decode expected arguments JSON: {expected_args_str}")

                self.assertEqual(full_output, full_expected,
                    f"Mismatched tool call objects.\nGot: {full_output}\nExpected: {full_expected}")


if __name__ == '__main__':
    unittest.main()
