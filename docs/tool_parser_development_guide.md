# 模型工具调用解析器开发指南

## 1. 概述

本文档旨在为开发者提供一个清晰、灵活的指南，用于为那些通过特定XML标签输出工具调用（Tool Call）的大语言模型（LLM）开发解析器。我们的目标是将模型输出的、由特定标签包裹的工具调用信息，准确地解析并转换为符合OpenAI接口规范的数据结构。

我们将以 `Qwen3` 模型的工具解析实现（[`Qwen3ToolParser`](app/handler/parser/qwen3.py:8)）为例，深入剖析基于核心基类 [`BaseToolParser`](app/handler/parser/base.py:46) 的工作原理，并展示如何利用其可扩展的设计来适配不同格式（JSON、XML、YAML等）的工具调用。

## 2. 核心设计：`BaseToolParser`

工具解析的核心逻辑被抽象并封装在基类 [`BaseToolParser`](app/handler/parser/base.py:46) 中。该设计采用了**模板方法模式**，将通用的解析流程（如标签识别、内容提取、流式处理）与具体的内容解析逻辑分离开来，从而实现了高度的可扩展性和代码复用。

### 2.1. 核心组件

#### a. 初始化 (`__init__`)

[`BaseToolParser`](app/handler/parser/base.py:46) 的构造函数接收两个关键参数，用于识别工具调用的边界：

-   `tool_open` (str): 标志工具调用代码块开始的XML标签。
-   `tool_close` (str): 标志工具调用代码块结束的XML标签。

#### b. 通用解析流程 (`parse` 和 `parse_stream`)

基类固化了两个核心的解析流程方法：

-   [`parse(content: str)`](app/handler/parser/base.py:59): 用于一次性处理和解析模型返回的完整文本。
-   [`parse_stream(chunk: str)`](app/handler/parser/base.py:83): 通过内置的状态机和缓冲区，专门用于处理分块返回的流式响应。

这两个方法负责**提取**位于 `tool_open` 和 `tool_close` 标签之间的内容，但它们**不关心内容的具体格式**。

#### c. 内容解析模板方法 (`_parse_tool_content`)

这是框架设计的关键所在。[`_parse_tool_content(tool_content: str)`](app/handler/parser/base.py:59) 是一个可被子类重写（override）的方法，专门用于**解析**从标签中提取出的 `tool_content` 字符串。

-   **默认实现**: 基类为该方法提供了一个默认实现，用于处理最常见的JSON格式。它会先使用 `json_repair` 库修复可能不规范的JSON，然后通过 `json.loads` 将其转换为Python字典。
-   **可扩展性**: 子类可以通过重写此方法，来注入任何自定义的解析逻辑，从而支持XML、YAML或任何其他自定义格式。

## 3. 如何适配新模型

得益于模板方法的设计，为新模型开发解析器变得异常简单和灵活。

### 场景一：新模型使用JSON格式（类似Qwen3）

如果你的新模型 `JsonLLM` 同样使用JSON格式，但标签不同（例如 `<tool>` 和 `</tool>`），你只需继承 `BaseToolParser` 并提供新的标签即可。你**无需**重写 `_parse_tool_content` 方法，因为它将自动继承并使用默认的JSON解析逻辑。

```python
# app/handler/parser/json_llm.py

from app.handler.parser.base import BaseToolParser

class JsonLLMToolParser(BaseToolParser):
    """Parser for JsonLLM model's tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open="<tool>",
            tool_close="</tool>"
        )
```

### 场景二：新模型使用非JSON格式（例如XML）

假设你的新模型 `XmlLLM` 返回XML格式的工具调用，标签为 `<function_call>` 和 `</function_call>`。

```xml
<function_call>
    <name>get_weather</name>
    <arguments>
        <city>Boston</city>
    </arguments>
</function_call>
```

在这种情况下，你需要：

1.  **继承 `BaseToolParser`** 并提供新的标签。
2.  **重写 `_parse_tool_content` 方法**，在其中实现XML解析逻辑，并确保返回一个符合规范的Python字典。

```python
# app/handler/parser/xml_llm.py

import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional
from app.handler.parser.base import BaseToolParser

class XmlLLMToolParser(BaseToolParser):
    """Parser for XmlLLM model's XML-based tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open="<function_call>",
            tool_close="</function_call>"
        )

    def _parse_tool_content(self, tool_content: str) -> Optional[Dict[str, Any]]:
        """
        Overrides the base method to parse XML tool call content.
        """
        try:
            # 注意：为简化示例，这里的XML解析比较简单
            # 在实际应用中可能需要更健壮的解析库和错误处理
            root = ET.fromstring(tool_content)
            name = root.find("name").text
            args_root = root.find("arguments")
            arguments = {child.tag: child.text for child in args_root}
            
            return {
                "name": name,
                "arguments": arguments
            }
        except ET.ParseError as e:
            print(f"Error parsing XML tool call content: {tool_content}, Error: {e}")
            return None
```

通过这种方式，你可以在不修改任何基类代码的情况下，无缝地将自定义的XML解析逻辑集成到框架的通用处理流程中。

## 4. 总结

本项目的工具调用解析框架通过一个通用的基类 `BaseToolParser` 和特定于模型的子类相结合，实现了**通用流程与具体实现的分离**。其核心优势在于：

-   **高内聚，低耦合**：通用逻辑（如流式处理）封装在基类，具体实现（如内容解析）委托给子类。
-   **易于扩展**：通过重写 `_parse_tool_content` 方法，可以轻松支持任意数据格式。
-   **代码简洁**：适配新模型的大部分工作被简化为定义标签和实现一个解析方法。