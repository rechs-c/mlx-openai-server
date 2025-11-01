# **模型工具调用解析器开发指南**

## 1. 概述

本文档旨在为开发者提供一个清晰的指南，用于为那些通过特定XML标签输出工具调用（Tool Call）的大语言模型（LLM）开发解析器。我们将以 `Qwen3` 模型的工具解析实现（[`Qwen3ToolParser`](app/handler/parser/qwen3.py:8)）为例，深入剖析其工作原理，并展示如何基于现有的框架（[`BaseToolParser`](app/handler/parser/base.py:46)）快速适配新的模型。

我们的目标是将模型输出的、由特定标签包裹的工具调用信息，准确地解析并转换为符合OpenAI接口规范的数据结构。

## 2. 核心设计：`BaseToolParser`

工具解析的核心逻辑被抽象并封装在基类 [`BaseToolParser`](app/handler/parser/base.py:46) 中。这种设计使得解析框架具备了高度的可扩展性和可复用性。开发者在适配新模型时，无需重写复杂的解析逻辑，只需继承该基类并提供模型特定的配置即可。

### 2.1. 初始化

[`BaseToolParser`](app/handler/parser/base.py:46) 的构造函数接收两个关键参数：

-   `tool_open` (str): 标志工具调用代码块开始的XML标签。
-   `tool_close` (str): 标志工具调用代码块结束的XML标签。

例如，在 `Qwen3` 的实现中，这两个参数被分别设置为 `<tool_call>` 和 `</tool_call>`。

```python
# app/handler/parser/qwen3.py

class Qwen3ToolParser(BaseToolParser):
    """Parser for Qwen3 model's tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open="<tool_call>",
            tool_close="</tool_call>"   
        )
```

### 2.2. 解析流程

[`BaseToolParser`](app/handler/parser/base.py:46) 提供了两种解析方法，以应对不同的应用场景：

#### a. 完整内容解析 (`parse`)

[`parse(content: str)`](app/handler/parser/base.py:59) 方法用于一次性处理模型返回的完整文本。其工作流程如下：

1.  **查找标签**：在输入字符串 `content` 中查找由 `tool_open` 和 `tool_close` 定义的工具调用代码块。
2.  **提取内容**：提取标签内部的字符串，该字符串应为一个JSON对象。
3.  **修复与解析**：
    -   首先，使用 `json_repair` 库对提取的字符串进行修复。这一步增强了解析器的鲁棒性，能够处理一些轻微不规范的JSON格式。
    -   然后，使用标准的 `json.loads` 将修复后的字符串解析为Python字典。
4.  **返回结果**：方法返回一个元组，包含解析出的工具调用列表（`tool_calls`）和剩余的非工具调用文本内容（`remaining_content`）。

#### b. 流式内容解析 (`parse_stream`)

在流式（streaming）响应的场景下，模型输出是逐块（chunk）返回的。[`parse_stream(chunk: str)`](app/handler/parser/base.py:83) 方法专门用于处理这种情况。

它通过一个简单的状态机（`self.state`）和缓冲区（`self.buffer`）来管理解析过程：

1.  **状态切换**：当在数据块中检测到 `tool_open` 标签时，状态切换为 `FOUND_PREFIX`，解析器开始进入“监听”模式。
2.  **内容缓冲**：在 `FOUND_PREFIX` 状态下，所有后续的数据块都会被追加到 `self.buffer` 中。
3.  **完成解析**：当在数据块中检测到 `tool_close` 标签时，标志着一个完整的工具调用代码块接收完毕。此时，解析器会：
    -   将 `tool_close` 标签之前的内容追加到缓冲区。
    -   对缓冲区中的完整内容执行与 `parse` 方法相同的JSON修复与解析流程。
    -   清空缓冲区，并将状态重置为 `NORMAL`，准备下一次解析。
4.  **返回结果**：该方法同样返回一个元组，包含解析出的单个工具调用对象和布尔值 `is_complete`，用于告知调用者该工具调用是否已解析完毕。

## 3. 如何适配新模型

得益于 [`BaseToolParser`](app/handler/parser/base.py:46) 的设计，为新模型开发工具解析器变得异常简单。假设有一个新的模型 `FooLLM`，它使用 `<function_call>` 和 `</function_call>` 作为工具调用的标签，你只需完成以下两步：

1.  在 `app/handler/parser/` 目录下创建一个新文件，例如 `foo_llm.py`。
2.  在该文件中，创建一个继承自 `BaseToolParser` 的新类，并在构造函数中传入 `FooLLM` 特定的标签：

```python
# app/handler/parser/foo_llm.py

from app.handler.parser.base import BaseToolParser

class FooLLMToolParser(BaseToolParser):
    """Parser for FooLLM model's tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open="<function_call>",
            tool_close="</function_call>"
        )
```

至此，`FooLLM` 的工具解析器便已开发完成，它可以直接复用基类中所有成熟的解析逻辑。

## 4. 总结

本项目的工具调用解析框架通过一个通用的基类 `BaseToolParser` 和特定于模型的子类（如 `Qwen3ToolParser`）相结合，实现了逻辑与配置的分离。这种设计不仅保证了代码的整洁和可维护性，也极大地简化了适配新模型的工作。开发者只需关注模型输出的特定标签格式，而无需关心背后复杂的解析和流式处理细节。
