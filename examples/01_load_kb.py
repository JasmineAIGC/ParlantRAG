"""
01_load_kb.py — 测试加载示例知识库文档

新实现不再加载 FlexRAG / llama-index 持久化索引，而是将 JSON 语料转换为
``id/content/source`` 文档片段，交给 ``ParlantRetriever`` 使用。

运行方式：
    python examples/01_load_kb.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples._loader import load_hotpotqa_documents  # noqa: E402


def main() -> None:
    docs = load_hotpotqa_documents(max_examples=40)
    print("✅ 示例知识库加载完成")
    print(f"   文档片段数：{len(docs)}")
    if docs:
        first = docs[0]
        print(f"   首条 ID：{first['id']}")
        print(f"   首条来源：{first['source']}")
        print(f"   内容预览：{first['content'][:120]}...")


if __name__ == "__main__":
    main()
