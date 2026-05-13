"""
02_retrieval.py — Parlant SDK 原生检索流程演示

从 HotpotQA 示例 JSON 构建文档片段，使用新的 ``rag_module`` 路径和
``ParlantRetriever`` 执行检索，不再加载 FlexRAG / llama-index 持久化索引。

运行方式：
    python examples/02_retrieval.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag_module.retrieval import ParlantRetriever  # noqa: E402

QUERY = "what is one of the stars of  The Newcomers known for？"
TOP_K = 5
DATA_FILE = Path("knowledge_base/hotpotqa.json")
MAX_EXAMPLES = 40


def load_documents(path: Path = DATA_FILE, max_examples: int = MAX_EXAMPLES) -> list[dict]:
    """将 HotpotQA 示例数据转换为检索器使用的 ``id/content/source`` 文档列表。"""
    rows = json.loads(path.read_text(encoding="utf-8"))[:max_examples]
    docs: list[dict] = []
    for row in rows:
        row_id = row.get("_id", "item")
        for title, sentences in row.get("context", []):
            content = " ".join(sentences).strip()
            if content:
                docs.append({"id": f"{row_id}:{title}", "content": content, "source": title})
    return docs


async def main() -> None:
    documents = load_documents()
    # 默认使用无外部依赖的关键词检索；如需向量检索，可替换为 from_dense_env/documents。
    retriever = ParlantRetriever.from_keyword(documents)

    print(f"{'='*60}")
    print(f"🔍 检索查询：{QUERY}")
    print(f"📚 文档片段数：{len(documents)}")
    print(f"{'='*60}")

    result = await retriever.retrieve(query=QUERY, top_k=TOP_K)

    if not result.chunks:
        print("\n⚠️  未检索到任何相关片段（retrieval_status=no_results）。")
        return

    print(f"\n📄 命中 {len(result.chunks)} 个片段（共请求 top_k={TOP_K}）：\n")
    for i, chunk in enumerate(result.chunks, 1):
        preview = chunk.content[:120].replace("\n", " ")
        if len(chunk.content) > 120:
            preview += "..."
        print(f"  [{i}] 得分：{chunk.score:.4f}  来源：{chunk.source}")
        print(f"       {preview}\n")


if __name__ == "__main__":
    asyncio.run(main())
