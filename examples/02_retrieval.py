"""
02_retrieval.py — 测试检索能力

加载 FlexRAG 构建的 FAISS 知识库，执行查询检索，打印命中片段。
不调用 LLM，仅验证向量检索流程。

运行方式（从项目根目录 或 examples/ 目录均可）：
    python examples/02_retrieval.py
    # 或
    cd examples && python 02_retrieval.py

前置条件：
    1. 用 FlexRAG scripts/build_knowledge_base.py 构建好知识库
    2. 复制并填写 .env.example → .env，至少配置以下变量：
         KNOWLEDGE_PERSIST_DIR=./knowledge_base/knowledge_persist_dir/hotpotqa
         EMBEDDING_BASE_URL=http://10.x.x.x:8018/v1
         EMBEDDING_MODEL=bge-large-zh-v1.5
    3. 启动 Embedding 服务（供检索时编码查询词使用）
    4. pip install litellm python-dotenv llama-index llama-index-vector-stores-faiss faiss-cpu httpx
"""
from __future__ import annotations

import asyncio
import os
import sys

# 确保从任意目录运行时都能找到 rag_module（无需设置 PYTHONPATH）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag_module.retrieval import FlexRAGRetriever  # noqa: E402

# 修改为你的实际查询词
QUERY = "请问知识库中有哪些相关内容？"
TOP_K = 5


async def main() -> None:
    # 加载知识库（所有配置从 .env 读取）
    print("⏳ 正在加载知识库...")
    retriever = FlexRAGRetriever.from_env()
    print("✅ 知识库加载完成\n")

    # 执行检索
    print(f"{'='*60}")
    print(f"🔍 检索查询：{QUERY}")
    print(f"{'='*60}")

    result = await retriever.retrieve(query=QUERY, top_k=TOP_K)

    if not result.chunks:
        print("\n⚠️  未检索到任何相关片段（retrieval_status=no_results）。")
        print("   请检查查询词与知识库内容是否匹配，或调整 top_k 参数。")
        return

    print(f"\n📄 命中 {len(result.chunks)} 个片段（共请求 top_k={TOP_K}）：\n")
    for i, chunk in enumerate(result.chunks, 1):
        preview = chunk.content[:120].replace("\n", " ")
        if len(chunk.content) > 120:
            preview += "..."
        print(f"  [{i}] 相似度得分：{chunk.score:.4f}  来源：{chunk.source}")
        print(f"       {preview}\n")


if __name__ == "__main__":
    asyncio.run(main())
