"""
01_load_kb.py — 测试加载 FlexRAG 构建的知识库

验证 FAISS 索引文件能否被正常加载，并打印基本的索引信息。
不发起任何检索查询，仅确认知识库文件完整可用。

运行方式（从项目根目录执行）：
    PYTHONPATH=. python examples/01_load_kb.py

前置条件：
    1. 用 FlexRAG scripts/build_knowledge_base.py 构建好知识库
    2. 复制并填写 .env.example → .env，至少配置以下变量：
         KNOWLEDGE_PERSIST_DIR=./knowledge_base
         EMBEDDING_BASE_URL=http://10.x.x.x:8018/v1
         EMBEDDING_MODEL=bge-large-zh-v1.5
    3. 启动 Embedding 服务（供 FlexRAGRetriever 使用）
    4. pip install litellm python-dotenv llama-index llama-index-vector-stores-faiss faiss-cpu httpx
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from rag_module.retrieval import FlexRAGRetriever  # noqa: E402


def main() -> None:
    persist_dir = os.getenv("KNOWLEDGE_PERSIST_DIR", "./knowledge_base")

    print(f"{'='*60}")
    print(f"📂 知识库路径：{persist_dir}")
    print(f"{'='*60}")

    # 检查索引文件是否存在
    faiss_path = os.path.join(persist_dir, "faiss.index")
    if not os.path.exists(faiss_path):
        print(f"\n❌ 未找到 FAISS 索引文件：{faiss_path}")
        print("   请先执行 FlexRAG scripts/build_knowledge_base.py 构建知识库。")
        return

    print(f"\n✅ FAISS 索引文件已找到：{faiss_path}")

    # 加载知识库
    print("\n⏳ 正在加载知识库...")
    retriever = FlexRAGRetriever.from_env(persist_dir=persist_dir)

    # 打印索引基本信息
    print(f"\n✅ 知识库加载成功")
    print(f"   文档片段总数：{retriever.doc_count}")
    print(f"   Embedding 模型：{retriever.embedding_model}")
    print(f"   Embedding 服务：{retriever.embedding_base_url}")


if __name__ == "__main__":
    main()
